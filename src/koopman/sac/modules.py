"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Feb.24th 2025
@func: 
| the modules for SAC algorithm
"""

import torch
import pickle
import numpy as np
import torch.nn as nn
from src.nets.kpm_lqr import KoopmanLQR
from nets.diffusion_policy.vision_encoder import get_resnet, replace_bn_with_gn
from env_ms.env_utils import from_array_to_tensor

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

######################################################## ENCODER ########################################################
class SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        ### vision encoder
        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        self.trunk = get_resnet('resnet18')

        ### IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        self.trunk = replace_bn_with_gn(self.trunk)
    
    def forward(self, obs_seq, is_detach=False):
        B, N = obs_seq.shape[0], obs_seq.shape[1]               # 
        obs_seq = self.trunk(obs_seq.flatten(end_dim=1))        # [B, 2, 3, 128, 128] -> [B*2, 3, 128, 128] -> [B*2, 512]
        if is_detach:
            obs_seq = obs_seq.detach()
        obs_seq = obs_seq.reshape(B, -1)                        # [B*2, 512] -> [B, 2*512]
        return obs_seq                                          # [B, 512]
    
    def copy_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        num_layer = 4
        num_conv_each_layer = 2
        tie_weights(src=source.trunk.conv1, trg=self.trunk.conv1)
        for i in range(num_layer):
            layer_src = getattr(source.trunk, f'layer{i+1}')
            layer_trg = getattr(self.trunk, f'layer{i+1}')
            for j in range(num_conv_each_layer):
                tie_weights(src=layer_src[j].conv1, trg=layer_trg[j].conv1)
                tie_weights(src=layer_src[j].conv2, trg=layer_trg[j].conv2)    

######################################################## CRITIC ########################################################
class Qfunction(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )
    
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.encoder = SeqEncoder() # NOTE: the encoder is for image observation encoding

        self.Q1 = Qfunction(env)
        self.Q2 = Qfunction(env)

    def forward(self, obs, act, is_detach=False):
        obs = self.encoder(obs, is_detach)
        q1 = self.Q1(obs, act)
        q2 = self.Q2(obs, act)
        return q1,q2
    
    def grad_clip(self, max_grad_norm):
        Q1_grad_norm = nn.utils.clip_grad_norm_(self.Q1.parameters(), max_grad_norm)
        Q2_grad_norm = nn.utils.clip_grad_norm_(self.Q2.parameters(), max_grad_norm)
        return Q1_grad_norm, Q2_grad_norm

######################################################## KPM-ACTOR ########################################################
class KPMActor(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        # initialization
        self.obs_dim = np.array(env.single_observation_space.shape).prod()                            # NOTE: only current obs as input
        self.act_dim = np.array(env.single_action_space.shape).prod()                                 # NOTE: action combination for next act-horizon 

        # encoder
        self.encoder = SeqEncoder()                                                                   # NOTE: the encoder is for image observation encoding

        # model backbone
        self.backbone = KoopmanLQR(k=self.obs_dim, 
                                   T=5, 
                                   g_dim=self.obs_dim, 
                                   u_dim=self.act_dim, 
                                   g_goal=None, 
                                   u_affine=None)
        
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))  # from [-1,1] (tanh) to specific action space
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))   # from [-1,1] (tanh) to specific action space

        # the exploration limit
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.log_std_init = torch.nn.Parameter(torch.Tensor([1.0]).log()) # NOTE: fixed here -> zero

        # args definition
        self.args = args

        # 🌊 set up goal reference for different situations 🌊
        goal_image_path = self.args.koopman_goal_image_path
        if isinstance(goal_image_path, str) and goal_image_path.endswith(".pkl"):
            with open(goal_image_path, "rb") as f:
                self.goal_obs = from_array_to_tensor(pickle.load(f)).to(self.args.device)   # NOTE: [2,3,H,W] -> `tensor` on the `device`
                self.goal_obs = self.goal_obs.unsqueeze(0)                                  # NOTE: [2,3,H,W] -> [1,2,3,H,W]
        else:
            self.goal_obs = None # NOTE: another choice for no goals here

    def _kpm_forward(self, x, is_detach):
        # 🌊 encode the goal images to be used in self.backbone 🌊
        if self.goal_obs is None:
            self.backbone._g_goal = torch.zeros((1, self.obs_dim)).squeeze(0).to(self.args.device) # [obs_dim,] | when no goals
        else:
            goal_obs = self.encoder(self.goal_obs, is_detach=is_detach)
            self.backbone._g_goal = goal_obs.squeeze(0)         # [B,obs_dim] -> [obs_dim,]
        # 🌊 LQR directly gives mu; Utilize constant log_std 🌊
        broadcast_shape = list(x.shape[:-1]) + [self.act_dim]   # [B,act_dim]
        mean = self.backbone(x)
        log_std = self.log_std_init + torch.zeros(*broadcast_shape).to(self.args.device)
        return mean, log_std

    def forward(self, x, is_detach=False):
        x = self.encoder(x, is_detach)          # NOTE: encode from image to feature map
        mean, log_std = self._kpm_forward(x, is_detach)    # NOTE: encode to get mean and log_std
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)  # From SpinUp / Denis Yarats | TODO: why should we shift the log_std ??? -> define the exploration 
        return mean, log_std

    def get_eval_action(self, x, is_detach=False):
        x = self.encoder(x, is_detach)          # NOTE: encode from image to feature map
        mean, _ = self._kpm_forward(x, is_detach)          # NOTE: encode to get mean and log_std
        action = torch.tanh(mean) * self.action_scale + self.action_bias # [-1,1] to the [specific single_action_space]
        return action

    def get_action(self, x, is_detach=False):
        # initialization
        mean, log_std = self(x, is_detach)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                              # NOTE: for reparameterization trick (mean + std * N(0,1)) 
        y_t = torch.tanh(x_t)                               # to [-1,1]
        action = y_t * self.action_scale + self.action_bias # [-1,1] shift to [single_action_space]
        log_prob = normal.log_prob(x_t)                     # log-probability
        # enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # log(y)=log(x)-log(1/(1-y^2))
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias # [-1,1] shift to [single_action_space]
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


if __name__ == "__main__":
    pass


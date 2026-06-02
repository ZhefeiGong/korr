"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Feb.25th 2025
@func: 
| the executive agent for SAC
"""

import os
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from nets.kpm_diffusion_policy_soft_actor_critic.modules import Critic, KPMActor

def is_ms1_env(env_id):
    """inherited from Policy-Decorator"""
    return 'OpenCabinet' in env_id or 'MoveBucket' in env_id or 'PushChair' in env_id

def from_array_to_tensor(array_seq):
    # NOTE: check uint8
    if array_seq.dtype == np.uint8:
        array_seq_tensor = torch.from_numpy(array_seq).to(torch.float32) / 255.0      # NOTE: from 0~255 to 0~1 and float32
    elif array_seq.dtype == torch.uint8:
        array_seq_tensor = array_seq.to(torch.float32) / 255.0                        # NOTE: from 0~255 to 0~1 and float32
    else:
        raise ValueError(f"Invalid dtype for 'array_seq': Expected 'uint8'.")
    return array_seq_tensor

######################################################## SAC AGENT ########################################################

class KPMAgentSAC(object):
    def __init__(self, env, args):
        
        # initialize actor-critic
        self.actor = KPMActor(env, args).to(args.device) # NOTE: the actor of Koopman model
        if args.critic_input == 'concat': env.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.single_action_space.shape[0] * 2,)) # NOTE: special for critic input concat
        self.critic = Critic(env).to(args.device)
        if is_ms1_env(args.env_id):
            for m in list(self.actor.modules()) + list(self.critic.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=1) # NOTE: weight -> `xavier uniform` for training stability
                    torch.nn.init.zeros_(m.bias)                    # NOTE: bias -> 0

        # initialize the weight of the actor and critic 
        self.critic_target = Critic(env).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict()) 
        
        # tie encoders between actor and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder) # NOTE: what they did in traditional SAC + CURL

        # alpha
        self.target_entropy = -torch.prod(torch.Tensor(env.single_action_space.shape).to(args.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=args.device)

        # optimziers
        self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=args.critic_lr)
        self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=args.residual_policy_lr)  
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=args.critic_lr)
        self.optimizer_kpm = optim.Adam(self.actor.backbone.parameters(), lr=args.residual_policy_lr) # NOTE: update the Koopman model
        
        # args storage
        self.args = args
        self.log_args = dict()

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def train(self, is_training=True):
        self.training = is_training
        self.actor.train(is_training)
        self.critic.train(is_training)
        self.critic_target.train(is_training)
    
    def _retrieve_actions(self, res_act, base_act):
        if self.args.critic_input == 'res':
            actions = res_act
        elif self.args.critic_input == 'sum':
            scaled_res_act = self.args.res_scale * res_act
            actions = base_act + scaled_res_act
        elif self.args.critic_input == 'concat':
            actions = torch.cat([res_act, base_act], dim=1)
        else:
            raise ValueError(f"Invalid value for 'critic_input': {self.args.critic_input}. Expected 'res', 'sum', or 'concat'.")
        return actions
    
    def update_critic(self, data, action_dict):
        
        # calculate the target q values
        with torch.no_grad():
            next_res_actions, next_log_pi, _ = self.actor.get_action(from_array_to_tensor(data.next_observations).to(self.args.device))
            next_actions = self._retrieve_actions(next_res_actions, action_dict['base_next'])
            q1_next_value, q2_next_value = self.critic_target(from_array_to_tensor(data.next_observations).to(self.args.device), next_actions)
            min_q_next_value = torch.min(q1_next_value, q2_next_value) - self.alpha * next_log_pi
            q_next_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_q_next_value).view(-1)
            # NOTE: "data.dones" is "stop_bootstrap", which is computed earlier according to "args.bootstrap_at_done"
        
        # calculate the current q values
        cur_actions = self._retrieve_actions(action_dict['residual'], action_dict['base'])
        q1_cur_value, q2_cur_value = self.critic(from_array_to_tensor(data.observations).to(self.args.device), cur_actions, is_detach=False) # NOTE: only update encoder here
        Q1_loss = F.mse_loss(q1_cur_value.view(-1), q_next_value)
        Q2_loss = F.mse_loss(q2_cur_value.view(-1), q_next_value)
        critic_loss =  Q1_loss + Q2_loss
        
        # optimize
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        Q1_grad_norm, Q2_grad_norm = self.critic.grad_clip(self.args.max_grad_norm)
        self.optimizer_critic.step()

        # store the log parameters
        self.log_args['q1_value'] = q1_cur_value.mean().item()
        self.log_args['q2_value'] = q2_cur_value.mean().item()
        self.log_args['Q1_loss'] = Q1_loss.item()
        self.log_args['Q2_loss'] = Q1_loss.item()
        self.log_args['critic_loss'] = critic_loss.item()
        self.log_args['Q1_grad_norm'] = Q1_grad_norm.item()
        self.log_args['Q2_grad_norm'] = Q2_grad_norm.item()
    
    def update_actor(self, data, action_dict):
        
        # loss computation
        residual_action, log_pi, _ = self.actor.get_action(from_array_to_tensor(data.observations).to(self.args.device), is_detach=True) # NOTE: no need to update encoder
        cur_actions = self._retrieve_actions(residual_action, action_dict['base'])
        q1_value, q2_value = self.critic(from_array_to_tensor(data.observations).to(self.args.device), cur_actions, is_detach=True) # NOTE: no need to update encoder
        min_q_value = torch.min(q1_value, q2_value)
        actor_loss = ((self.alpha * log_pi) - min_q_value).mean()
        
        # optimize actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.optimizer_actor.step()

        # store the log parameters
        self.log_args['actor_loss'] = actor_loss.item()
        self.log_args['actor_grad_norm'] = actor_grad_norm.item()

    def update_alpha(self, data):
        
        # compute log_pi
        with torch.no_grad():
            _, log_pi, _ = self.actor.get_action(from_array_to_tensor(data.observations).to(self.args.device), is_detach=True) # NOTE: no need to update encoder
        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
        
        # optimize
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        # store the log parameters
        self.log_args['alpha'] = self.alpha
        self.log_args['alpha_loss'] = alpha_loss.item()

    def update_kpm(self, data, action_dict):
        # calculate KPM process
        obs_cur = self.actor.encoder(from_array_to_tensor(data.observations).to(self.args.device), is_detach=False) # NOTE: need update the weight of the encoder
        obs_nxt = self.actor.encoder(from_array_to_tensor(data.next_observations).to(self.args.device), is_detach=False) # NOTE: need update the weight of the encoder
        act_cur = self._retrieve_actions(action_dict['residual'], action_dict['base'])
        obs_nxt_pred = self.actor.backbone._predict_koopman(obs_cur, act_cur)

        # loss 
        kpm_loss = F.mse_loss(obs_nxt_pred, obs_nxt)
        
        # optimize
        self.optimizer_kpm.zero_grad()
        kpm_loss.backward()
        kpm_grad_norm = nn.utils.clip_grad_norm_(self.actor.backbone.parameters(), self.args.max_grad_norm)
        self.optimizer_kpm.step()

        # store the log parameters
        self.log_args['kpm_loss'] = kpm_loss.item()
        self.log_args['kpm_grad_norm'] = kpm_grad_norm.item()

    def update_critic_target(self):
        for param, target_param in zip(self.critic.Q1.parameters(), self.critic_target.Q1.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.critic.Q2.parameters(), self.critic_target.Q2.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def update_log(self, writer, global_step):
        for key, value in self.log_args.items():
            writer.add_scalar(f"sac/{key}", value, global_step)
    
    def save_ckpt(self, global_step):
        os.makedirs(f'{self.args.log_path}/checkpoints', exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha 
        }, f'{self.args.log_path}/checkpoints/{global_step}.pt')

if __name__ == "__main__":
    pass



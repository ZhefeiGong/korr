"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Mar.19th 2025
@func: 
| the modules for SAC algorithm
"""

import torch
import numpy as np
import torch.nn as nn
from src.koopman.ppo_dir.kpm_lqr import KoopmanLQR

def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu") # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)
    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


######################################################## CRITIC ########################################################
class KPMCritic(nn.Module):
    def __init__(self,
                 obs_dim,
                 critic_hidden_size=512,
                 critic_num_layers=2,
                 critic_activation="SiLU",
                 critic_last_layer_activation=None,
                 critic_last_layer_std=1.0,
                 bias_on_last_layer=True,
                 critic_last_layer_bias_const=0.0,
                 ):
        super().__init__()

        # initialization
        self.input_dim = obs_dim
        self.output_dim = 1 # NOTE: only one value
        self.output_std = critic_last_layer_std
        self.activation = critic_activation
        self.hidden_sizes = [critic_hidden_size] * critic_num_layers
        self.last_layer_bias_const = critic_last_layer_bias_const

        ### former model structure

        # first layer
        act_func = getattr(nn, self.activation)
        layers = []
        layers.append(
            layer_init(
                nn.Linear(self.input_dim, 
                          self.hidden_sizes[0]), 
                nonlinearity=critic_activation
            )
        )
        layers.append(act_func())

        # mid layers
        for i in range(1, len(self.hidden_sizes)):
            layers.append(
                layer_init(
                    nn.Linear(self.hidden_sizes[i - 1], 
                              self.hidden_sizes[i]), 
                    nonlinearity=self.activation
                )
            )
            layers.append(act_func())
    
        # last layer
        layers.append(
            layer_init(
                nn.Linear(self.hidden_sizes[-1], 
                          self.output_dim, 
                          bias=bias_on_last_layer),
                std=self.output_std,
                nonlinearity="Tanh",
                bias_const=self.last_layer_bias_const,
            )
        )
        self.backbone = nn.Sequential(*layers)

        # whether to add an activation for the last layer
        if critic_last_layer_activation is not None:
            self.backbone.add_module(
                "output_activation",
                getattr(nn, critic_last_layer_activation)(),
            )
            print(self.backbone)
    
    def forward(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.backbone(nobs)


######################################################## KPM-ACTOR ########################################################
class KPMActor(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 obs_lift_dim,
                 learn_std=False,
                 init_logstd=-3,
                 actor_hidden_size=512,
                 actor_num_layers=2,
                 actor_activation="SiLU",
                 action_head_std=0.01,
                 bias_on_last_layer=False
                 ):
        super().__init__()

        # initialization
        self.obs_dim = obs_dim
        self.obs_lift_dim = obs_lift_dim
        self.act_dim = act_dim

        ### Lift Function
        # first layer
        self.hidden_sizes = [actor_hidden_size] * actor_num_layers
        act_func = getattr(nn, actor_activation)
        layers = []
        layers.append(layer_init(nn.Linear(self.obs_dim, self.hidden_sizes[0]), nonlinearity=actor_activation))
        layers.append(act_func())
        # mid layers
        for i in range(1, len(self.hidden_sizes)):
            layers.append(layer_init(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]), nonlinearity=actor_activation))
            layers.append(act_func())
        # last layer
        layers.append(layer_init(nn.Linear(self.hidden_sizes[-1], self.obs_lift_dim, bias=bias_on_last_layer), std=action_head_std, nonlinearity="Tanh"))
        self.lifttransform = nn.Sequential(*layers)

        ### model backbone
        self.backbone = KoopmanLQR(T=5, 
                                   g_dim=self.obs_lift_dim, 
                                   u_dim=self.act_dim, 
                                   g_goal=None, 
                                   g_affine=None,
                                   u_affine=None)
        
        ### standard deviation
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.act_dim) * init_logstd,
            requires_grad=learn_std,
        )

        # without goals
        self.goal_obs = None # another choice for no goals here
    
    def set_goal(self, nobs):
        assert nobs.shape[-1] == self.obs_dim, "goal_obs mismatch obs_dim"
        self.goal_obs = nobs # [1, obs_dim,]
    
    def kpm_loss(self, 
                 nobs: torch.Tensor, 
                 nact: torch.Tensor,
                 next_nobs: torch.Tensor):
        nobs = self.lifttransform(nobs)
        next_nobs = self.lifttransform(next_nobs)
        next_nobs_pred = self.backbone._predict_koopman(nobs, nact) # calculate KPM process
        kpm_loss = torch.nn.functional.mse_loss(next_nobs_pred, next_nobs) # loss
        return kpm_loss
    
    def forward(self, nobs: torch.Tensor):
        goal = self.lifttransform(self.goal_obs)[-1] # [obs_lift_dim,]
        nobs = self.lifttransform(nobs) # [N,obs_lift_dim]
        self.backbone.set_goal(goal) # set the goal here
        action_mean: torch.Tensor = self.backbone(nobs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd


if __name__ == "__main__":
    pass


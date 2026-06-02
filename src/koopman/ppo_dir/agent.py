"""
@author: 
| copyright @ Zhefei(Jeffrey) Gong
@date: 
| Mar.19th 2025
@func: 
| the executive agent for SAC
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.dataset.normalizer import LinearNormalizer
from src.models.utils import PrintParamCountMixin
from torch.distributions import Normal
from src.koopman.ppo_dir.modules import KPMCritic, KPMActor

class KPMResidualPolicy(nn.Module, PrintParamCountMixin):
    def __init__(
        self,
        obs_shape,
        action_shape,
        obs_lift_dim=256,
        actor_hidden_size=512,
        actor_num_layers=2,
        critic_hidden_size=512,
        critic_num_layers=2,
        actor_activation="SiLU",
        critic_activation="SiLU",
        init_logstd=-3,
        action_head_std=0.01,
        action_scale=0.1,
        critic_last_layer_bias_const=0.0,
        critic_last_layer_std=1.0,
        critic_last_layer_activation=None, # default None (offl.)
        learn_std=False,
        **kwargs,
    ):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
            actor_hidden_sizes: list of hidden layer sizes for the actor network
            critic_hidden_sizes: list of hidden layer sizes for the critic network
            activation: activation function to use (e.g., nn.ReLU, nn.Tanh)
        """
        super().__init__()
        
        self.action_dim = action_shape[-1]
        self.action_scale = action_scale
        self.obs_dim = np.prod(obs_shape) # condition only on `observation`
        self.obs_lift_dim = obs_lift_dim # the dimension of the observation after lifting process
        
        # actor
        self.actor = KPMActor(
            obs_dim=self.obs_dim,
            act_dim=self.action_dim,
            obs_lift_dim=self.obs_lift_dim,
            learn_std=learn_std,
            init_logstd=init_logstd,
            actor_hidden_size=actor_hidden_size,
            actor_num_layers=actor_num_layers,
            actor_activation=actor_activation,
            action_head_std=action_head_std, # the std for the last-layer initialization
            bias_on_last_layer=False,
            )
        
        # critic
        self.critic = KPMCritic(
            obs_dim=self.obs_dim,
            critic_hidden_size=critic_hidden_size,
            critic_num_layers=critic_num_layers,
            critic_activation=critic_activation,
            critic_last_layer_activation=critic_last_layer_activation, 
            critic_last_layer_std=critic_last_layer_std, # the std for the last-layer initialization
            bias_on_last_layer=True,
            critic_last_layer_bias_const=critic_last_layer_bias_const,
            )
        
        # init other params
        self.actor_logstd = self.actor.actor_logstd
        self.normalizer = None
        
        self.print_model_params()

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(
            self,
            nobs: torch.Tensor,
            action: torch.Tensor = None,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_mean, action_logstd = self.actor(nobs)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(nobs), # NOTE: values from critic
            action_mean,
        )

    def get_action(self, nobs: torch.Tensor) -> torch.Tensor:
        action_mean, _ = self.actor(nobs)
        return action_mean * self.action_scale

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(normalizer.state_dict())

    def bc_loss(self, 
                res_nobs: torch.Tensor, 
                gt_res_action: torch.Tensor) -> torch.Tensor:
        """
        Compute the behavior cloning loss for the policy

        Args:
            res_nobs: the observation tensor, i.e., state + base action
            gt_res_action: the action tensor, i.e., the ground truth residual

        the gt_res_action needs to be scaled by self.action_scale before passing it in
        """
        action_mean, _ = self.actor(res_nobs)
        gt_res_action_scaled = gt_res_action / self.action_scale
        return torch.nn.functional.mse_loss(action_mean, gt_res_action_scaled)

if __name__ == "__main__":
    pass



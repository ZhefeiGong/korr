import argparse
import os
import sys
from pathlib import Path
from src.gym import get_rl_env
from gymnasium import Env # NOTE: needs to be before torch imports
import torch # NOTE: needs to be after isaac gym imports
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.behavior.base import Actor  # noqa
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from typing import Any, List, Optional
from ipdb import set_trace as bp  # noqa
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.eval.rollout_rl import calculate_success_rate
from src.visualization.render_mp4 import unpickle_data
from src.data_processing.utils import filter_and_concat_robot_state
from src.common.geometry import np_action_6d_to_quat, np_action_quat_to_6d_rotation, isaac_quat_to_rot_6d
from src.common.geometry import isaac_quat_to_pytorch3d_quat, pytorch3d_quat_to_isaac_quat
import copy
from scipy.spatial.transform import Rotation as R

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-path", 
                        type=str, 
                        default="/path/to/outputs/one_leg/state_res_rl/low/kpm/05-42-36/models/res_kpm_ppo_state_dp_wo_lqr/actor_chkpt_best_success_rate.pt") # NOTE: retrieve ckpt from local
    parser.add_argument("--base-wt-path", 
                        type=str, 
                        default="/path/to/outputs/one_leg/state/low/06-11-24.326459/models/berry-bun-1/actor_chkpt_best_success_rate.pt") # NOTE: retrieve ckpt from local
    parser.add_argument("--pkl-path",
                        type=str,
                        default="/path/to/data/raw/raw/diffik/sim/one_leg/rollout/low/success/2025-03-28T10:28:39.925320.pkl")
    parser.add_argument("--pkl-path-ref",
                        type=str,
                        default="/path/to/data/raw/raw/diffik/sim/one_leg/rollout/low/obs.pkl")
    parser.add_argument("--gpu", 
                        type=int, 
                        default=0)
    parser.add_argument("--observation-space", 
                        choices=["image", "state"], 
                        default="state") # NOTE: prefer "image" for video rollout
    args = parser.parse_args()
    return args

def merge_base_config_with_root_config(base_cfg: DictConfig, cfg: DictConfig):
    if "residual_policy" in cfg.actor:
        OmegaConf.update(base_cfg.actor, "residual_policy", cfg.actor.residual_policy, merge=True)
    if "critic" in cfg:
        OmegaConf.update(base_cfg.actor, "critic", cfg.critic, merge=True)
        OmegaConf.update(base_cfg.actor, "init_logstd", cfg.init_logstd, merge=True)
    return base_cfg

def convert_to_tensor(data):
    if isinstance(data, dict):  
        return {key: convert_to_tensor(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_tensor(item) for item in data]
    elif isinstance(data, np.ndarray):  
        return torch.from_numpy(data)
    elif isinstance(data, (int, float)):
        return torch.tensor(data, dtype=torch.float32)
    else:  
        return data

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data

# def r2_score_numpy(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares (SSR)
#     ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)  # Total sum of squares (SST)
#     return 1 - ss_res / ss_tot  # Compute R² score

# def check_kpm_const(Z, Z_next):
#     K = np.linalg.pinv(Z) @ Z_next  # Compute the Koopman approximation matrix K
#     Z_pred = Z @ K  # Predict the next state using K
#     r2 = r2_score_numpy(Z_next, Z_pred)  # Compute the R² score
#     print("Koopman's R²:", r2)

def from_quat_rel_to_6d_abs(actions, observations):
    # Recover the absolute position
    abs_pos = np.array([rs['robot_state']["ee_pos"] for rs in observations[:-1]]) + actions[:, :3] # Until the last timestep
    # Recover the absolute quaternion
    delta_quat = R.from_quat(actions[:, 3:7]) # from_quat is (x, y, z, w) by defaut - same with IsaacGym Design
    # Get the position quat from the robot state
    pos_quat = R.from_quat([rs['robot_state']["ee_quat"] for rs in observations[:-1]]) # Until the last timestep
    # Recover absolute quaternion
    abs_quat = pos_quat * delta_quat
    # Construct the recovered actions
    recovered_actions = np.concatenate([abs_pos, abs_quat.as_quat(), actions[:, -1:]], axis=1)  # Shape (N, 10)
    # Convert quaternion back to 6D rotation representation
    recovered_actions = np_action_quat_to_6d_rotation(recovered_actions)
    # Return
    return recovered_actions

def from_6d_abs_to_quat_rel(actions, observations):
    # Change to quaternion
    actions = np_action_6d_to_quat(actions)
    # Get the action quat
    abs_quat = R.from_quat(actions[:, 3:7])
    # Get the position quat from the robot state
    pos_quat = R.from_quat([rs['robot_state']["ee_quat"] for rs in observations[:-1]])
    # Calculate the delta quat between the pos_quat and the action_quat
    delta_quat = pos_quat.inv() * abs_quat
    # Calculate the delta position
    delta_pos = actions[:, :3] - np.array([rs['robot_state']["ee_pos"] for rs in observations[:-1]])
    # Insert the delta quat into the actions
    converted_actions = np.concatenate([delta_pos, delta_quat.as_quat(), actions[:, -1:]], axis=1)
    return converted_actions


if __name__ == "__main__":
    
    # Parse the arguments
    args = get_args()
    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ### Load Model
    # Get the residual root config
    if not args.wt_path: raise KeyError("Undefined weight path (wt_path).")
    wts = torch.load(args.wt_path)
    cfg: DictConfig = OmegaConf.create(wts['config'])
    # Get the base config
    if not args.base_wt_path: raise KeyError("Undefined weight path (base_wt_path).")
    base_wts = torch.load(args.base_wt_path)
    base_cfg: DictConfig = OmegaConf.create(base_wts['config'])
    merge_base_config_with_root_config(base_cfg, cfg)
    # Assign seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    # Load agent
    if cfg.base_policy.actor.name == "diffusion": # ResidualDiffusionPolicy inherited from DiffusionPolicy
        agent = ResidualDiffusionPolicy(device, base_cfg) # DP + Residual RL
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg) # MLP + Residual RL
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")
    agent.load_state_dict(wts['model_state_dict'])
    agent.eval()
    agent.to(device)

    pickle_data = unpickle_data(args.pkl_path)
    observations = pickle_data['observations']

    ### Load observation Data
    nor_observations = []
    for observation in observations: 
        obs = {'robot_state': observation['robot_state'],
               'parts_poses': observation['parts_poses']}
        obs = convert_to_tensor(obs)
        obs['robot_state'] = filter_and_concat_robot_state(obs['robot_state']).unsqueeze(0)# [1, dim_robot_state]
        obs['parts_poses'] = obs['parts_poses'].unsqueeze(0) # [1, dim_parts_poses]
        obs = move_to_device(obs, device)
        nor_obs = agent.process_obs(obs) # normalize the observations
        nor_observations.append(nor_obs)
    nor_observations = torch.cat(nor_observations, dim=0)

    ### Observation
    nor_states_cur = copy.deepcopy(nor_observations[:-1])
    nor_states_next = copy.deepcopy(nor_observations[1:])

    ### Action
    actions = np.array([act for act in pickle_data['actions']])
    un_nor_actions = np.float32(from_quat_rel_to_6d_abs(actions, observations))
    un_nor_actions = move_to_device(convert_to_tensor(un_nor_actions), device) # to tensor and gpu
    nor_actions = agent.normalizer(un_nor_actions, "action", forward=True) # NOTE: unnormalize the actions to executable actions

    ### Calculate
    with torch.no_grad():
        kpm_loss = agent.residual_policy.actor.kpm_loss(nobs=nor_states_cur, nact=nor_actions, next_nobs=nor_states_next)
    print('Koopman Loss :', kpm_loss.item())

    ### Comparison
    pickle_data_ref = unpickle_data(args.pkl_path_ref)
    pickle_data_ref = move_to_device(convert_to_tensor(pickle_data_ref), device)
    print('nobs :', torch.nn.functional.mse_loss(nor_states_cur, pickle_data_ref['nobs']).item())
    print('next_nobs :', torch.nn.functional.mse_loss(nor_states_next, pickle_data_ref['next_nobs']).item())
    print('nact :', torch.nn.functional.mse_loss(nor_actions, pickle_data_ref['nact']).item())

    ### Test r6d to quat to r6d
    nact = unpickle_data(args.pkl_path_ref)['nact']
    nact_quat = np.float32(from_6d_abs_to_quat_rel(nact, pickle_data['observations']))
    nact_6d = np.float32(from_quat_rel_to_6d_abs(nact_quat, pickle_data['observations']))
    nact = move_to_device(convert_to_tensor(nact), device)
    nact_6d = move_to_device(convert_to_tensor(nact_6d), device)
    print('recon nact :', torch.nn.functional.mse_loss(nact_6d, nact).item())




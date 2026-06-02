import os
import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from src.visualization.render_mp4 import unpickle_data, pickle_data
from src.behavior.base import Actor
from src.eval.evaluate_model import LocalCheckpointWrapper
from src.behavior import get_actor
from src.behavior.diffusion import DiffusionPolicy
from src.data_processing.utils import filter_and_concat_robot_state
from src.common.geometry import proprioceptive_quat_to_6d_rotation

def print_nested_dict(d, indent=0):
    """ recursively print the dictionary hierarchy """

    for key, value in d.items():
        print("  " * indent + f"- {key}: {type(value)}")
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)

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

def gen_goal_from_pickle(pickle_path, save_path):
    """ fetch the last frame of the pickle """

    # load data
    data = unpickle_data(pickle_path)
    print("Data type:", type(data))
    if isinstance(data, dict):
        print("Pickle file contains a dictionary with the following structure:")
        print_nested_dict(data) # NOTE: visualize the pickle
    else:
        print("Pickle file contains:", data)

    # retrieve data
    observations = data['observations'] # NOTE: obs is slightly larger than act and rwd, cause it contains the one more reset obs (initial obs)
    rewards = data['rewards'] # NOTE: rewards are corresponding to actions
    assert len(rewards)+1 == len(observations), "mismatch the length between observations and rewards"
    suc_idx_1st = np.where(np.array(rewards) == 1.0)[0][-1] # NOTE: find the first timestep to satisfy the requirement -> one_leg-1(first) | lamp-2(last) | round_table-2(last)
    goal_pickle = {
        "observation": observations[suc_idx_1st+1],
        "reward": rewards[suc_idx_1st],
        "success": data['success'],
        "task": data['task'],
        "action_type": data['action_type'],
    }
    save_path_goal = os.path.join(save_path, "goal.pkl")
    pickle_data(goal_pickle, save_path_goal)
    print(f'save to {save_path_goal}')

    # save image for visualization
    Image.fromarray(goal_pickle['observation']['color_image1']).save(os.path.join(save_path, "color_image1.png"))
    Image.fromarray(goal_pickle['observation']['color_image2']).save(os.path.join(save_path, "color_image2.png"))

    # back
    return

def process_obs(obs, actor):
    # Robot state is [pos, ori_quat, pos_vel, ori_vel, gripper]
    robot_state = obs["robot_state"] # NOTE: robot-state | 14(one_leg)
    # Parts poses is [pos, ori_quat] for each part
    parts_poses = obs["parts_poses"] # NOTE: parts-pose | 42(one_leg)
    # Make the robot state have 6D proprioception
    if robot_state.shape[-1] == 14:
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)
    robot_state = actor.normalizer(robot_state, "robot_state", forward=True)
    parts_poses = actor.normalizer(parts_poses, "parts_poses", forward=True)
    obs = torch.cat([robot_state, parts_poses], dim=-1)
    # Clamp the observation to be bounded to [-5, 5]
    obs = torch.clamp(obs, -3, 3)
    return obs

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

def run_goal_from_timestep(wt_path, save_path, device='cuda'):

    ##### Load Corresponding Model
    run = LocalCheckpointWrapper(wt_path)
    cfg = OmegaConf.create(run.config)
    assert cfg.control.control_mode == 'pos'
    # Temporary fix for residual missing field
    if "base_policy" in cfg:
        print("Applying residual field hotfix")
        cfg.action_dim = cfg.base_policy.action_dim
    # Temporary fix for dagger missing field
    if "student_policy" in cfg:
        print("Applying dagger field hotfix")
        cfg.action_dim = cfg.student_policy.action_dim
    # Temporary fix for critic missing field in actor config
    if "critic" in cfg:
        print("Applying critic field hotfix")
        cfg.actor.critic = cfg.critic
        cfg.actor.init_logstd = cfg.init_logstd
        cfg.discount = cfg.base_policy.discount
    # Activate the actor here
    actor: Actor = get_actor(cfg=cfg, device=device)
    # Set the inference steps of the actor
    if isinstance(actor, DiffusionPolicy):
        actor.inference_steps = 4
    actor.load_state_dict(run.checkpoint["model_state_dict"]) # NOTE: load from local ckpt
    actor.eval()
    actor.to(device)

    ##### Load
    obs = unpickle_data(os.path.join(save_path, 'goal.pkl'))['observation'] # only need observation
    obs = convert_to_tensor(obs) # from array to tensor
    obs['robot_state'] = filter_and_concat_robot_state(obs['robot_state']).unsqueeze(0) # [1, dim_robot_state]
    obs['parts_poses'] = obs['parts_poses'].unsqueeze(0) # [1, dim_parts_poses]
    obs = move_to_device(obs, device)

    ##### Run
    base_action = actor.action(obs) # executable action (unnormalized action)
    base_nobs = {'robot_state': obs["robot_state"].cpu().numpy(), 
                 'parts_poses': obs["parts_poses"].cpu().numpy()} # executable obs (unnormalized obs)
    goal_pickle = {
        "nobs": base_nobs, # unnormalized observation
        "nact": base_action.cpu().numpy(), # unnormalized base action
    }
    save_path_res_goal = os.path.join(save_path, "goal_obs.pkl")
    pickle_data(goal_pickle, save_path_res_goal)
    print(f'save to {save_path_res_goal}')


if __name__ == "__main__":

    pickle_path = "/path/to/saved/trajectory/2025-03-24T22:56:17.412972.pkl"
    wt_path = "/path/to/actor_chkpt_best_success_rate.pt"
    save_path = "/path/to/save/path/data/kpm_goals/round_table"
    gen_goal_from_pickle(pickle_path, save_path) # build the goal pickle
    run_goal_from_timestep(wt_path, save_path) # build the pickle for residual observation



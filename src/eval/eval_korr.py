import argparse
import os
import sys
import copy
import json
import random
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
from src.behavior.residual_carp import ResidualCARP
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.eval.rollout_korr import calculate_success_rate
from src.eval.eval_utils import replace_path_in_dict
from src.visualization.render_mp4 import unpickle_data


def convert_to_tensor(data, device):
    if isinstance(data, dict):  
        return {key: convert_to_tensor(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_tensor(item, device) for item in data]
    elif isinstance(data, np.ndarray):  
        return torch.from_numpy(data).to(device)
    elif isinstance(data, (int, float)):
        return torch.tensor(data, dtype=torch.float32).to(device)
    else:  
        return data

def merge_base_config_with_root_config(base_cfg: DictConfig, cfg: DictConfig):
    if "residual_policy" in cfg.actor:
        OmegaConf.update(base_cfg.actor, "residual_policy", cfg.actor.residual_policy, merge=True)
    if "critic" in cfg:
        OmegaConf.update(base_cfg.actor, "critic", cfg.critic, merge=True)
        OmegaConf.update(base_cfg.actor, "init_logstd", cfg.init_logstd, merge=True)
    return base_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### MUST ASSIGN
    parser.add_argument("--wt-path", type=str, default=None) # NOTE: retrieve ckpt from local
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)

    ### OPTIONAL ASSIGN
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--randomness", type=str, default=None)
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        choices=[
            "one_leg",
            "lamp",
            "round_table",
            "desk",
            "square_table",
            "cabinet",
            "mug_rack",
            "factory_peg_hole",
            "bimanual_insertion",
        ],
        default=None,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)
    parser.add_argument("--store-full-resolution-video", action="store_true")
    parser.add_argument("--visualize", action="store_true") # NOTE: whether visualize through window -> headless
    parser.add_argument("--action-type", type=str, default="pos", choices=["delta", "pos", "relative"]) # NOTE: Action type for the robot. Options are 'delta' and 'pos'.
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--april-tags", action="store_true")
    parser.add_argument("--observation-space", choices=["image", "state"], default="state") # NOTE: prefer "image" for video rollout
    parser.add_argument("--stop-after-n-success", type=int, default=0)
    parser.add_argument("--break-on-n-success", action="store_true")
    parser.add_argument("--record-for-coverage", action="store_true")
    parser.add_argument("--save-rollouts-suffix", type=str, default="")
    parser.add_argument("--is-save-rollouts", action="store_true")
    parser.add_argument("--is-save-failures", action="store_true")
    parser.add_argument("--is-sample-perturbations", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Parse the arguments
    args = parser.parse_args()
    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # Get the environment
    print(f"Creating the environment with action_type {args.action_type} (this needs to be changed to enable recreation the env for each run)")
    env: Optional[Env] = None

    # Get the residual root config
    if not args.wt_path: raise KeyError("Undefined weight path (wt_path).")
    wts = torch.load(args.wt_path)
    cfg: DictConfig = OmegaConf.create(wts['config'])
    assert cfg.control.control_mode == args.action_type, "mismatch control mode"
    # print(OmegaConf.to_yaml(cfg))
    # Update former checkpoint paths
    old2new_paths = {

        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/vqvae_ckpt" : "/home/dingpengxiang/jeffrey/workspace/korr/depot/vqvae_ckpt",
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/data" : "/home/dingpengxiang/jeffrey/workspace/korr/depot/data",
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/data/kpm_goals": "/home/dingpengxiang/jeffrey/workspace/korr/depot/data/kpm_goals",
        
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/one_leg/state/low/06-11-24.326459/models/berry-bun-1": "/home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/dp/one_leg/low",
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/one_leg/state/med/08-47-59.329727/models/smooth-valley-1": "/home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/dp/one_leg/med",
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/one_leg/state/high/03-47-28.178376/models/clean-night-1": "/home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/dp/one_leg/high",
        
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/carp_ckpt/one_leg_low/12-25-53/models/obs1_pred16_act8_bestvq_2": "/home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/carp/one_leg/low/obs1_pred16_act8_bestvq_2",
    
    }
    for key, value in old2new_paths.items():
        replace_path_in_dict(cfg, key, value)
    
    # Get the base config
    # base_wt_path = "/path/to/outputs/one_leg/state/low/06-11-24.326459/models/berry-bun-1/actor_chkpt_best_success_rate.pt"
    # base_wts = torch.load(base_wt_path)
    if cfg.env.task not in cfg.base_policy.wt_path:
        import re
        match = re.search(r"\d{2}-\d{2}-\d{2}\.\d+/models/.*", cfg.base_policy.wt_path)
        base_wt_path = f"/path/to/outputs/{cfg.env.task}/state/{cfg.env.randomness}/{match.group()}"
    else:
        base_wt_path = cfg.base_policy.wt_path
    base_wts = torch.load(base_wt_path)
    base_cfg: DictConfig = OmegaConf.create(base_wts['config'])
    for key, value in old2new_paths.items():
        replace_path_in_dict(base_cfg, key, value)
    merge_base_config_with_root_config(base_cfg, cfg)

    # Assign seed
    if args.seed is None:
        args.seed = cfg.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    # Load agent
    if cfg.base_policy.actor.name == "diffusion": # ResidualDiffusionPolicy inherited from DiffusionPolicy
        agent = ResidualDiffusionPolicy(device, base_cfg) # DP + Residual RL
    elif cfg.base_policy.actor.name == 'carp':
        agent = ResidualCARP(device, base_cfg) # CARP + Residual RL
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg) # MLP + Residual RL
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")    
    agent.load_state_dict(wts['model_state_dict'])
    agent.eval()
    agent.to(device)
    residual_policy = agent.residual_policy
    
    # Set up goal reference for different situations
    if "KPM" in cfg.actor.residual_policy._target_:
        # NOTE: set goal
        g_goal_path = cfg.actor.residual_policy.g_goal_path
        if isinstance(g_goal_path, str) and g_goal_path.endswith(".pkl"): # when goals
            # only observation
            residual_nobs = unpickle_data(g_goal_path)['nobs'] # only need observation
            raw_g_goal_nobs = convert_to_tensor(residual_nobs, device) # from array to tensor
            g_goal_nobs = agent.process_obs(raw_g_goal_nobs) # normalize the obs
            agent.residual_policy.actor.set_goal(g_goal_nobs) # set in residual policy
        # set lqr cache
        agent.residual_policy.actor.backbone.set_riccati_cache_to_zero(device)
    
    # Set up goal reference for MLP situations
    if "WGoal" in cfg.actor.residual_policy._target_:
        # set goal
        g_goal_path = cfg.actor.residual_policy.g_goal_path
        if isinstance(g_goal_path, str) and g_goal_path.endswith(".pkl"): # when goals
            # observation
            residual_nobs = unpickle_data(g_goal_path)['nobs'] # only need observation
            raw_g_goal_nobs = convert_to_tensor(residual_nobs, device) # from array to tensor
            g_goal_nobs = agent.process_obs(raw_g_goal_nobs) # normalize the obs
            residual_policy.set_goal(g_goal_nobs) # set in residual policy
    
    # Set hyper-parameters from ckpt
    if args.task is None:
        args.task = cfg.env.task
    if args.randomness is None:
        args.randomness = cfg.env.randomness
    if args.max_rollout_steps is None:
        args.max_rollout_steps = cfg.num_env_steps
    
    # Set save directory
    save_dir = (
        trajectory_save_dir(
            controller="diffik",
            domain="sim",
            task=args.task,
            demo_source="rollout",
            randomness=args.randomness,
            suffix=args.save_rollouts_suffix,
            create=True,
        )
        if args.is_save_rollouts
        else None
    )
    
    # Initialize the env
    env = get_rl_env(
        gpu_id=args.gpu,
        task=args.task,
        concat_robot_state=False, # fixed
        ctrl_mode=cfg.control.controller,
        num_envs=args.n_envs,
        randomness=args.randomness, # initialization randomness
        observation_space=args.observation_space, # "image" or "state"
        max_env_steps=5_000, # fixed
        resize_img=False, # fixed
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=args.action_type,
        april_tags=args.april_tags,
        verbose=args.verbose,
        headless=not args.visualize,
    )
    
    # Perform the rollouts
    rollout_stats = calculate_success_rate(
        agent=agent,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=args.max_rollout_steps,
        epoch_idx=0,
        discount=cfg.discount,
        rollout_save_dir=save_dir, # NOTE: whether to save the directory
        is_save_failures=args.is_save_failures, # NOTE: whether save the failure cases
        n_parts_assemble=args.n_parts_assemble,
        compress_pickles=args.compress_pickles,
        resize_video=not args.store_full_resolution_video,
        break_on_n_success=args.break_on_n_success,
        stop_after_n_success=args.stop_after_n_success,
        record_first_state_only=args.record_for_coverage,
        is_sample_perturbations=args.is_sample_perturbations,
    )
    success_rate = rollout_stats.success_rate
    
    # Save success rate of the run
    save_data = {
        'model': cfg.actor.residual_policy._target_,
        'type': cfg.wandb.name,
        'randomness': args.randomness, # current utilized randomness
        'perturbation': args.is_sample_perturbations,
        'max_rollout_steps': args.max_rollout_steps,
        'success_rate': rollout_stats.success_rate,
        'n_rollouts': rollout_stats.n_rollouts,
        'n_success': rollout_stats.n_success,
        'wt_path': args.wt_path,
        'seed': args.seed,
    }
    save_file_root = "/home/dingpengxiang/jeffrey/workspace/korr/depot/outputs"
    save_file_name = f"korr.jsonl"
    # save_file_root = "/path/to/outputs/eval"
    # save_file_name = f"{args.task}_{cfg.env.randomness}.jsonl"
    with open(os.path.join(save_file_root, save_file_name), "a") as f:
        f.write(json.dumps(save_data) + "\n")
    
    # Output
    print(f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})")
    



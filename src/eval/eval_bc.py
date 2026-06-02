import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from src.gym import get_rl_env
from gymnasium import Env # NOTE: needs to be before torch imports
import torch # NOTE: needs to be after isaac gym imports
from omegaconf import DictConfig, ListConfig, OmegaConf
from src.behavior.base import Actor  # noqa
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.behavior.carp import Coarse2FineAutoRegressivePolicy  # noqa
from src.eval.rollout_bc import calculate_success_rate # noqa
from src.eval.eval_utils import replace_path_in_dict
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from typing import Any, List, Optional
from ipdb import set_trace as bp  # noqa



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-path", type=str, required=True) # NOTE: retrieve ckpt from local
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
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
        default=None
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)
    parser.add_argument("--save-rollouts", action="store_true") # NOTE: whether to save the rollouts
    parser.add_argument("--save-failures", action="store_true") # NOTE: whether to save all of the rollouts (including failures)
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
    parser.add_argument("--seed", type=int, default=None)

    # Parse the arguments
    args = parser.parse_args()
    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # Get the environment
    print(f"Creating the environment with action_type {args.action_type} (this needs to be changed to enable recreation the env for each run)")
    env: Optional[Env] = None
    # Get the config
    cfg: DictConfig = OmegaConf.create(torch.load(args.wt_path)["config"]) # local config
    # Set the task name
    if args.task is None:
        args.task = cfg.task
    # Set the timeout
    if args.max_rollout_steps is None:
        if args.task in ['one_leg']:
            rollout_max_steps = 700
        else:
            rollout_max_steps = 1000
    else:
        rollout_max_steps = args.max_rollout_steps
    
    # Assign seed
    if args.seed is None:
        args.seed = cfg.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Update former checkpoint paths
    old2new_paths = {
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/outputs/vqvae_ckpt" : "/home/dingpengxiang/jeffrey/workspace/korr/depot/vqvae_ckpt",
        "/lyushangke/zhefei/CloseLoop/Furniture-Assembly/data" : "/home/dingpengxiang/jeffrey/workspace/korr/depot/data",
    }
    for key, value in old2new_paths.items():
        replace_path_in_dict(cfg, key, value)

    # Activate the actor here
    actor: Actor = get_actor(cfg=cfg, device=device)

    # record
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.4f}M'
    params_str = f'[#para] : ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('actor', actor),)])
    print(params_str)
    
    # Set the inference steps of the actor
    if isinstance(actor, DiffusionPolicy):
        actor.inference_steps = 4  # NOTE: DDIM steps for inference
    state_dict = torch.load(args.wt_path)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Load dictionary
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.to(device)
    
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
        if args.save_rollouts
        else None
    )
    
    # Load the evaluating environment here
    if env is None:
        env = get_rl_env(
            gpu_id=args.gpu,
            task=args.task,
            num_envs=args.n_envs,
            randomness=args.randomness,
            observation_space=args.observation_space, # NOTE: "image" or "state"
            max_env_steps=5_000,
            resize_img=False, # fixed
            act_rot_repr="rot_6d", # fixed
            action_type=args.action_type,
            april_tags=args.april_tags,
            verbose=args.verbose,
            headless=not args.visualize,
        )
    
    # Perform the rollouts
    actor.set_task(task2idx[args.task])
    rollout_stats = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=rollout_max_steps,
        epoch_idx=0,
        discount=cfg.discount,
        rollout_save_dir=save_dir, # NOTE: whether to save the directory
        save_failures=args.save_failures, # NOTE: whether save the failure cases
        n_parts_assemble=args.n_parts_assemble,
        compress_pickles=args.compress_pickles,
        resize_video=not args.store_full_resolution_video,
        break_on_n_success=args.break_on_n_success,
        stop_after_n_success=args.stop_after_n_success,
        record_first_state_only=args.record_for_coverage,
    )
    success_rate = rollout_stats.success_rate
    print(f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})")

    # Save success rate of the run
    save_data = {
        'task': args.task,
        'max_rollout_steps': args.max_rollout_steps,
        'success_rate': rollout_stats.success_rate,
        'n_rollouts': rollout_stats.n_rollouts,
        'n_success': rollout_stats.n_success,
        'wt_path': args.wt_path,
        'seed': args.seed,
    }
    save_file_root = "/home/dingpengxiang/jeffrey/workspace/korr/depot/outputs"
    save_file_name = f"bc.jsonl"
    # save_file_root = "/path/to/outputs/eval"
    # save_file_name = f"{args.task}_{cfg.env.randomness}.jsonl"
    with open(os.path.join(save_file_root, save_file_name), "a") as f:
        f.write(json.dumps(save_data) + "\n")



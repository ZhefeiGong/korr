from gymnasium import Env
from omegaconf import DictConfig  # noqa: F401
import os
import copy
import torch
import matplotlib.pyplot as plt
import collections
import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401
from typing import Dict, Optional, Union
from pathlib import Path
from src.behavior.base import Actor
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import task2idx
from src.common.files import get_processed_path, trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import filter_and_concat_robot_state
from src.data_processing.utils import resize, resize_crop
from tensordict import TensorDict
from copy import deepcopy


RolloutStats = collections.namedtuple(
    "RolloutStats",
    [
        "success_rate",
        "n_success",
        "n_rollouts",
        "epoch_idx",
        "rollout_max_steps",
        "total_return",
        "total_reward",
    ],
)


RolloutSaveValues = collections.namedtuple(
    "RolloutSaveValues",
    [
        "robot_states",
        "imgs1",
        "imgs2",
        "bas_actions", # base normalized actions
        "res_actions", # residual normalized actions
        "exe_actions", # executable unnormalized actions
        "rewards",
        "parts_poses",
    ],
)


def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass


def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass


def squeeze_and_numpy(d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]):
    """
    Recursively squeeze and convert tensors to numpy arrays
    Convert scalars to floats
    Leave NoneTypes alone
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = squeeze_and_numpy(v)

        elif v is None:
            continue

        elif isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            d[k] = v.squeeze()

        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    return d


def tensordict_to_list_of_dicts(tensordict):
    list_of_dicts = []
    keys = list(tensordict.keys())
    num_elements = tensordict[keys[0]].shape[0]

    for i in range(num_elements):
        dict_element = {}
        for key in keys:
            dict_element[key] = tensordict[key][i].cpu().numpy()
        list_of_dicts.append(dict_element)

    return list_of_dicts


class SuccessTqdm(tqdm):
    def __init__(
        self,
        num_envs: int,
        n_rollouts: int,
        task_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_envs = num_envs
        self.n_rollouts = n_rollouts
        self.task_name = task_name
        self.round = 0
        self.success_in_prev_rounds = 0

    def pbar_desc(self, n_success: int):
        total = self.round * self.num_envs
        n_success += self.success_in_prev_rounds
        success_rate = n_success / total if total > 0 else 0
        self.set_description(
            f"Performing rollouts ({self.task_name}): "
            f"round {self.round}/{self.n_rollouts//self.num_envs}, "
            f"success: {n_success}/{total} ({success_rate:.1%})"
        )

    def before_round(self, n_success: int):
        self.success_in_prev_rounds = n_success
        self.round += 1

        self.pbar_desc(0)


def rollout(
    env: Env,
    agent: Actor,
    rollout_max_steps: int,
    pbar: SuccessTqdm = None,
    resize_video: bool = True,
    n_parts_assemble: int = 1,
    is_save_rollouts: bool = False,
    is_sample_perturbations: bool = False,
) -> Optional[RolloutSaveValues]:
    
    # Get first observation
    with suppress_all_output(False):
        obs = env.reset()
        agent.reset()
    video_obs = deepcopy(obs)
    
    # Resize the images in the observation if they exist
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")
    if resize_video:
        resize_image(video_obs, "color_image1")
        resize_crop_image(video_obs, "color_image2")
    
    # Save first visualization and rewards
    robot_states = [TensorDict(video_obs["robot_state"], batch_size=env.num_envs)] # robot propri. state | one more initial 
    imgs1 = [] if "color_image1" not in video_obs else [video_obs["color_image1"].cpu()] # view in gripper | one more initial 
    imgs2 = [] if "color_image2" not in video_obs else [video_obs["color_image2"].cpu()] # view in third-person | one more initial 
    parts_poses = [video_obs["parts_poses"].cpu()] # other parts' poses | one more initial 
    bas_actions = list() # normalized base action
    res_actions = list() # normalized residual action
    exe_actions = list() # executable action
    rewards = torch.zeros((env.num_envs, rollout_max_steps), dtype=torch.float32)
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")
    step_idx = 0
    
    # # Move to corresponding device
    # # agent.normalizer = agent.normalizer.to(agent.device)
    # # agent.model = agent.model.to(agent.device)
    agent = agent.to(agent.device)
    
    while not done.all():
        
        # Convert from robot state dict to robot state tensor
        obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"]) # NOTE: combine the robot state dict as a tensor
        nor_bas_act = agent.base_action_normalized(obs) # normalized base actions
        nor_bas_obs = agent.process_obs(obs) # normalized observations
        res_obs = torch.cat([nor_bas_obs, nor_bas_act], dim=-1)
        _, _, _, _, naction_mean = (agent.residual_policy.get_action_and_value(res_obs)) # residual policy
        nor_res_act = naction_mean # NOTE: deterministic results (mean) when in evaluation mode
        naction = nor_bas_act + nor_res_act * agent.residual_policy.action_scale # NOTE: combine the base actions and residual actions
        exe_action = agent.normalizer(naction, "action", forward=False) # NOTE: unnormalize the actions to executable actions

        # Run
        obs, reward, done, _ = env.step(exe_action, sample_perturbations=is_sample_perturbations)

        # # NOTE: Visualization for KPM | save through step_idx        
        # # if step_idx % 25 == 0 and step_idx < 300:
        # if step_idx % 25 == 0 :
        #     with torch.no_grad():
        #         ### params initialization <-> important for 
        #         num_dim = 64
        #         ### calculate loss
        #         obs_tmp = copy.deepcopy(obs)
        #         obs_tmp["robot_state"] = env.filter_and_concat_robot_state(obs_tmp["robot_state"])
        #         _nor_bas_obs_nxt = agent.process_obs(obs_tmp) # normalized observations
        #         # kpm_loss = agent.residual_policy.actor.kpm_loss(nobs=nor_bas_obs, nact=naction, next_nobs=_nor_bas_obs_nxt) # koopman loss
        #         _nobs = agent.residual_policy.actor.lift_transform(nor_bas_obs)
        #         _next_nobs = agent.residual_policy.actor.lift_transform(_nor_bas_obs_nxt)
        #         _next_nobs_pred = agent.residual_policy.actor.backbone._predict_koopman(_nobs, naction)
        #         mean_mse_loss = torch.mean((_next_nobs_pred - _next_nobs)**2)
        #         each_mse_loss = torch.mean((_next_nobs_pred - _next_nobs)**2, dim=0, keepdim=True).T
        #         ### visualization
        #         mean_mse_loss = mean_mse_loss.item()
        #         each_mse_loss = each_mse_loss.cpu().squeeze().numpy()
        #         # norm_each_mse_loss = (each_mse_loss - np.min(each_mse_loss)) / (np.max(each_mse_loss) - np.min(each_mse_loss))
        #         ### Create a figure with two subplots
        #         fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        #         ### **Subfigure 1: Full MSE Histogram**
        #         axes[0].bar(np.arange(num_dim), each_mse_loss, color="royalblue")
        #         axes[0].set_xlabel("Index")
        #         axes[0].set_ylabel("Loss / MSE")
        #         axes[0].set_title(f"Histogram - Mean MSE: {mean_mse_loss:.4f}")
        #         for idx in np.where(each_mse_loss > 0.02)[0]:
        #             axes[0].text(idx, each_mse_loss[idx], f"{idx}\n{each_mse_loss[idx]:.3f}", ha="center", va="bottom", fontsize=4, color="darkred")
        #         ### **Subfigure 2: Clipped MSE Histogram**
        #         threshold = 0.01
        #         above_threshold_indices = np.where(each_mse_loss > threshold)[0]  # indices above the threshold
        #         above_threshold_values = each_mse_loss[above_threshold_indices]  # values above the threshold
        #         below_threshold_values = np.clip(each_mse_loss, 0, threshold)  # values clipped to the threshold
        #         # Plot values below the threshold
        #         axes[1].bar(np.arange(num_dim), below_threshold_values, color="royalblue")
        #         # Plot values above the threshold in a different color (e.g., darkred)
        #         axes[1].bar(above_threshold_indices, above_threshold_values, color="lightblue")
        #         # Set the y-axis limits
        #         axes[1].set_ylim(0, threshold)
        #         axes[1].set_xlabel("Index")
        #         axes[1].set_ylabel("Loss (Clipped at 0.01)")
        #         axes[1].set_title("Clipped Loss Histogram (0-0.01)")
        #         ### Save the figure
        #         root_path = "/path/to/outputs/repository/one_leg_64"
        #         file_name = f"stp_{step_idx}-mse_{mean_mse_loss:.4f}.png"
        #         plt.savefig(os.path.join(root_path, file_name), dpi=300, bbox_inches="tight")
        #         plt.close()
        
        # Store the obs
        video_obs = deepcopy(obs)
        
        # Resize the images in the observation if they exist
        resize_image(obs, "color_image1")
        resize_crop_image(obs, "color_image2")
        
        # Save observations for the policy
        if resize_video:
            resize_image(video_obs, "color_image1")
            resize_crop_image(video_obs, "color_image2")
        
        # Store the results for visualization and logging
        if is_save_rollouts:
            robot_states.append(TensorDict(video_obs["robot_state"], batch_size=env.num_envs)) # robot states
            if "color_image1" in video_obs: imgs1.append(video_obs["color_image1"].cpu()) # images
            if "color_image2" in video_obs: imgs2.append(video_obs["color_image2"].cpu()) # images
            bas_actions.append(nor_bas_act.cpu()) # normalized base actions
            res_actions.append(nor_res_act.cpu()) # normalized residual actions
            exe_actions.append(exe_action.cpu()) # unnormalized executable actions

            # # NOTE: Visualization for KPM
            # with torch.no_grad():
            #     import copy
            #     obs_tmp = copy.deepcopy(obs)
            #     obs_tmp["robot_state"] = env.filter_and_concat_robot_state(obs_tmp["robot_state"])
            #     _nor_bas_obs_nxt = agent.process_obs(obs_tmp) # normalized observations
            #     kpm_loss = agent.residual_policy.actor.kpm_loss(nobs=nor_bas_obs, nact=exe_action, next_nobs=_nor_bas_obs_nxt)
            #     print(kpm_loss.item())

            parts_poses.append(video_obs["parts_poses"].cpu()) # info from other parts

        # Always store rewards as they are used to calculate success
        rewards[:, step_idx] = reward.squeeze().cpu()
        
        # Update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            n_success = (rewards.sum(dim=1) == n_parts_assemble).sum().item()
            pbar.pbar_desc(n_success)
            pbar.update()
        
        # Check done or not
        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")
        if done.all():
            break
    
    return RolloutSaveValues(
        torch.stack(robot_states, dim=1) if robot_states else [],
        torch.stack(imgs1, dim=1) if imgs1 else [],
        torch.stack(imgs2, dim=1) if imgs2 else [],
        torch.stack(bas_actions, dim=1) if bas_actions else [],
        torch.stack(res_actions, dim=1) if res_actions else [],
        torch.stack(exe_actions, dim=1) if exe_actions else [],
        rewards,
        torch.stack(parts_poses, dim=1) if parts_poses else [],
    )


@torch.no_grad()
def calculate_success_rate(
    env: Env,
    agent: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    discount: float = 0.99,
    rollout_save_dir: Optional[Path] = None,
    is_save_failures: bool = False,
    n_parts_assemble: Optional[int] = None,
    compress_pickles: bool = False,
    resize_video: bool = True,
    n_steps_padding: int = 30,
    break_on_n_success: bool = False,
    stop_after_n_success: int = 0,
    record_first_state_only: bool = False,
    is_sample_perturbations: bool = False,
) -> RolloutStats:
    
    # Bar
    pbar = SuccessTqdm(
        num_envs=env.num_envs,
        n_rollouts=n_rollouts,
        task_name=env.task_name,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
        desc="Performing rollouts",
        leave=True,
        unit="step",
    )
    
    # Initialization
    if n_parts_assemble is None:
        n_parts_assemble = env.n_parts_assemble
    n_success = 0
    all_robot_states = list()
    all_imgs1 = list()
    all_imgs2 = list()
    all_bas_actions = list()
    all_res_actions = list()
    all_exe_actions = list()
    all_rewards = list()
    all_parts_poses = list()
    all_success = list()
    is_save_rollouts = rollout_save_dir is not None

    # Run
    pbar.pbar_desc(n_success)
    for i in range(n_rollouts // env.num_envs):
        # Update the progress bar
        pbar.before_round(n_success)

        # Perform a rollout with the current model
        # NOTE: rollout in the environment
        rollout_data: RolloutSaveValues = rollout(
            env,
            agent,
            rollout_max_steps,
            pbar=pbar,
            resize_video=resize_video,
            n_parts_assemble=n_parts_assemble,
            is_save_rollouts=is_save_rollouts,
            is_sample_perturbations=is_sample_perturbations,
        )

        # Calculate the success rate
        success = rollout_data.rewards.sum(dim=1) == n_parts_assemble
        n_success += success.sum().item()

        # Save the results from the rollout
        if is_save_rollouts:
            all_robot_states.extend([rollout_data.robot_states[i] for i in range(env.num_envs)])
            all_imgs1.extend(rollout_data.imgs1)
            all_imgs2.extend(rollout_data.imgs2)
            all_bas_actions.extend(rollout_data.bas_actions)
            all_res_actions.extend(rollout_data.res_actions)
            all_exe_actions.extend(rollout_data.exe_actions)
            all_rewards.extend(rollout_data.rewards)
            all_parts_poses.extend(rollout_data.parts_poses)
            all_success.extend(success)

        if break_on_n_success and n_success >= stop_after_n_success:
            print(f"Current number of success {n_success} greater than breaking threshold {stop_after_n_success}. Breaking")
            break

    total_reward = np.sum([np.sum(rewards.numpy()) for rewards in all_rewards])
    episode_returns = [
        np.sum(rewards.numpy() * discount ** np.arange(len(rewards)))
        for rewards in all_rewards
    ]

    if record_first_state_only:
        first_robot_states = []
        first_part_poses = []
        first_success = []

    # Save rollouts here
    print(f"Checking if we should save rollouts (rollout_save_dir: {rollout_save_dir})")
    if is_save_rollouts:

        have_img_obs = len(all_imgs1) > 0
        print(f"Saving rollouts, have image observations: {have_img_obs} (will make dummy video if False)")
        total_reward = 0

        for rollout_idx in trange(len(all_robot_states), desc="Saving rollouts", leave=False):
            # Get the rewards and images for this rollout
            robot_states = tensordict_to_list_of_dicts(all_robot_states[rollout_idx])
            actions = all_exe_actions[rollout_idx].numpy() # NOTE: only save the executable actions
            rewards = all_rewards[rollout_idx].numpy()
            parts_poses = all_parts_poses[rollout_idx].numpy()
            success = all_success[rollout_idx].item()
            task = env.furniture_name
            if record_first_state_only:
                first_robot_states.append(robot_states[0])
                first_part_poses.append(parts_poses[0])
                first_success.append(success)
                continue
            video1 = (
                all_imgs1[rollout_idx].numpy() # hand-image
                if have_img_obs
                else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8) # dummy video
            )
            video2 = (
                all_imgs2[rollout_idx].numpy() # third-image
                if have_img_obs
                else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8) # dummy video
            )
            # Number of steps until success, i.e., the index of the final reward received
            n_steps = (np.where(rewards == 1)[0][-1] + 1 if success else rollout_max_steps) # NOTE: save the trajectory until the last reward timestep
            n_steps += n_steps_padding # 
            trim_start_steps = 0 # fixed to 0 as tradition
            # Save here
            if rollout_save_dir is not None and (is_save_failures or success): # NOTE: whether save the failure cases

                # Save the raw rollout data
                save_raw_rollout(
                    robot_states=robot_states[trim_start_steps : n_steps + 1],
                    imgs1=video1[trim_start_steps : n_steps + 1],
                    imgs2=video2[trim_start_steps : n_steps + 1],
                    parts_poses=parts_poses[trim_start_steps : n_steps + 1],
                    actions=actions[trim_start_steps:n_steps],
                    rewards=rewards[trim_start_steps:n_steps],
                    success=success,
                    task=task,
                    action_type=env.action_type,
                    rollout_save_dir=rollout_save_dir,
                    compress_pickles=compress_pickles,
                )
                
                # # NOTE: Visualization for KPM
                # from src.koopman.misc.vis_kpm_relation import convert_to_tensor, move_to_device
                # from src.visualization.render_mp4 import pickle_data
                # import copy
                # import os
                # _robot_states=robot_states[trim_start_steps : n_steps + 1]
                # _parts_poses=parts_poses[trim_start_steps : n_steps + 1]
                # _observations = []
                # for i in range(len(_robot_states)):
                #     _robot_state = convert_to_tensor(_robot_states[i])
                #     _robot_state = filter_and_concat_robot_state(_robot_state).unsqueeze(0)# [1, dim_robot_state]
                #     _parts_pose = convert_to_tensor(_parts_poses[i]).unsqueeze(0) # [1, dim_parts_poses]
                #     obs = {'robot_state': _robot_state, 'parts_poses': _parts_pose,}
                #     obs = move_to_device(obs, agent.device)
                #     nor_obs = agent.process_obs(obs) # normalize the observations
                #     _observations.append(nor_obs)
                # _observations = torch.cat(_observations, dim=0)
                # _nor_observations_cur = copy.deepcopy(_observations[:-1])
                # _nor_observations_next = copy.deepcopy(_observations[1:])
                # _actions = copy.deepcopy(actions[trim_start_steps:n_steps])
                # _actions = move_to_device(convert_to_tensor(_actions), agent.device)
                # _nor_actions = agent.normalizer(_actions, "action", forward=True) # unnormalize the actions to executable actions
                # with torch.no_grad():
                #     kpm_loss = agent.residual_policy.actor.kpm_loss(nobs=_nor_observations_cur, nact=_actions, next_nobs=_nor_observations_next)
                # print(kpm_loss.item())
                # save_pickle = {
                #     'nobs': _nor_observations_cur.cpu().numpy(),
                #     'nact': _actions.cpu().numpy(),
                #     'next_nobs': _nor_observations_next.cpu().numpy()
                # }
                # save_path_res_ref = os.path.join(rollout_save_dir, "obs.pkl")
                # pickle_data(save_pickle, save_path_res_ref)

        if record_first_state_only:
            first_state_npz = str(rollout_save_dir / "first_states.npz")
            print(f"Saving first states to: {first_state_npz}")
            np.savez(
                first_state_npz,
                robot_states=np.asarray(first_robot_states),
                part_poses=np.asarray(first_part_poses),
                success=np.asarray(first_success),
            )

    pbar.close()

    return RolloutStats(
        success_rate=n_success / n_rollouts,
        n_success=n_success,
        n_rollouts=n_rollouts,
        epoch_idx=epoch_idx,
        rollout_max_steps=rollout_max_steps,
        total_return=np.sum(episode_returns),
        total_reward=total_reward,
    )



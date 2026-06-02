import os
from pathlib import Path
from ipdb import set_trace as bp
import random
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler
from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config
from src.gym.observation import DEFAULT_STATE_OBS
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import wandb
from wandb.apis.public.runs import Run
from wandb.errors.util import CommError
from src.gym import get_rl_env
import gymnasium as gym

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.behavior.residual_carp import ResidualCARP
from src.visualization.render_mp4 import unpickle_data

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)

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

@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    # calculate from back to front
    for t in reversed(range(steps_per_iteration)):
        # next-value and 
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value # final
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1] # regular
        # temporal difference error -> δ_t = r_t + γ*(1−d_{t+1}))*V_{t+1} − V_t​
        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        # generalized advantage estimation: A_t = δ_t + γ*λ*(1−d_{t+1})A_{t+1}
        advantages[t] = lastgaelam = (delta + discount * gae_lambda * nextnonterminal * lastgaelam) # 
    # R = A + V
    returns = advantages + values
    return advantages, returns

@hydra.main(config_path="../config") # NOTE: default: v1.1 here
def main(cfg: DictConfig):

    # NOTE: retrieve the cfg from yaml file
    OmegaConf.set_struct(cfg, False)
    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id # NOTE: slurm for 
    # NOTE: Ensure `exactly only one` of cfg.base_policy.wandb_id or cfg.base_policy.wt_path is set
    assert (
        sum([cfg.base_policy.wandb_id is not None, 
             cfg.base_policy.wt_path is not None,])== 1
    ), "Exactly one of base_policy.wandb_id or base_policy.wt_path must be set"

    # NOTE: Build a new Run
    global_step = 0
    iteration = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0

    # NOTE: Load the behavior cloning actor
    base_wts = cfg.base_policy.wt_path # local ckpt
    base_cfg: DictConfig = OmegaConf.create(torch.load(base_wts)["config"]) # local config
    merge_base_bc_config_with_root_config(cfg, base_cfg) # NOTE: merge the base and rl configs 
    cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    # run_name = f"{int(time.time())}__{cfg.actor_name}_ppo__{cfg.seed}"
    run_name = cfg.wandb.name
    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    # NOTE: setup the seeds and params
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    # NOTE: initialize the environments
    env: gym.Env = get_rl_env(
        gpu_id=gpu_id,
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False, # fixed
        concat_robot_state=True, # fixed, concate the robot states into one feature
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        task=cfg.env.task,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state", # fixed
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000, # fixed
    )
    n_parts_to_assemble = env.n_parts_assemble # the number of parts to assemble

    # NOTE: Load the agent
    if cfg.base_policy.actor.name == "diffusion": # ResidualDiffusionPolicy inherited from DiffusionPolicy
        agent = ResidualDiffusionPolicy(device, base_cfg) # DP + Residual RL
    elif cfg.base_policy.actor.name == 'carp':
        agent = ResidualCARP(device, base_cfg) # CARP + Residual RL
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg) # MLP + Residual RL
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")
    agent.to(device) # already move to device
    agent.eval() # set to evaluation mode
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4 # for DDIM

    # NOTE: wrapper the environments and init other trianing utilities
    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        sample_perturbations=cfg.sample_perturbations,
        device=device,
    )

    # NOTE: for actor training
    optimizer_actor = optim.AdamW(
        agent.actor_parameters, # search by agent without name of `critic`
        lr=cfg.learning_rate_actor,
        betas=cfg.get("optimizer_betas_actor", (0.9, 0.999)),
        eps=1e-5,
        weight_decay=1e-6,
    )
    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.actor_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    # NOTE: for critic training
    optimizer_critic = optim.AdamW(
        agent.critic_parameters, # search by agent with name of `critic`
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )
    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.critic_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    # NOTE: load the possible weight for agent
    agent.load_base_state_dict(base_wts)
    residual_policy = agent.residual_policy
    steps_per_iteration = cfg.data_collection_steps

    # NOTE: set up goal reference for Koopman situations
    if "KPM" in cfg.actor.residual_policy._target_:
        # set goal
        g_goal_path = cfg.actor.residual_policy.g_goal_path
        if isinstance(g_goal_path, str) and g_goal_path.endswith(".pkl"): # when goals
            # observation
            residual_nobs = unpickle_data(g_goal_path)['nobs'] # only need observation
            raw_g_goal_nobs = convert_to_tensor(residual_nobs, device) # from array to tensor
            g_goal_nobs = agent.process_obs(raw_g_goal_nobs) # normalize the obs
            residual_policy.actor.set_goal(g_goal_nobs) # set in residual policy
        # set lqr cache
        residual_policy.actor.backbone.set_riccati_cache_to_zero(device)
    # NOTE: set up goal reference for MLP situations
    if "WGoal" in cfg.actor.residual_policy._target_:
        # set goal
        g_goal_path = cfg.actor.residual_policy.g_goal_path
        if isinstance(g_goal_path, str) and g_goal_path.endswith(".pkl"): # when goals
            # observation
            residual_nobs = unpickle_data(g_goal_path)['nobs'] # only need observation
            raw_g_goal_nobs = convert_to_tensor(residual_nobs, device) # from array to tensor
            g_goal_nobs = agent.process_obs(raw_g_goal_nobs) # normalize the obs
            residual_policy.set_goal(g_goal_nobs) # set in residual policy

    # NOTE: visualize the training params
    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # NOTE: init the wandb to record the training process
    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled", # NOTE: whether upload to wandb
    )

    # NOTE: Print the run name and storage location
    print(f"Run name: {run.name}")
    print(f"Run storage location: {run.dir}")

    # NOTE: Storage initialization
    buf_obs: torch.Tensor = torch.zeros((steps_per_iteration, cfg.num_envs, residual_policy.obs_dim,)) # like: [max_env_steps, num_envs, obs_dim]
    buf_next_obs: torch.Tensor = torch.zeros((steps_per_iteration, cfg.num_envs, residual_policy.obs_dim,)) # like: [max_env_steps, num_envs, obs_dim]

    buf_res_actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape) # like: [max_env_steps, num_envs, 10] -> normalized residual actions
    buf_exe_actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape) # like: [max_env_steps, num_envs, 10] -> normalized executable actions
    buf_logprobs = torch.zeros((steps_per_iteration, cfg.num_envs)) # like: [max_env_steps, num_envs]
    buf_rewards = torch.zeros((steps_per_iteration, cfg.num_envs)) # like: [max_env_steps, num_envs]
    buf_dones = torch.zeros((steps_per_iteration, cfg.num_envs)) # like: [max_env_steps, num_envs]
    buf_values = torch.zeros((steps_per_iteration, cfg.num_envs)) # like: [max_env_steps, num_envs]

    # NOTE: Others here
    start_time = time.time()
    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()
    model_save_dir: Path = Path("models") / wandb.run.name # NOTE: create model save dir
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Run PPO to sample and train
    while global_step < cfg.total_timesteps:
        
        # NOTE: Initialization
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()
        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0
        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()
        
        # NOTE: Begin to sample in PPO
        print(f"Eval mode: {eval_mode}")
        residual_policy.eval() # NOTE: set to training mode
        for step in range(0, steps_per_iteration):
            
            # Only count environment steps during training
            if not eval_mode:
                global_step += cfg.num_envs
            # Get the base normalized action
            base_naction = agent.base_action_normalized(next_obs) # get the first top action([N,action_dim]) in the storage | normalized base actions
            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs) # proprio. state + parts poses | normalized base nobs
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
            
            buf_dones[step] = next_done # NOTE: storage here
            buf_obs[step] = next_residual_nobs # NOTE: storage here -> normalized observation
            
            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (residual_policy.get_action_and_value(next_residual_nobs)) # residual policy
            residual_naction = residual_naction_samp if not eval_mode else naction_mean # NOTE: deterministic results (mean) when in evaluation mode
            naction = base_naction + residual_naction * residual_policy.action_scale # NOTE: combine the base actions and residual actions
            action = agent.normalizer(naction, "action", forward=False) # NOTE: unnormalize the actions to executable actions
            
            # Interact with environment
            next_obs, reward, next_done, truncated, info = env.step(action)
            if cfg.truncation_as_done:
                next_done = next_done | truncated
            
            buf_exe_actions[step] = naction.cpu() # NOTE: storage here -> normalized executable actions -> `naction` rather than `action`
            buf_values[step] = value.flatten().cpu() # NOTE: storage here
            buf_res_actions[step] = residual_naction.cpu() # NOTE: storage here -> normalized residual actions
            buf_logprobs[step] = logprob.cpu() # NOTE: storage here
            buf_rewards[step] = reward.view(-1).cpu() # NOTE: storage here
            
            # NOTE: storage here -> last timestep
            if step + 1 >= steps_per_iteration:
                assert len(buf_next_obs) == steps_per_iteration, "mismatch the length of buf_next_obs"
                buf_next_obs[0 : steps_per_iteration-1] = copy.deepcopy(buf_obs[1 : steps_per_iteration]) # load the former obs
                with torch.no_grad():
                    _base_naction = agent.base_action_normalized(next_obs) # normalized actions
                    _next_nobs = agent.process_obs(next_obs) # normalized observation (proprio._state + parts_poses)
                    _next_residual_nobs = torch.cat([_next_nobs, _base_naction], dim=-1) # state_obs + base_action
                buf_next_obs[-1] = _next_residual_nobs
            
            next_done = next_done.view(-1).cpu()
            if step > 0 and (env_step := step + 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={buf_rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )
        
        # NOTE: Calculate the success rate
        # Find the rewards that are `not zero`
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (buf_rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()
        if success_rate > 0:
            # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
            # Count total timesteps in successful trajectories
            timesteps_in_success = buf_rewards[:, env_success] # NOTE: [Len, Success_Num]
            # Find index of last reward in each trajectory
            # This has all timesteps including and after episode is done
            success_dones = timesteps_in_success.cumsum(dim=0) >= n_parts_to_assemble
            last_reward_idx = success_dones.int().argmax(dim=0)
            # Calculate the total number of timesteps in successful trajectories
            total_timesteps_in_success = (last_reward_idx + 1).sum().item()
            # Calculate the share of successful timesteps
            success_timesteps_share = total_timesteps_in_success / buf_rewards.numel()
            # Mean successful episode length
            mean_success_episode_length = (total_timesteps_in_success / env_success.sum().item())
            # Max successful episode length
            max_success_episode_length = last_reward_idx.max().item()
        else:
            success_timesteps_share = 0
            mean_success_episode_length = 0
            max_success_episode_length = 0
        print(
            f"SR: {success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
            f", STS: {success_timesteps_share:.4%}, MSEL: {mean_success_episode_length:.2f}"
        )
        
        # NOTE: log results if in evaluation mode
        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue
            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "iteration": iteration,
                        "training_cum_time": training_cum_time,
                    },
                    model_path,
                )
                # wandb.save(model_path) # NOTE: upload to wandb cloud (offl.)
                print(f"Evaluation success rate improved. Model saved to {model_path}")
            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        # NOTE: get training information
        b_obs = buf_obs.reshape((-1, residual_policy.obs_dim)) # unfold to (num_envs * max_steps, obs_dim)
        b_next_obs = buf_next_obs.reshape((-1, residual_policy.obs_dim)) # unfold to (num_envs * max_steps, obs_dim)
        b_res_actions = buf_res_actions.reshape((-1,) + env.action_space.shape) # unfold to (num_envs * max_steps, action_space) -> residual actions
        b_exe_actions = buf_exe_actions.reshape((-1,) + env.action_space.shape) # unfold to (num_envs * max_steps, action_space) -> executable actions
        b_logprobs = buf_logprobs.reshape(-1) # unfold to (num_envs * max_steps)
        b_values = buf_values.reshape(-1) # unfold to (num_envs * max_steps)
        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu() # Value Critic

        # NOTE: bootstrap value if not done
        advantages, returns = calculate_advantage(
            buf_values,
            buf_rewards,
            buf_dones,
            next_value,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )
        b_advantages = advantages.reshape(-1).cpu() # NOTE: 1024(num_envs) * 700(max_steps) -> 716_800
        b_returns = returns.reshape(-1).cpu() # NOTE: 1024(num_envs) * 700(max_steps) -> 716_800

        # NOTE: Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size) # NOTE: range(num_envs * max_steps)
        clipfracs = []
        residual_policy.train() # NOTE: set to training mode
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                
                # NOTE: retrieve dataset
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]
                # Get the `minibatch` and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_res_actions = b_res_actions[mb_inds].to(device) # -> normalized residual actions
                mb_exe_actions = b_exe_actions[mb_inds].to(device) # -> normalized executable actions
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)
                
                # NOTE: Calculate the loss
                # 💡 Residual policy return: a_t, logπ_θ(a_t|s_t), H[π_θ(⋅∣s_t)], value, mean
                with torch.enable_grad():
                    _, newlogprob, entropy, newvalue, action_mean = (residual_policy.get_action_and_value(mb_obs, mb_res_actions))
                logratio = newlogprob - mb_logprobs # log(π_θ/π_θ_{old})
                ratio = logratio.exp() # r(θ)
                # 💡 Kullback–Leibler divergence: D_{KL} = (π_θ_{old}∣∣π_θ) = E[r_t − 1 − logr_t] -> for record and early stop
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]
                # 💡 Normalize the advantages here
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # 💡 Policy loss: E[max(−A_t * r_t, −A_t⋅clip(r_t, 1−ϵ, 1+ϵ))]
                policy_loss = 0
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) # clip here
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # minimize the negative value -> maximize the expectation
                # 💡 Value loss : 1/2*(V(s_t)−R_t)^2 | return = advantage + oldvalue
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss: # whether clip the v_loss to prevent excessive value
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                # 💡 Entropy loss : −βH[π_θ(⋅∣s_t)]
                entropy_loss = entropy.mean() * cfg.ent_coef
                ppo_loss = pg_loss - entropy_loss
                # 💡 Add the auxiliary regularization loss
                residual_l1_loss = torch.mean(torch.abs(action_mean))
                residual_l2_loss = torch.mean(torch.square(action_mean))
                # 💡 Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:
                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss
                    policy_loss += cfg.residual_l1 * residual_l1_loss # no need here (offl.)
                    policy_loss += cfg.residual_l2 * residual_l2_loss # no need here (offl.)
                loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef
                # 💡 Add additional Koopman loss only for A&B in koopman
                if "KPM" in cfg.actor.residual_policy._target_:
                    mb_next_obs = b_next_obs[mb_inds].to(device)
                    kpm_loss = residual_policy.actor.kpm_loss(nobs=mb_obs,
                                                              nact=mb_exe_actions,
                                                              next_nobs=mb_next_obs)
                    loss += kpm_loss
                # 💡 Add additional dynamics loss only for legacy policy w/ dyn
                if "WDynamics" in cfg.actor.residual_policy._target_:
                    mb_next_obs = b_next_obs[mb_inds].to(device)
                    dyn_loss = residual_policy.dyn_loss(nobs=mb_obs,
                                                        nact=mb_exe_actions,
                                                        next_nobs=mb_next_obs)
                    loss += dyn_loss
                
                # NOTE: Optimize the total loss
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(residual_policy.parameters(), cfg.max_grad_norm) # clip the grad
                optimizer_actor.step()
                optimizer_critic.step()
                
                # NOTE: early stop only when meeting KL overflow
                if cfg.target_kl is not None and approx_kl > cfg.target_kl: 
                    print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}")
                    early_stop = True
                    break
            if early_stop:
                break
        
        # NOTE: log the info into wandb
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        action_norms = torch.norm(b_res_actions[:, :3], dim=-1).cpu() # only consider position (xyz)
        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
        wandb.log(
            {
                "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                "training/SPS": sps,
                "charts/rewards": buf_rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "charts/mean_success_episode_length": mean_success_episode_length,
                "charts/max_success_episode_length": max_success_episode_length,
                "charts/action_norm_mean": action_norms.mean(), # the manifold of residual policy output -> mean
                "charts/action_norm_std": action_norms.std(), # the manifold of residual policy output -> std
                "values/advantages": b_advantages.mean().item(),
                "values/returns": b_returns.mean().item(),
                "values/values": b_values.mean().item(),
                "values/mean_logstd": residual_policy.actor_logstd.mean().item(),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/residual_l1": residual_l1_loss.item(),
                "losses/residual_l2": residual_l2_loss.item(),
                "histograms/values": wandb.Histogram(buf_values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(buf_logprobs),
                "histograms/rewards": wandb.Histogram(buf_rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # NOTE: Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # NOTE: Checkpoint every cfg.checkpoint_interval steps
        if cfg.checkpoint_interval > 0 and iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                    "training_cum_time": training_cum_time,
                },
                model_path,
            )
            # wandb.save(model_path) # NOTE: upload to wandb cloud (offl.)
            print(f"Model saved to {model_path}")
        
        # NOTE: Print some stats at the end of the iteration
        print(f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}") # NOTE: end for one iteration (sample + optimize)

    # NOTE: Print some stats at the end of the iteration
    print(f"Training finished in {(time.time() - start_time):.2f}s") # NOTE: end of the whole training process


if __name__ == "__main__":
    main()



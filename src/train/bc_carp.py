import os
import random
from src.gym import get_rl_env # furniture-bench needs to be before torch&Env imports
from gymnasium import Env # furniture-bench needs to be before torch&Env imports
import torch # furniture-bench needs to be before torch&Env imports
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import hydra
import numpy as np
import wandb
from diffusers.optimization import get_scheduler
from ipdb import set_trace as bp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm, trange
from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths, path_override
from src.common.hydra import to_native
from src.common.pytorch_util import dict_to_device
from src.eval.eval_utils import get_model_from_api_or_cached
from src.eval.rollout_bc import do_rollout_evaluation
from src.models.ema import SwitchEMA
from src.dataset.dataloader import FixedStepsDataloader
from src.dataset.dataset import ImageDataset, StateDataset
from src.carp.optim.lr_control import lr_wd_annealing

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)

def set_dryrun_params(cfg: DictConfig):
    if cfg.dryrun:
        OmegaConf.set_struct(cfg, False)
        cfg.training.steps_per_epoch = 10 if cfg.training.steps_per_epoch != -1 else -1
        cfg.data.data_subset = 5
        cfg.data.dataloader_workers = 0
        cfg.training.eval_every = 1
        if cfg.rollout.rollouts:
            cfg.rollout.every = 1
            cfg.rollout.loss_threshold = float("inf")
        cfg.wandb.mode = "disabled"
        OmegaConf.set_struct(cfg, True)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

@hydra.main(config_path="../config", config_name="il_carp_state")
def main(cfg: DictConfig):
    
    ############ Get all params ############
    set_dryrun_params(cfg) # whether in debug mode
    OmegaConf.resolve(cfg) # update the cfg through the analysis result
    if cfg.get("seed") is None: # initialize the random seed if non-exist
        OmegaConf.set_struct(cfg, False)
        cfg.seed = np.random.randint(0, 2**32 - 1)
        OmegaConf.set_struct(cfg, True)
    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    # Other params
    env: Optional[Env] = None
    device = torch.device(f"cuda:{cfg.training.gpu_id}" if torch.cuda.is_available() else "cpu")
    state_dict = None
    # Set value for train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0
    prev_best_success_rate = 0
    global_step = 0

    ############ Build dataset ############
    # Build data path
    if cfg.data.data_paths_override is None:
        data_path = get_processed_paths(
            controller=to_native(cfg.control.controller),
            domain=to_native(cfg.data.environment),
            task=to_native(cfg.data.task),
            demo_source=to_native(cfg.data.demo_source),
            randomness=to_native(cfg.data.randomness),
            demo_outcome=to_native(cfg.data.demo_outcome),
            suffix=to_native(cfg.data.suffix),
        )
    else:
        data_path = path_override(cfg.data.data_paths_override)
    print(f"Using data from {data_path}")
    # Build dataset here
    dataset: Union[ImageDataset, StateDataset]
    if cfg.observation_type == "image":
        dataset = ImageDataset(
            dataset_paths=data_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            data_subset=cfg.data.data_subset,
            control_mode=cfg.control.control_mode,
            predict_past_actions=cfg.data.predict_past_actions,
            pad_after=cfg.data.get("pad_after", True),
            max_episode_count=cfg.data.get("max_episode_count", None),
            minority_class_power=cfg.data.get("minority_class_power", False), # for sim-to-real
            load_into_memory=cfg.data.get("load_into_memory", True),
        ) # dataset already normalized
    elif cfg.observation_type == "state":
        dataset = StateDataset(
            dataset_paths=data_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            data_subset=cfg.data.data_subset,
            control_mode=cfg.control.control_mode,
            predict_past_actions=cfg.data.predict_past_actions,
            pad_after=cfg.data.get("pad_after", True),
            max_episode_count=cfg.data.get("max_episode_count", None),
            include_future_obs=cfg.data.include_future_obs,
        ) # dataset already normalized
    else:
        raise ValueError(f"Unknown observation type: {cfg.observation_type}")
    # Split the dataset into train and test (effective, meaning that this is after upsampling)
    train_size = int(len(dataset) * (1 - cfg.data.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # Set parameters from dataset
    OmegaConf.set_struct(cfg, False) # set to 'edit' mode
    cfg.robot_state_dim = dataset.robot_state_dim       # from dataset
    cfg.action_dim = dataset.action_dim                 # from dataset
    if cfg.observation_type == "state":
        cfg.parts_poses_dim = dataset.parts_poses_dim   # from dataset
    cfg.actor.obs_dim = dataset.robot_state_dim + dataset.parts_poses_dim # from dataset and state-based only
    
    # Create the policy network
    actor: Actor = get_actor(cfg, device,)
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)
    
    ############ Create dataloaders ############
    # Set parameters from dataset
    cfg.data_path = [str(f) for f in data_path] # set the data path in the cfg object
    cfg.n_episodes = len(dataset.episode_ends) # update the cfg object with the action dimension
    cfg.n_samples = dataset.n_samples # update the cfg object with the number of samples in the dataset
    cfg.timestep_obs_dim = actor.timestep_obs_dim # update the cfg object with the observation dimension
    trainload_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    trainloader = (
        FixedStepsDataloader(**trainload_kwargs, n_batches=cfg.training.steps_per_epoch)
        if cfg.training.steps_per_epoch != -1
        else DataLoader(**trainload_kwargs)
    )
    testload_kwargs = dict(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    testloader = (
        FixedStepsDataloader(
            **testload_kwargs,
            n_batches=max(
                int(round(cfg.training.steps_per_epoch * cfg.data.test_split)), 1
            ),
        )
        if cfg.training.steps_per_epoch != -1
        else DataLoader(**testload_kwargs)
    )
    # Set corresponding parameters
    cfg.training.max_global_step = cfg.training.num_epochs * len(trainloader)
    cfg.training.warmup_step = cfg.actor.twp * len(trainloader)
    OmegaConf.set_struct(cfg, True) # set to 'protect' mode
    
    # ############ Build encoder for image-based ############
    # if cfg.observation_type == "image":
    #     opt_encoder = torch.optim.AdamW(
    #         params=actor.encoder_parameters(),
    #         lr=cfg.training.encoder_lr,
    #         weight_decay=cfg.regularization.weight_decay,
    #     )
    #     lr_scheduler_encoder = get_scheduler(
    #         name=cfg.lr_scheduler.name,
    #         optimizer=opt_encoder,
    #         num_warmup_steps=cfg.lr_scheduler.encoder_warmup_steps,
    #         num_training_steps=len(trainloader) * cfg.training.num_epochs,
    #     )
    #     optimizers.append(("encoder", opt_encoder))
    #     lr_schedulers.append(lr_scheduler_encoder)

    ############ Init wandb ############
    config_dict = OmegaConf.to_container(cfg, resolve=True) # change from cfg to dict() in python
    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        name=cfg.wandb.name,
        resume=None, # not in resume
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),# user id
        config=config_dict,
        mode=cfg.wandb.mode,
        notes=cfg.wandb.notes,
    )
    print(f"Run name: {run.name}")
    print(f"Run storage location: {run.dir}")
    wandb.config.update(config_dict) # In sweeps, the init is ignored, so to make sure that the cfg is saved correctly to wandb we need to log it manually
    train_size = int(dataset.n_samples * (1 - cfg.data.test_split))
    test_size = dataset.n_samples - train_size
    dataset_stats = {
        "num_samples_train": train_size,
        "num_samples_test": test_size,
        "num_episodes_train": int(len(dataset.episode_ends) * (1 - cfg.data.test_split)),
        "num_episodes_test": int(len(dataset.episode_ends) * cfg.data.test_split),
        "dataset_metadata": dataset.metadata,
    }
    wandb.summary.update(dataset_stats) # save stats to wandb and update the cfg object
    starttime = now()
    wandb.summary["start_time"] = starttime # save start time to wandb
    model_save_dir = Path(cfg.training.model_save_dir) / wandb.run.name # create model save dir
    model_save_dir.mkdir(parents=True, exist_ok=True) # make the directory of the model saving
    print(f"Job started at: {starttime}")
    pbar_desc = f"Epoch ({cfg.task}, {cfg.observation_type}{f', {cfg.vision_encoder.model}' if cfg.observation_type == 'image' else ''})"
    tglobal = trange(
        cfg.training.start_epoch,
        cfg.training.num_epochs,
        initial=cfg.training.start_epoch,
        total=cfg.training.num_epochs,
        desc=pbar_desc,
    )

    ############ Train begin here  ############
    for epoch_idx in tglobal:

        # Initialize
        epoch_loss = list()
        test_loss = list()
        epoch_log = {"epoch": epoch_idx,}
        train_losses_log = defaultdict(list)

        # Training loop
        actor.train()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=len(trainloader))
        for batch in tepoch:

            # Training Process
            batch = dict_to_device(batch, device)
            min_vlr, max_vlr, min_vwd, max_vwd = lr_wd_annealing(cfg.actor.tsche, 
                                                                 actor.ar_opt.optimizer, 
                                                                 cfg.actor.tlr, 
                                                                 cfg.actor.twd, 
                                                                 cfg.actor.twde, 
                                                                 global_step,
                                                                 cfg.training.warmup_step,
                                                                 cfg.training.max_global_step,
                                                                 wp0=cfg.actor.twp0, 
                                                                 wpe=cfg.actor.twpe)
            loss, losses_log, keywords = actor.compute_loss(batch)
            grad_norm, scale_log2 = actor.backward_update(loss)

            # Record to log the loss and gradients
            train_losses_log["grad_norm"] = grad_norm.item()
            train_losses_log["min_vlr"] = min_vlr
            train_losses_log["max_vlr"] = max_vlr
            train_losses_log["min_vwd"] = min_vwd
            train_losses_log["max_vwd"] = max_vwd
            for k, v in losses_log.items():
                train_losses_log[k].append(v)
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)

            # Record the usage of keywords
            for key, value in keywords.items():
                train_losses_log[key] = value

            # Update the global step
            global_step += 1
            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        # Log training params
        epoch_log["epoch_loss"] = np.mean(epoch_loss)
        for k, v in train_losses_log.items():
            epoch_log[f"train_{k}"] = np.mean(v)
        # Prepare the save dict once and we can reuse below
        save_dict = {
            "model_state_dict": actor.state_dict(),
            "best_test_loss": best_test_loss,
            "best_success_rate": best_success_rate,
            "epoch": epoch_idx,
            "global_step": global_step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Evaluation loop        
        if (cfg.training.eval_every > 0 and (epoch_idx + 1) % cfg.training.eval_every == 0):
            
            # Test begin
            actor.eval()
            eval_losses_log = defaultdict(list)
            test_tepoch = tqdm(testloader, desc="Validation", leave=False)
            for test_batch in test_tepoch:
                with torch.no_grad():
                    # device transfer for test_batch
                    test_batch = dict_to_device(test_batch, device)
                    # get test loss
                    test_loss_val, losses_log, _ = actor.compute_loss(test_batch)
                    # logging test loss
                    test_loss_cpu = test_loss_val.item()
                    test_loss.append(test_loss_cpu)
                    test_tepoch.set_postfix(loss=test_loss_cpu)
                    # append the losses to the log
                    for k, v in losses_log.items():
                        eval_losses_log[k].append(v)
            test_tepoch.close()

            # Update the epoch log with the mean of the evaluation losses
            epoch_log["test_epoch_loss"] = test_loss_mean = np.mean(test_loss)
            for k, v in eval_losses_log.items():
                epoch_log[f"test_{k}"] = np.mean(v)

            # Rollout the policy to verify the success rate
            if (
                cfg.rollout.rollouts
                and (epoch_idx + 1) % cfg.rollout.every == 0
                and np.mean(test_loss_mean) < cfg.rollout.loss_threshold
            ):
                # Do not load the environment until we successfuly made it this far
                if env is None:
                    env = get_rl_env(
                        cfg.training.gpu_id,
                        task=cfg.rollout.task,
                        num_envs=cfg.rollout.num_envs, # envs 
                        randomness=cfg.rollout.randomness, # the randomization of the initialization
                        observation_space=cfg.observation_type, # image or state
                        resize_img=False,
                        act_rot_repr=cfg.control.act_rot_repr,
                        action_type=cfg.control.control_mode, # pos
                        parts_poses_in_robot_frame=cfg.rollout.parts_poses_in_robot_frame,
                        headless=True,
                        verbose=True,
                    )
                best_success_rate = do_rollout_evaluation(
                    config=cfg,
                    env=env,
                    save_rollouts_to_file=cfg.rollout.save_rollouts,
                    save_rollouts_to_wandb=False,
                    actor=actor,
                    best_success_rate=best_success_rate,
                    epoch_idx=epoch_idx,
                )
            
            # Save the model if the test loss is the best so far
            if (cfg.training.store_best_test_loss_model and test_loss_mean < best_test_loss):
                best_test_loss = test_loss_mean
                save_path = str(model_save_dir / "carp_chkpt_best_test_loss.pt")
                torch.save(save_dict, save_path)
            
            # Save the model if the success rate is the best so far
            if (cfg.training.store_best_success_rate_model and best_success_rate > prev_best_success_rate):
                prev_best_success_rate = best_success_rate
                save_path = str(model_save_dir / f"carp_chkpt_best_success_rate.pt")
                torch.save(save_dict, save_path)
            
            # Save checkpoint regularly
            if (cfg.training.checkpoint_interval > 0 and (epoch_idx + 1) % cfg.training.checkpoint_interval == 0):
                save_path = str(model_save_dir / f"carp_chkpt_{epoch_idx}.pt")
                torch.save(save_dict, save_path)
            
        # We store the last model at the end of each epoch for better checkpointing
        if cfg.training.store_last_model:
            save_path = str(model_save_dir / "carp_chkpt_last.pt")
            torch.save(save_dict, save_path)

        # Log epoch stats
        wandb.log(epoch_log, step=global_step)
        tglobal.set_postfix(
            time=now(),
            loss=epoch_log["epoch_loss"],
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )

    tglobal.close()
    wandb.finish()

if __name__ == "__main__":
    main()



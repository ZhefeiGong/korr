############################ config ############################
# env
source /path/to/miniforge3/bin/activate
conda activate korr
# folder
cd /path/to
# config
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/path/to/NVIDIA-Linux-x86_64-550.54.14

############################ path ############################
export DATA_DIR_PROCESSED=/path/to/data
export DATA_DIR_RAW=/path/to/data/raw
export WANDB_ENTITY=your_wandb_name
export WANDB_API_KEY="..."

############################ run ############################
#--- state ---
python src/train/ppo_res.py \
--config-name rl_ppo_dp \
env.task=round_table \
env.randomness=low \
num_env_steps=1000 \
wandb.entity=your_wandb_name \
wandb.name=resip_ppo_state_dp \
debug=false \
checkpoint_interval=100 \
actor.residual_policy._target_=src.models.residual.ResidualPolicy \
base_policy.wt_path=/path/to/your/bsae/policy/actor_chkpt_best_success_rate.pt \
actor.residual_policy.g_goal_path=/path/to/your/goal_obs.pkl \


### goal-condition
# actor.residual_policy._target_=src.models.residual_w_g.ResidualPolicyWGoal \




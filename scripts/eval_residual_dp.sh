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

############################ Koopman ############################
paths=(
    "/path/to/your/actor_chkpt_best_success_rate.pt"
    "/path/to/your/actor_chkpt_best_success_rate.pt"
    "..."
)

for path in "${paths[@]}"; do
  echo "Evaluating: $path"
  python -m src.eval.eval_bc_korr \
    --n-envs 1024 \
    --n-rollouts 1024 \
    --wt-path "$path"

  python -m src.eval.eval_bc_korr \
    --n-envs 1024 \
    --n-rollouts 1024 \
    --wt-path "$path" \
    --is-sample-perturbations

  # python -m src.eval.eval_bc_korr \
  #   --n-envs 1024 \
  #   --n-rollouts 1024 \
  #   --wt-path "$path" \
  #   --randomness med 

  # python -m src.eval.eval_bc_korr \
  #   --n-envs 1024 \
  #   --n-rollouts 1024 \
  #   --wt-path "$path" \
  #   --randomness high
  
done




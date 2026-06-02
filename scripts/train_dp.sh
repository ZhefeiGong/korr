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
python -m src.train.bc_dp \
  task=round_table \
  randomness=low \
  wandb.entity=your_wandb_name \
  wandb.name=dp_roundtable_b256 \
  dryrun=false \
  training.batch_size=256 \
  training.eval_every=10 \
  rollout.every=10



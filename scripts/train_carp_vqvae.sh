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
for i in {0..9}
do
  python -m src.train.bc_vqvae \
    task=round_table \
    randomness=low \
    pred_horizon=32 \
    vqvae.patch_nums=[1,3,6,8] \
    vqvae.act_dim_sep=$i \
    wandb.entity=your_wandb_name \
    dryrun=false \
    training.num_epochs=500
done




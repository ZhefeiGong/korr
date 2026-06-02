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
paths=(
  "/path/to/your/folder/x/models/x-action-dim/x_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/y/models/y-action-dim/y_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/z/models/z-action-dim/z_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r1/models/r1-action-dim/r1_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r2/models/r2-action-dim/r2_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r3/models/r3-action-dim/r3_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r4/models/r4-action-dim/r4_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r5/models/r5-action-dim/r5_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/r6/models/r6-action-dim/r6_dim_chkpt_best_test_loss.pt"
  "/path/to/your/folder/gripper/models/gripper-action-dim/gripper_dim_chkpt_best_test_loss.pt"
)
vae_ckpt_paths="["
for p in "${paths[@]}"; do
  vae_ckpt_paths+="\"$p\", "
done
vae_ckpt_paths="${vae_ckpt_paths%, }]"
echo "Constructed VAE paths: $vae_ckpt_paths"

python -m src.train.bc_carp \
  task=round_table \
  randomness=low \
  pred_horizon=32 \
  vqvae.patch_nums=[1,3,6,8] \
  wandb.entity=your_wandb_name \
  wandb.name=obs1_pred32_act8_bestvq_tdepth16 \
  dryrun=false \
  training.num_epochs=5000 \
  training.batch_size=256 \
  training.eval_every=10 \
  rollout.every=10 \
  actor.tdepth=16 \
  actor.vae_ckpt_paths="$vae_ckpt_paths"



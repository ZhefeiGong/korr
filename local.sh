
### add ssh key 
eval "$(ssh-agent -s)"
chmod 600 ~/jeffrey/id_rsa_ubuntu
ssh-add ~/jeffrey/id_rsa_ubuntu
git config --global user.email "zhefeigong@gmail.com"
git config --global user.name "zhefeigong"
git config user.name
git config user.email
git config --list

### activate miniforge
source /home/dingpengxiang/jeffrey/miniforge3/bin/activate
conda activate furn_asm


# refer to https://github.com/ankile/robust-rearrangement
### install Isaac-Gym
conda create -n furn_asm python=3.8 -y
conda activate furn_asm
wget --no-check-certificate https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/packages/IsaacGym_Preview_4_Package.tar.gz
tar -xzf IsaacGym_Preview_4_Package.tar.gz
pip install -e /home/dingpengxiang/jeffrey/isaacgym/python --no-cache-dir --force-reinstall
### install furniture-bench
git clone --recursive git@github.com:ankile/robust-rearrangement.git
cd robust-rearrangement/furniture-bench
pip install -e .
pip install ipdb # extra
### test furniture bench
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
python -m furniture_bench.scripts.run_sim_env --furniture lamp --scripted
python -m furniture_bench.scripts.run_sim_env --furniture round_table --scripted
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --no-action
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --input-device keyboard
### install robust-rearrangement
cd ..
pip install -e .


# refer to https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_bench.html#set-up-connection
### initialize serial number of realsense-d435i
export CAM_WRIST_SERIAL=335522072596
export CAM_FRONT_SERIAL=135122074283
export CAM_REAR_SERIAL=344322073523
### test connection of camera
python furniture_bench/scripts/run_cam_april.py
### calibrate 
python furniture_bench/scripts/calibration.py --target setup_front
python furniture_bench/scripts/calibration.py --target obstacle
python furniture_bench/scripts/calibration.py --target one_leg
# python furniture_bench/scripts/calibration.py --target lamp
# python furniture_bench/scripts/calibration.py --target round_table


### execute policy in simulation from the workstation | `one_leg` only
# env
source /home/dingpengxiang/jeffrey/miniforge3/bin/activate
conda activate furn_asm
# folder
cd /home/dingpengxiang/jeffrey/workspace/korr
# carp
python -m src.eval.eval_bc \
--n-envs 64 \
--n-rollouts 64 \
--task one_leg \
--max-rollout-steps 700 \
--action-type pos \
--observation-space state \
--randomness low \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/carp/one_leg/low/rbt_obs1_pred16_act8_bestvq_tdepth32/carp_chkpt_last.pt \
--visualize 
# dp
python -m src.eval.eval_bc \
--n-envs 64 \
--n-rollouts 64 \
--task one_leg \
--max-rollout-steps 700 \
--action-type pos \
--observation-space state \
--randomness low \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/dp/one_leg/low/actor_chkpt_best_success_rate.pt \
--visualize 
python -m src.eval.eval_bc \
--n-envs 64 \
--n-rollouts 64 \
--task one_leg \
--max-rollout-steps 700 \
--action-type pos \
--observation-space state \
--randomness med \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/dp/one_leg/med/actor_chkpt_best_success_rate.pt \
--visualize 
# dp - resip
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/low/mlp/res_ppo_state_dp/actor_chkpt_best_success_rate.pt \
--visualize 
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/med/mlp/res_ppo_state_dp_2/actor_chkpt_best_success_rate.pt
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/high/mlp/res_ppo_state_dp_3/actor_chkpt_best_success_rate.pt
# dp - korr
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/low/kpm/nor_res_kpm_ppo_state_dp_wo_lqr_wo_g/actor_chkpt_best_success_rate.pt \
--visualize 
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/med/kpm/nor_res_kpm_ppo_state_dp_wo_lqr_wo_g/actor_chkpt_best_success_rate.pt
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_dp/one_leg/high/kpm/nor_res_kpm_ppo_state_dp_wo_lqr_wo_g/actor_chkpt_best_success_rate.pt
# carp - resip
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_carp/one_leg/low/mlp/res_ppo_state_dp_2/actor_chkpt_best_success_rate.pt
# carp - korr
python -m src.eval.eval_korr \
--n-envs 64 \
--n-rollouts 64 \
--wt-path /home/dingpengxiang/jeffrey/workspace/korr/depot/ckpt/korr_carp/one_leg/low/kpm/nor_res_kpm_ppo_state_dp_wo_lqr_2/actor_chkpt_best_success_rate.pt






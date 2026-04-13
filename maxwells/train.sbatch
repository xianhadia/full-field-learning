#!/bin/bash
#SBATCH -J ...                           # Job name
#SBATCH -A ...                         # Charge account
#SBATCH -N 1 --gres=gpu:l40s:1                      # Number of nodes and GPUs required
#SBATCH -t 240                                      # Duration of the job
#SBATCH -q embers                                   # QOS name
#SBATCH -o Report-%j.out                            # Combined output and error messages file
#SBATCH --mail-type=...                  # Mail preferences
#SBATCH --mail-user=...              # e-mail address for notifications

export DDE_BACKEND=pytorch

# ==========================================
# 1D TRAINING COMMANDS
# ==========================================

# FE Yielded: ~0.294787 
python train.py --dim 1d -train 768 -val 128 -test 1000 --iters 10000 --eval_interval 100 --start_eval_iter 5000 --pe_x 8 --pe_cross 2 

# FE Yielded: 0.267584
python train.py --dim 1d -train 1280 -val 320 -test 1000 --iters 10000 --eval_interval 100 --start_eval_iter 5000 --pe_x 8 --pe_cross 4 

# FE Yielded: 0.250814
python train.py --dim 1d -train 2304 -val 576 -test 1000 --iters 25000 --eval_interval 250 --pe_x 8 --pe_cross 4 --start_eval_iter 10000

# FE Yielded: 0.221188
python train.py --dim 1d -train 4352 -val 1088 -test 1000 --iters 25000 --eval_interval 250 --pe_x 12 --pe_cross 4 --start_eval_iter 10000

# FE Yielded: 0.183343
python train.py --dim 1d -train 8448 -val 2112 -test 1000 --iters 25000 --eval_interval 250 --pe_x 12 --pe_cross 4 --start_eval_iter 10000 


# ==========================================
# 2D TRAINING COMMANDS
# ==========================================

# FE Yielded: 0.290236
python train.py --dim 2d -train 768 -val 128 -test 1000 --iters 10000 --eval_interval 100 --start_eval_iter 5000 --pe_x 8 --pe_y 8 --pe_cross 2

# FE Yielded: 0.267393
python train.py --dim 2d -train 1280 -val 320 -test 1000 --iters 10000 --eval_interval 100 --pe_x 8 --pe_y 8 --pe_cross 4 --start_eval_iter 8000

# FE Yielded: 0.247983
python train.py --dim 2d -train 2304 -val 576 -test 1000 --iters 25000 --eval_interval 250 --pe_x 8 --pe_y 8 --pe_cross 4 --start_eval_iter 10000

# FE Yielded: 0.219736
python train.py --dim 2d -train 4352 -val 1088 -test 1000 --iters 25000 --eval_interval 250 --pe_x 12 --pe_y 12 --pe_cross 4 --start_eval_iter 10000

# FE Yielded: 0.176473
python train.py --dim 2d -train 8448 -val 2112 -test 1000 --iters 25000 --eval_interval 250 --pe_x 12 --pe_y 12 --pe_cross 4 --start_eval_iter 10000 --batch 8
#!/bin/bash
#SBATCH -J MERGEDEEPONET                           
#SBATCH -A ....                         
#SBATCH -N 1 --gres=gpu:l40s:1                    
#SBATCH -t 60                                     
#SBATCH -q embers                                  
#SBATCH -o Report-%j.out                          
#SBATCH --mail-type=BEGIN,END,FAIL                 
#SBATCH --mail-user=...          

export DDE_BACKEND=pytorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# --- Helper Function to Train, Test, and Move Results ---
run_experiment() {
    # Arguments: 1: Target_Parent_Dir (e.g., diffusion16), 2+: Training Args
    TARGET_BASE=$1
    shift
    
    echo "=========================================================="
    echo "STARTING: $TARGET_BASE with args $@"
    echo "=========================================================="
    
    # 1. Run Training (outputs to output/TIMESTAMP_ID)
    python train.py "$@"
    
    # 2. Identify the output directory created by train.py
    LATEST_DIR=$(ls -td output/*/ | head -1)
    
    # 3. Create the organized directory structure
    # This creates folders like diffusion16/2D/ or diffusion25/1D/
    # We extract the 'dim' from the arguments to sub-categorize
    DIM_FOLDER="2d"
    [[ "$*" == *"--dim 1d"* ]] && DIM_FOLDER="1d"
    
    FINAL_DEST="${TARGET_BASE}/${DIM_FOLDER}"
    mkdir -p "$FINAL_DEST"
    
    # 4. Run Evaluation (using your uploaded test.py)
    # Note: test.py requires --coarse_h and --coarse_w
    # python test_FE.py --dir "$LATEST_DIR" --eval_model best 
    
    # 5. Run Timing Benchmark (using your uploaded test_time.py)
    # python test_time.py "$LATEST_DIR" --trials 50
    
    # 6. Move the entire result folder (including all logs) to the target location
    mv "$LATEST_DIR" "$FINAL_DEST/"
    
    echo "COMPLETED: Results moved to $FINAL_DEST"
    echo " "
}

# ==========================================================
# DIFFUSION 16
# ==========================================================
# 2D
run_experiment "diffusion16" --dim 2d --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fourier16.npz" \
    --iters 20000 --eval_start_after 15000 --pe_x 4 --pe_y 4 --pe_cross 2 --batch 8

# 1D
run_experiment "diffusion16" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fourier16.npz"  \
    --iters 30000 --eval_start_after 25000 --pe_y 36 --batch 64 --lr 3e-4 --activation silu

# ==========================================================
# DIFFUSION 25
# ==========================================================
# 2D
run_experiment "diffusion25" --dim 2d --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fourier25.npz" \
    --iters 35000 --eval_start_after 15000 --pe_x 8 --pe_y 8 --pe_cross 4 --batch 16
    
run_experiment "diffusion25"  --dim 1d --stride_y_train 1 --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fourier16.npz" \
    --iters 30000 --eval_start_after 25000 --pe_y 36 --batch 64 --lr 3e-4 --activation silu

# ==========================================================
# REACTION-DIFFUSION 16 (Fisher)
# ==========================================================
# 2D
run_experiment "reaction_diffusion16" --dim 2d --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fisher16.npz" \
    --iters 30000 --eval_start_after 15000 --pe_x 8 --pe_y 8 --pe_cross 4 --batch 8

# 1D
run_experiment "reaction_diffusion16" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier16.npz"../data/Y_fisher16.npz" \
    --iters 10000 --eval_start_after 1000 --pe_y 8 --batch 8 --lr 5e-4

# ==========================================================
# REACTION-DIFFUSION 25 (Fisher)
# ==========================================================
# 2D
run_experiment "reaction_diffusion25" --dim 2d --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fisher25.npz" \
    --iters 30000 --eval_start_after 15000 --pe_x 8 --pe_y 8 --pe_cross 4 --batch 8

# 1D
run_experiment "reaction_diffusion25" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fisher25.npz" \
    --iters 10000 --eval_start_after 1000 --pe_y 8 --batch 8 --lr 5e-4

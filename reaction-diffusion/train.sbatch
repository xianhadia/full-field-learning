#!/bin/bash
#SBATCH -J TRAIN_MERGE
#SBATCH -A [YOUR_ACCOUNT]
#SBATCH -N 1 --gres=gpu:l40s:1
#SBATCH -t 60
#SBATCH -o logs/train-%j.out
#SBATCH --mail-type=END,FAIL

export DDE_BACKEND=pytorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_experiment() {
    TARGET_BASE=$1
    shift
    
    # 1. Run Training
    python train.py "$@"
    
    # 2. Find the timestamped directory created
    LATEST_DIR=$(ls -td output/*/ | head -1)
    
    # 3. Determine subfolder (1d or 2d)
    DIM_FOLDER="2d"
    [[ "$*" == *"--dim 1d"* ]] && DIM_FOLDER="1d"
    
    FINAL_DEST="${TARGET_BASE}/${DIM_FOLDER}"
    mkdir -p "$FINAL_DEST"
    
    # 4. Move artifacts and cleanup timestamped folder
    mv "$LATEST_DIR"* "$FINAL_DEST/"
    rmdir "$LATEST_DIR"
    
    echo "Done: $TARGET_BASE ($DIM_FOLDER)"
}

# --- DIFFUSION 16 ---
run_experiment "diffusion16" --dim 2d --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fourier16.npz" --iters 20000 --pe_x 4 --pe_y 4 --batch 8
run_experiment "diffusion16" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fourier16.npz" --iters 30000 --pe_y 36 --batch 64 --activation silu

# --- DIFFUSION 25 ---
run_experiment "diffusion25" --dim 2d --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fourier25.npz" --iters 35000 --pe_x 8 --pe_y 8 --batch 16
run_experiment "diffusion25" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fourier25.npz" --iters 30000 --pe_y 36 --batch 64 --activation silu

# --- REACTION-DIFFUSION 16 (Fisher) ---
run_experiment "reaction_diffusion16" --dim 2d --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fisher16.npz" --iters 30000 --pe_x 8 --pe_y 8 --batch 8
run_experiment "reaction_diffusion16" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier16.npz" --y_path "../data/Y_fisher16.npz" --iters 10000 --pe_y 8 --batch 8 --lr 5e-4

# --- REACTION-DIFFUSION 25 (Fisher) ---
run_experiment "reaction_diffusion25" --dim 2d --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fisher25.npz" --iters 30000 --pe_x 8 --pe_y 8 --batch 8
run_experiment "reaction_diffusion25" --dim 1d --stride_y_train 1 --x_path "../data/X_fourier25.npz" --y_path "../data/Y_fisher25.npz" --iters 10000 --pe_y 8 --batch 8 --lr 5e-4

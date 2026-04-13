#!/bin/bash
#SBATCH -J MaxwellsDeepONet
#SBATCH - ...                            # Charge account
#SBATCH -N1 --gres=gpu:l40s:1                       # Number of nodes and GPUs required
#SBATCH -t 4:00:00
#SBATCH -q embers
#SBATCH -o Master-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=...            # e-mail address for notifications

export CUDA_VISIBLE_DEVICES=0
export DDE_BACKEND=pytorch

# Function to train and evaluate a specific configuration
run_pipeline() {
    local dim=$1
    local train_size=$2
    local val_size=$3
    local iters=$4        
    local pe_x=$5
    local pe_cross=$6
    local pe_y=$7
    local extra_args=$8

    echo "================================================================"
    echo "STARTING: DIM=$dim, TRAIN=$train_size, PE_X=$pe_x"
    echo "================================================================"

    # 1. Training
    # train.py automatically creates a timestamped folder in output/
    if [ "$dim" == "1d" ]; then
        cmd="python train.py --dim 1d -train $train_size -val $val_size -test 1000 --iters $iters --pe_x $pe_x --pe_cross $pe_cross $extra_args"
    else
        cmd="python train.py --dim 2d -train $train_size -val $val_size -test 1000 --iters $iters --pe_x $pe_x --pe_y $pe_y --pe_cross $pe_cross $extra_args"
    fi
    
    # Run training and capture the output directory
    output_dir=$($cmd | grep "All artifacts saved to:" | awk '{print $NF}')
    
    if [ -z "$output_dir" ]; then
        echo "Error: Training failed to return an output directory."
        return
    fi

    echo "Artifacts located at: $output_dir"

    # 2. Testing for Fractional Error (FE)
    echo "Running Accuracy Evaluation..."
    python test.py --dir "$output_dir" --eval_model best

    # 3. Timing Evaluation (GPU)
    echo "Running GPU Timing Benchmark..."
    python test_time.py "$output_dir" --device cuda --trials 100

    # 4. Timing Evaluation (CPU)
    echo "Running CPU Timing Benchmark..."
    python test_time.py "$output_dir" --device cpu --trials 50
}
# ==========================================
# 1D MODELS
# ==========================================
run_pipeline "1d" 768  128  10000 8  2 0 "--eval_interval 100 --start_eval_iter 5000"
run_pipeline "1d" 1280 320  10000 8  4 0 "--eval_interval 100 --start_eval_iter 5000"
run_pipeline "1d" 2304 576  25000 8  4 0 "--eval_interval 250 --start_eval_iter 10000"
run_pipeline "1d" 4352 1088 25000 12 4 0 "--eval_interval 250 --start_eval_iter 10000"
run_pipeline "1d" 8448 2112 25000 12 4 0 "--eval_interval 250 --start_eval_iter 10000"

# ==========================================
# 2D MODELS
# ==========================================
run_pipeline "2d" 768  128  10000 8  2 8 "--eval_interval 100 --start_eval_iter 5000"
run_pipeline "2d" 1280 320  10000 8  4 8 "--eval_interval 100 --start_eval_iter 8000"
run_pipeline "2d" 2304 576  25000 8  4 8 "--eval_interval 250 --start_eval_iter 10000"
run_pipeline "2d" 4352 1088 25000 12 4 12 "--eval_interval 250 --start_eval_iter 10000"
run_pipeline "2d" 8448 2112 25000 12 4 12 "--eval_interval 250 --start_eval_iter 10000 --batch 8"

echo "Full pipeline completed successfully."
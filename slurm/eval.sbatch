#!/bin/bash
#SBATCH -A lu2025-7-98
#SBATCH -p gpua40i
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
module load GCCcore/13.3.0 Python/3.12.3
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

MODEL="$1"     # path to the model folder
BACKEND="$2"   # mntp for GPT-BERT, causal for GPT-2 / Chatty BERT
cd external/babylm-eval/strict
bash scripts/eval_zero_shot.sh "$MODEL" "$BACKEND"
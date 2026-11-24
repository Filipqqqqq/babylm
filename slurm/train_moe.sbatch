#!/bin/bash
#SBATCH -A lu2025-7-98
#SBATCH -p gpua40i
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -J moe-seed42
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-user=filip.svensson2001@gmail.com
#SBATCH --mail-type=FAIL,END

module purge
module load GCCcore/13.3.0 Python/3.12.3
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

nvidia-smi
python scripts/train/train_causal_lm.py \
  --config configs/train/edan70/moe_strict_small.yaml

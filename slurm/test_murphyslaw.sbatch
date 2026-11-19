#!/bin/bash
#SBATCH -A lu2025-7-98
#SBATCH -p gpua40i
#SBATCH --gres=gpu:1
#SBATCH -t 00:15:00
#SBATCH -J test-murphyslaw
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-user=filip.svensson2001@gmail.com
#SBATCH --mail-type=FAIL,END

module purge
module load GCCcore/13.3.0 Python/3.12.3
cd ~/babylm
source .venv/bin/activate
export HF_HUB_OFFLINE=1

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/train/train_hybrid_lm.py \
  --config configs/train/submission/murphyslaw_strict_small.yaml \
  --set run.name=cluster_test \
  --set training.total_steps=30 \
  --set data.max_train_blocks=64 \
  --set data.max_val_blocks=8

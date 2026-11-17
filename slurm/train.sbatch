#!/bin/bash
# Train on kcist: sbatch slurm/train.sbatch [hydra overrides]
#SBATCH --job-name=symmbfn
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$HOME/Masterthesis/symmbfn"
export PATH="$HOME/micromamba/envs/symmbfn/bin:$PATH"

# DataLoader workers otherwise hit "Too many open files".
ulimit -n "$(ulimit -Hn)"
mkdir -p wandb

# Offline unless WANDB_MODE=online is exported (needs `wandb login`). Logger kwarg and
# env var must agree, or wandb 0.10.33 crashes in wandb.init.
python symmbfn/model/symmbfn_train.py logging.wandb.mode="${WANDB_MODE:-offline}" "$@"

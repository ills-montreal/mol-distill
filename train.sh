#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=9
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export DATASET=MOSES
export WORKING_DIR=/home/fransou/distill
export SLURM_DIR=$SLURM_TMPDIR/tmp_dir
export DATA_DIR=/home/fransou/scratch/distill

echo "Starting job on dataset $DATASET"

module load python/3.10 scipy-stack rdkit
source /home/fransou/DISTILL/bin/activate

echo "Running script on dataset $DATASET with dim $1, gnn-type $2 and n-layer $3"
cd $WORKING_DIR/mol-distill

wandb offline
python molDistill/train_gm.py \
  --dataset $DATASET \
  --data-dir $DATA_DIR/data_train \
  --wandb \
  --dim $1 \
  --gnn-type $2 \
  --n-layer $3 \
  --out-dir $DATA_DIR/ckpt/$4


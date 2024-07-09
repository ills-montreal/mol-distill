#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export DATASET=hERG
export DISTILL_DIR=/home/fransou/distill
export SLURM_DIR=$SLURM_TMPDIR/tmp_dir/distill

echo "Starting job on dataset $DATASET"

mkdir -p $SLURM_DIR/data $SLURM_DIR/mol-distill

cp -r $DISTILL_DIR/mol-distill $SLURM_DIR
cp -r $DISTILL_DIR/data $SLURM_DIR

module load python/3.11 scipy-stack rdkit
source /home/fransou/DISTILL/bin/activate

echo "Running script on dataset $DATASET with dim $1, gnn-type $2 and n-layer $3"
cd $SLURM_DIR/mol-distill

wandb offline
python molDistill/downstream_eval.py \
  --embedders $1 \
  --data-path $SLURM_DIR/data \
  --n-runs $2

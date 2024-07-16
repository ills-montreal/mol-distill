#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export DATASET=MOSES
export WORKING_DIR=/home/fransou/distill
export SLURM_DIR=$SLURM_TMPDIR/tmp_dir/distill
export DATA_DIR=/home/fransou/scratch/distill/data_train

echo "Starting job on dataset $DATASET"

mkdir -p $SLURM_DIR/data

#cp -r /home/fransou/scratch/distill/data_train/$DATASET.zip $SLURM_DIR/
#unzip $SLURM_DIR/$DATASET.zip -d $SLURM_DIR/data

module load python/3.10 scipy-stack rdkit
source /home/fransou/DISTILL/bin/activate

echo "Running script on dataset $DATASET with dim $1, gnn-type $2 and n-layer $3"
cd $SLURM_DIR/mol-distill

wandb offline
python molDistill/train_gm.py \
  --dataset $DATASET \
  --data-dir $DATA_DIR \
  --wandb \
  --dim $1 \
  --gnn-type $2 \
  --n-layer $3 \
  --out-dir $DISTILL_DIR/mol-distill/results/$4


#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

export DATASET=ZINC

echo "Starting job on dataset $DATASET  and model $MODELS"

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir
mkdir distill
cd distill

cp -r /home/fransou/distill/mol-distill .
cp -r /home/fransou/distill/data/$DATASET .


module load python/3.11
module load scipy-stack
module load rdkit
source /home/fransou/DISTILL/bin/activate

cd distill/mol-distill


echo "Running script on dataset $DATASET with dim $1, gnn-type $2 and n-layer $3"
wandb offline
python molDistill/train_gm.py --dataset $DATASET --data-dir $SLURM_TMPDIR/tmp_dir/distill/data --wandb --dim $1 --gnn-type $2 --n-layer $3

cp -r wandb/* /home/fransou/distill/wandb
cp -r results/* /home/fransou/distill/mol-distill/results
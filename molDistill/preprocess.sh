#!/bin/bash
#SBATCH --job-name=distill_mol
#SBATCH --account=def-ibenayed
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out


export DISTILL_DIR=/home/fransou/distill

export SLURM_DIR=$SLURM_TMPDIR/tmp_dir/distill

mkdir -p $SLURM_DIR/data $SLURM_DIR/mol-distill

cp -r $DISTILL_DIR/mol-distill $SLURM_DIR

module load python/3.10 scipy-stack rdkit

cd $SLURM_DIR/mol-distill

source /home/fransou/DISTILL/bin/activate
python molDistill/preprocess_tdc_dataset_multifiles.py \
  --dataset $1 \
  --data-path $DISTILL_DIR/data \
  --i0 $2
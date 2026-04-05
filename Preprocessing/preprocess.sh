#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/preprocessing/preprocess_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/preprocessing/preprocess_%A.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Preprocessing/cnn_preprocess.py
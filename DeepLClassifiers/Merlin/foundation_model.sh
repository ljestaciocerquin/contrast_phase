#!/bin/bash
#SBATCH --job-name=merlin
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx_qos
#SBATCH --array=0
#SBATCH --output=/projects/net_contrast_classification/jobs/deep_class/merlin/merlin_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/deep_class/merlin/merlin_%A_%a.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate merlin_env

cd /projects/net_contrast_classification/contrast_phase

python DeepLClassifiers/Merlin/foundation_model.py
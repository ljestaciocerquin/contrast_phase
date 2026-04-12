#!/bin/bash
#SBATCH --job-name=paired_preprocess
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --qos=rtx_qos
#SBATCH --array=0-29
#SBATCH --output=/projects/net_contrast_classification/jobs/preprocessing/pairs/preprocess_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/preprocessing/pairs/preprocess_%A_%a.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Preprocessing/paired_preprocess.py
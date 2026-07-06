#!/bin/bash
#SBATCH --job-name=seg_preprocess
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --qos=rtx_qos
#SBATCH --array=0-7
#SBATCH --output=/projects/net_contrast_classification/jobs/preprocessing/segmentation_data/preprocess_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/preprocessing/segmentation_data/preprocess_%A_%a.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Preprocessing/Segmentation_data/seg_preprocess.py
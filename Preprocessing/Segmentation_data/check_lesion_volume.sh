#!/bin/bash
#SBATCH --job-name=lesion_vol
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=3-00:00:00
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/preprocessing/seg_data/lesion_vol_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/preprocessing/seg_data/lesion_vol_%A.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Preprocessing/Segmentation_data/check_lesion_volume.py
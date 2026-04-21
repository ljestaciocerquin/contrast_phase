#!/bin/bash
#SBATCH --job-name=contrast_3d
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx_qos
#SBATCH --array=0-1
#SBATCH --output=/projects/net_contrast_classification/jobs/deep_class/contrast/contrast_3d_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/deep_class/contrast/contrast_3d_%A_%a.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python DeepLClassifiers/Contrast/contrast_3d_cnn.py
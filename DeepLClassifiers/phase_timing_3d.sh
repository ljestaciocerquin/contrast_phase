#!/bin/bash
#SBATCH --job-name=phase_timing_3d
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/deep_class/phase_timing_3d_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/deep_class/phase_timing_3d_%A.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python DeepLClassifiers/phase_timing_3d_cnn.py
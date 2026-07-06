#!/bin/bash
#SBATCH --job-name=time_estim
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx_qos
#SBATCH --array=0-5
#SBATCH --output=/projects/net_contrast_classification/jobs/deep_class/time_interval/time_estim_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/deep_class/time_interval/time_estim_%A_%a.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python DeepLClassifiers/Time_interval/time_estim.py
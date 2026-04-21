#!/bin/bash
#SBATCH --job-name=time_estim
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/deep_class/time_interval/time_estim_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/deep_class/time_interval/time_estim_%A.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python DeepLClassifiers/Time_interval/time_estim.py
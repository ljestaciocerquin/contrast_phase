#!/bin/bash
#SBATCH --job-name=time_estim
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/time_estim/time_estim_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/time_estim/time_estim_%A.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Time_estim/time_estim.py
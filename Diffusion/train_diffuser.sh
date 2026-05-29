#!/bin/bash
#SBATCH --job-name=diffuse
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/diffusion/diff_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/diffusion/diff_%A.err


source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class



cd /projects/net_contrast_classification/contrast_phase

python Diffusion/train_diffuser_v3.py
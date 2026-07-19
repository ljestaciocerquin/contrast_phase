#!/bin/bash
#SBATCH --job-name=eval_seg
#SBATCH --partition=rtx2080ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --qos=rtx_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/segmentation/eval_seg_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/segmentation/eval_seg_%A.err


# echo "Running task $SLURM_ARRAY_TASK_ID"

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

cd /projects/net_contrast_classification/contrast_phase

python Segmentation/evaluate_seg_quality.py
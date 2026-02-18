#!/bin/bash
#SBATCH --job-name=radiomics
#SBATCH --partition=rtx2080ti
#SBATCH --nodelist=hamilton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --qos=rtx_qos
#SBATCH --array=0-3
#SBATCH --output=/projects/net_contrast_classification/jobs/radiomics_job/radiomics_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/radiomics_job/radiomics_%A_%a.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

python /projects/net_contrast_classification/contrast_phase/run_radiomics.py $SLURM_ARRAY_TASK_ID

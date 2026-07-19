#!/bin/bash
#SBATCH --job-name=radiomics
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/net_contrast_classification/jobs/radiomics_job/radiomics_%A_%a.out
#SBATCH --error=/projects/net_contrast_classification/jobs/radiomics_job/radiomics_%A_%a.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate class

python /projects/net_contrast_classification/contrast_phase/Radiomics/radiomics_extract.py $SLURM_ARRAY_TASK_ID

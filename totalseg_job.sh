#!/bin/bash
#SBATCH --job-name=totalseg
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
#SBATCH --output=totalseg_%A_%a.out
#SBATCH --error=totalseg_%A_%a.err

module load miniconda
conda activate class

python /projects/net_contrast_classification/contrast_phase/process_totalseg.py $SLURM_ARRAY_TASK_ID

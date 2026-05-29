#!/bin/bash
#SBATCH --job-name=reg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10GB 
#SBATCH --time=3-00:00:00
#SBATCH --qos=cpu_qos
#SBATCH --output=/projects/net_contrast_classification/jobs/registration/reg_%A.out
#SBATCH --error=/projects/net_contrast_classification/jobs/registration/reg_%A.err

source /home/k.minkova/miniconda3/etc/profile.d/conda.sh
conda activate guidereg

cd /projects/net_contrast_classification/contrast_phase/GuideReg/guidereg

# guidereg pipeline --config configs/examples/liver_ct.yaml
guidereg propagate --config configs/examples/propagate_tumor.yaml

#!/bin/bash
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-10:00:00
#SBATCH --mem=64GB
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --requeue


source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate jupyter_kernel;
python3 convert.py;
conda deactivate;
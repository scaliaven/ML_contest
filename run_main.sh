#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --partition=h100_1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1      
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate jupyter_kernel
ml cuda/11.6.2
python main.py
conda deactivate
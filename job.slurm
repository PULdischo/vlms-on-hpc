#!/bin/bash
#SBATCH --job-name=nanoNets      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=apjanco@princeton.edu

module purge
module load anaconda3/2024.6
conda activate qwen
HF_HUB_OFFLINE=1
python main.py

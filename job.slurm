#!/bin/bash
#SBATCH --job-name=vlms-on-hpc    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                # total memory (RAM) per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100        # target A100 nodes (avoid V100 OOM)
#SBATCH --output=log/%j.out      # stdout log
#SBATCH --error=log/%j.err       # stderr log
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=apjanco@princeton.edu

module purge
module load anaconda3/2025.6
conda activate vlm
export HF_HUB_OFFLINE=1
mkdir -p log
python main.py

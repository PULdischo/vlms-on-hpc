# VLMs on HPC

This is a set of scripts that can be used to run text recognition (OCR/HTR) on Princeton's high-performance computing clusters.  

The main goal is to
- Download images from a IIIF endpoint
- Download an open-source model from HuggingFace Hub
- Recognize text in the images and save them as markdown

For Princeton faculty, staff, and students, you can request an account on Adroit [here](https://forms.rc.princeton.edu/registration/).

Regularly updated documentation on Adroit can be found [here](https://researchcomputing.princeton.edu/systems/adroit) 

To connect to the Adroit cluster 

```bash
ssh <username>@adroit.princeton.edu
``` 

the password is the same one you'd use for other CAS logins. You'll need to accept a Duo Push or other authentication.  If you're off campus, keep in mind that you must connect through the campus VPN.  

Alternatively, go to Adroit Cluster Shell Access from [myadroit.princeton.edu](https://myadroit.princeton.edu)

Once logged in, you'll be in your home directory on the login node. For me, it's `/home/aj7878` You have very limited space on the login node (server).

You will want to navigate to your folder in the shared network drive. For example, 
```bash
cd /scratch/network/<username>
```

Create a folder for your project and change to it
```bash
mkdir quiche && cd quiche
```

Once in your directory, you can clone this repository into your folder
```
git clone https://github.com/PULdischo/vlms-on-hpc.git .
```

now activate conda to manage Python dependencies
```bash 
module load anaconda3/2024.6
``` 
> note that a newer version may be available 

create a virtual enviornment 
```bash
conda env create -f conda_env.yml
```

The HPC node does not have access to the Internet, so you need to download all model and image files in advance on the login node. 

To do this: 
```bash
python fetch.py model <huggingface/repo-name>
```
then 
```bash
python fetch.py images <IIIF manifest URL>
```

All images will be saved in the img folder.  This is the same folder that will hold the markdown files. For example `0001.jpg` will have a `0001.md` file in the same folder. 

Before running your job, you'll want to look at and update the job.slurm file. 
To open an editor in the terminal, type `nano job.slurm`
The main things to note and ajust are:
- your job name
- the number of GPUs. Start with one. If you get a CUDA memory error, you can add them as needed. For example, `gpu:2` calls for two GPUs. 
- update the email for notifications (mail-user)

The file will look something like this: 

```bash
#!/bin/bash
#SBATCH --job-name=quiche        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=2G                 # total memory (RAM) per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:4
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=apjanco@princeton.edu

module purge
module load anaconda3/2024.6
conda activate vlm
HF_HUB_OFFLINE=1
python main.py
```

When you have the model downloaded, the images ready and everything is set to go
```bash
sbatch job.slurm
```

You can view the status of your running jobs here: https://myadroit.princeton.edu/pun/sys/dashboard/activejobs

As the markdown files are generated, you will see them appear in the img folder. `ls img/`

Once a job is completed, you will get an email report on the resources used and efficentcy of your job. Based on your utilization,  

For example: 

```bash
================================================================================
                              Slurm Job Statistics
================================================================================
         Job ID: 2556969
  NetID/Account: aj7878/pustaff
       Job Name: quiche
          State: TIMEOUT
          Nodes: 1
      CPU Cores: 1
     CPU Memory: 2GB
           GPUs: 4
  QOS/Partition: gpu-short/gpu
        Cluster: adroit
     Start Time: Wed Jul 2, 2025 at 11:08 AM
       Run Time: 02:00:01
     Time Limit: 02:00:00

                              Overall Utilization
================================================================================
  CPU utilization  [|||||||||||||||||||||||||||||||||||||||||||||||99%]
  CPU memory usage [||||||||||||||||||||||||||||||||||||||         77%]
  GPU utilization  [|||||||||                                      18%]
  GPU memory usage [|||||                                          10%]
```

To move you images and text off the Adroit servers, you can do the following 

Log in to Huggingface with your token
```bash 
hugginface-cli login
```
Then enter your token 
Now you can push all your files to HuggingFace Hub with 
```bash
python fetch.py to-hub <your HF username>/<new repo name> 
```
By default, your dataset is private. You can publish as public by adding `--public`

Further reading: 
- https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

- https://github.com/davidt0x/hf_tutorial
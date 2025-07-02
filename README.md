# VLMs on HPC

This is a set of scripts that can be used to run text recognition (OCR) on Princeton's high performance computing clusters.  

The main goal is to
- Download images from a IIIF endpoint
- Download an opensource model from HuggingFace Hub
- Recognize text in the images and save them as markdown


To connect to the Adroit cluster 
`ssh <username>@adroit.princeton.edu`
alternatively, go to Adroit Cluster Shell Access from [myadroit.princeton.edu](https://myadroit.princeton.edu)

You will want to navigate to your folder in the shared netword drive. For example, `cd /scratch/network/<username>`

Once in your directory, you can clone this repository
`git clone https://github.com/PULdischo/vlms-on-hpc.git`

now activate conda to manage Python dependencies
`module load anaconda3/2024.6` note that a newer version may be available 

create a virtual enviornment 
`conda env create -f conda_env.yml`

The HPC node does not have access to the Internet, so you need to download all model and image files in advance on the login node. 

To do this: 
`python fetch.py model`
then 
`python fetch.py images <IIIF manifest URL>`

All images will be saved in the img folder.  This is the same folder that will hold the markdown files. For example `0001.jpg` will have a `0001.md` file in the same folder. 

When you have the model downloaded, the images ready and everything is set to go
`sbatch job.slurm`

Further reading: 
- https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

- https://github.com/davidt0x/hf_tutorial
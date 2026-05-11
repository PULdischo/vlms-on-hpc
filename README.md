# VLMs on HPC

This is a set of scripts that can be used to run text recognition (OCR/HTR) on Princeton's high-performance computing clusters.  

The main goal is to
- Create a folder of images to process. These can be downloaded from a IIIF endpoint, a bunch of PDF files, HEIC files from your phone...
- Download an open-source model from HuggingFace Hub
- Recognize text in the images and save them as markdown, a spreadsheet or searchable static webpage.

For Princeton faculty, staff, and students, you can request an account on Adroit [here](https://forms.rc.princeton.edu/registration/).

Regularly updated documentation on Adroit can be found [here](https://researchcomputing.princeton.edu/systems/adroit) 

## Connect to Adroit

Start by going to [myadroit.princeton.edu](https://myadroit.princeton.edu)  
The password is the same one you'd use for other CAS logins. You'll need to accept a Duo Push or other authentication.   

If you're off campus, keep in mind that you must connect through the campus VPN.  

- Click on the Files dropdown menu and select the bottom option `/scratch/network/<your_username>`
- Then click on the `>_Open in Terminal` button
- type `wget https://raw.githubusercontent.com/PULdischo/vlms-on-hpc/refs/heads/main/setup.sh`
- then `bash setup.sh`

You can also connect over ssh if you prefer and run the same steps

```bash
ssh <username>@adroit.princeton.edu
``` 

## Prepare your workspace 

The HPC node does not have access to the Internet, so you need to download all model and image files in advance on the login node. You can find Hugging Face Hub models here: https://huggingface.co/models?pipeline_tag=image-text-to-text

To download the model: 
```bash
python fetch.py model <huggingface/repo-name> # default is "nanonets/Nanonets-OCR-s"
```
This command saves the location of your downloaded model snapshots in a file called "model_info.json".  You can also set the path to the model in `main.py`.

I find it helpful to do a test run on the login node to check for errors.  With your virtual enviornment activated, you can run `python main.py`. If everything is set up properly, you'll get an error from vLLM that it can't find the GPU (the login node doesn't have one).


## Image files or PDFs?
This project currently supports the processing of image collections from a IIIF endpoint like Princeton's [DPUL](https://dpul.princeton.edu/) or a folder of PDF files. 

### Image collections 

Many major libraries and museums support the International Image Interoperability Framework (IIIF). There's useful list of members of the IIIF community [here](https://iiif.io/guides/finding_resources/). In addition to providing a viewer for researchers, IIIF serves data about the collection to the web. You can access this data in a IIIF manifest. This is a package of metadata in JSON format that includes information about the materials, metadata and links for all of the images in the collection. 

For example, [this manuscript](https://www.loc.gov/resource/music.musapschmidt-10006011/?sp=1) from the Library of Congress includes a link to 

>  IIIF Presentation Manifest  
>  [Manifest (JSON/LD)](https://www.loc.gov/item/musapschmidt06012/manifest.json)  

If you click on the link, you'll go to: https://www.loc.gov/item/musapschmidt06012/manifest.json
That is the manifest's URI (Uniform Resource Identifier) a link that returns the data you need.

You may also find the IIIF logo.  !["IIIF logo. Alternating blue and red letters that spell IIIF"](https://iiif.io/assets/images/logos/logo-sm.png)  

[This item](https://eap.bl.uk/archive-file/EAP699-9-1) from the British Library's Endangered Archives Programme has the logo in the lower right corner. If you click on it, you get the JSON data for the IIIF manifest. 
For this example, the URI looks like this: https://eap.bl.uk/archive-file/EAP699-9-1/manifest?manifest=https%3A//eap.bl.uk/archive-file/EAP699-9-1/manifest

It's up to you to find relevant collections for your work and to find the IIIF manifest's URI. But, once you have that, you can download all the images and their metadata.  

This project includes a script to download a manifest and associated images. To run it, type:

```bash
python fetch.py images <IIIF manifest URI>
```

All images will be saved in the img folder.  This is the same folder that will hold the markdown files. For example `0001.jpg` will have a `0001.md` file in the same folder. 

## Running your first compute job 

Before running your job, you'll want to look at and update the job.slurm file (there is a job.pdf.slurm file for pdf processing). 
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
module load anaconda3/2025.6
conda activate vlm
HF_HUB_OFFLINE=1
python main.py
```

When you have the model downloaded, the images ready and everything is set to go
```bash
sbatch job.slurm
```

You can view the status of your running jobs here: https://myadroit.princeton.edu/pun/sys/dashboard/activejobs

As the markdown files are generated, you will see them appear in the markdown folder.

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
You can get the same information by typing `jobstats 2556969`

### PDFs 

Processing PDFs is very similar to processing images.  The `main_pdf.py` script is very similar to `main.py`. It loads files in a `pdfs` directory and checks that they are valid PDF files. Note that they don't have to have the `.pdf` suffix. We're checking the contents of the file, not its name. We then convert each page of the PDF into an image in memory.  The intermediate images aren't saved to disk. One important parameter to know about is image dpi. On line 85 of `main_pdf.py`, you'll find 

```python
pix = page.get_pixmap(dpi=100)  
```

The dpi setting controls the size and detail of the generated image. For most typed documents, 50-100 dots per square inch is more than sufficent. Relatively low resolution uses less memory and is significantly faster.  The image is converted into image tokens, so a smaller image has fewer tokens.  If the resolution is too high, the model may not be able to fit all of the image and text tokens into its context length.

For example, 
```bash 
Error opening pdfs/156002856211550797464202220250174197763: The decoder prompt (length 10896) is longer than the maximum model length of 8192. 
```

The main_pdf script will process any files that you put in `pdfs` and save the result in a `markdown` folder. The files will have the same name, but a different suffix.  

Before running your job, you'll want to look at and update the job.pdf.slurm file. 
To open an editor in the terminal, type `nano job.pdf.slurm`
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
#SBATCH --mem=20G                 # total memory (RAM) per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=<username>@princeton.edu

module purge
module load anaconda3/2025.6
conda activate vlm
HF_HUB_OFFLINE=1
python main_pdf.py
```

When you have the model downloaded, the images ready and everything is set to go
```bash
sbatch job.pdf.slurm
```

## Moving your files from Adroit

To move you images and text off the Adroit servers, you can do the following 

### Create a HuggingFace Dataset and push to the hub
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

### Use SCP
While connected to Adroit, find the files that you'd like to transfer.
For example, my markdown output files might be in: 
`/scratch/network/myusername/quice/markdown`

Open the terminal on your computer. 
You can copy the folder from Adroit to your computer with the `scp` command. 
`scp <origin> <destination>`
for example, 
`scp myusername@adroit.princeton.edu:/scratch/network/myusername/quice/markdown/ ~/Downloads/`

Further reading: 
- https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face

- https://github.com/davidt0x/hf_tutorial

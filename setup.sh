cd /scratch/network/$USER
git clone https://github.com/PULdischo/vlms-on-hpc.git
module load anaconda3/2025.6
conda env create -f /scratch/network/$USER/vlms-on-hpc/conda_env.yml

rsync -avu $HOME/.conda /scratch/network/$USER/
rm -Rf $HOME/.conda
cd $HOME && ln -s /scratch/network/$USER/.conda .conda

rsync -avu $HOME/.triton /scratch/network/$USER/
rm -Rf $HOME/.triton
cd $HOME && ln -s /scratch/network/$USER/.triton .triton
echo "export HF_HOME=/scratch/network/$USER/.cache/huggingface/" >> $HOME/.bashrc
source $HOME/.bashrc
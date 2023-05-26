#!/bin/bash

install_dir=$1
primary_user=$2

miniconda_version=latest
installer=https://repo.anaconda.com/miniconda/Miniconda3-$miniconda_version-Linux-x86_64.sh

# export statement seems to have no effect, so add to PATH
# via Dockerfile $ENV command.
# export PATH=$install_dir/bin:$PATH

# download installer, run it, then remove it
sudo wget --quiet $installer -O ./miniconda.sh
sudo chmod 300 ./miniconda.sh
sudo bash ./miniconda.sh -b -p "$install_dir"
sudo rm ./miniconda.sh

"$install_dir"/bin/pip install python-dotenv

# configure bash and zsh to run conda
echo ". $install_dir/etc/profile.d/conda.sh" \
  >> /home/"$primary_user"/.bashrc
echo ". $install_dir/etc/profile.d/conda.sh" \
  >> /home/"$primary_user"/.zshrc
conda init bash
conda init zsh

# note: miniconda permission mode = 0755
sudo chown -R "$primary_user":"$primary_user" "$install_dir"

# add mamba for faster package installs (in images that use this as base)
conda install mamba -n base -c conda-forge
mamba update --name base --channel defaults conda

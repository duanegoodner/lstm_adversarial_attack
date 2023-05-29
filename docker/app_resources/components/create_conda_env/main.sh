#!/bin/bash


env_yml=$1
env_path=$2
user=$3
conda_dir=$4

sudo mkdir "$env_path"
sudo chown "$user":"$user" "$env_path"
mamba env create --prefix "$env_path" --file "$env_yml" --force
mamba clean --all --yes --force-pkgs-dirs
find "$conda_dir" -follow -type f -name '*.a' -delete
find "$conda_dir" -follow -type f -name '*.pyc' -delete
sudo echo "conda activate $env_path" >> /home/"$user"/.bashrc
sudo echo "conda activate $env_path" >> /home/"$user"/.zshrc

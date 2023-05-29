#!/bin/bash


username=$1
pubkey_file=$2

sudo echo $(sudo cat $pubkey_file)
sudo useradd -s /bin/bash -m "$username"
sudo mkdir -p /home/"$username"/.ssh
sudo touch -a /home/"$username"/.ssh/authorized_keys
sudo chown -R "$username":"$username" /home/"$username"/.ssh
sudo chmod 600 /home/"$username"/.ssh/authorized_keys
sudo chmod 700 /home/"$username"/.ssh
sudo cat "$pubkey_file" | sudo tee /home/"$username"/.ssh/authorized_keys >/dev/null
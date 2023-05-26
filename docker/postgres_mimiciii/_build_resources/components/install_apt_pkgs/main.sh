#!/bin/bash

pkglist=$1

sudo apt-get update --fix-missing
sudo apt-get install -y $(cat $pkglist)
sudo apt-get autoremove -y
sudo apt-get clean -y
sudo rm -rf /var/lib/apt/lists/*



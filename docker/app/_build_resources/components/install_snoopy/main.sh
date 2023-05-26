#!/bin/bash

# install with wget instead of apt b/c apt install
# requires extra yes response that can't send via cmd line

installer_url=https://github.com/a2o/snoopy/raw/install/install/install-snoopy.sh
installer_name=install-snoopy.sh
sudo mkdir ./build_snoopy
sudo wget --quiet $installer_url -O ./build_snoopy/$installer_name
sudo chmod +x ./build_snoopy/$installer_name
sudo ./build_snoopy/$installer_name stable
sudo rm -rf ./build_snoopy


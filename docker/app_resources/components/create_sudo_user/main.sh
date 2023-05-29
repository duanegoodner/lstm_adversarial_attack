#!/bin/bash

# Run as: root

create_sudo_user() {
  new_username=$1


  apt-get update --fix-missing
  apt-get install -y sudo
  apt-get autoremove -y
  apt-get clean -y
  rm -rf /var/lib/apt/lists/*
  useradd -s /bin/bash -m "$new_username"
  mkdir -p /home/"$new_username"
  echo "$new_username" ALL=\(root\) NOPASSWD:ALL \
    > /etc/sudoers.d/"$new_username"
  chmod 0440 /etc/sudoers.d/"$new_username"
}

create_sudo_user "$1" "$2"
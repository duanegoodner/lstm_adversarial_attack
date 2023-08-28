#!/bin/bash

# Run as: root

create_sudo_user() {
  new_username=$1
  new_uid=$2
  new_gid=$3


  apt-get update --fix-missing
  apt-get install -y sudo
  apt-get autoremove -y
  apt-get clean -y
  rm -rf /var/lib/apt/lists/*
  groupadd -g "$new_gid" "$new_username"
  useradd -s /bin/bash -m -u "$new_uid" -g "$new_gid" "$new_username"
  mkdir -p /home/"$new_username"
  echo "$new_username" ALL=\(root\) NOPASSWD:ALL \
    > /etc/sudoers.d/"$new_username"
  chmod 0440 /etc/sudoers.d/"$new_username"
}

create_sudo_user "$1" "$2" "$3"
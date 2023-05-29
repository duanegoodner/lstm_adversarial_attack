#!/bin/bash


reset_permissions() {
  # ensure permissions in a safe state (in case of container restart)
  sudo gpasswd trusted -M ''
  sudo usermod -a -G "$WORK_GROUP" "$PRIMARY_USER"
  sudo chmod 755 -R "$CONTAINER_PROJECT_ROOT"
  sudo setfacl -R -bn "$CONTAINER_PROJECT_ROOT"
}
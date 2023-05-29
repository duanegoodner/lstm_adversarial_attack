#!/bin/bash


stop_jupyter_lab() {
  $CONDA_BIN_DIR/jupyter lab stop
}

reset_permissions() {
  # ensure permissions in a safe state (in case of container restart)
  sudo gpasswd trusted -M ''
  sudo usermod -a -G "$WORK_GROUP" "$PRIMARY_USER"
  sudo chmod 755 -R "$WORK_DIR"
  sudo setfacl -R -bn "$WORK_DIR"
}
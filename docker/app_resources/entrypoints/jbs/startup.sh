#!/bin/bash


start_rsyslogd() {
  rsyslogd_pid=$(pidof rsyslogd)
  if [ -n "$rsyslogd_pid" ]; then
    echo rsyslogd is already running. pid="$rsyslogd_pid"
  else
    echo starting rsyslogd
    sudo rsyslogd
    echo rsyslogd now running. pid="$(pidof rsyslogd)"
  fi
}

set_initial_permissions() {
  sudo gpasswd trusted -M ''
  sudo usermod -a -G "$WORK_GROUP" "$PRIMARY_USER"
  sudo chown -R "$PRIMARY_USER":"$WORK_GROUP" "$CONTAINER_PROJECT_ROOT"
  sudo chmod 775 -R "$CONTAINER_PROJECT_ROOT"
}

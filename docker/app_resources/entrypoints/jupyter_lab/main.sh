#!/bin/bash


trap 'true' SIGTERM
trap 'true' SIGINT


$CONDA_BIN_DIR/jupyter lab --no-browser --ip 0.0.0.0 &
tail -f /dev/null &
wait $!

$CONDA_BIN_DIR/jupyter lab stop


echo closing
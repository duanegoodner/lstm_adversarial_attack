#!/bin/bash


trap 'true' SIGTERM
trap 'true' SIGINT

tail -f /dev/null &
wait $!

echo closing
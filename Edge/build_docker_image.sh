#!/usr/bin/env bash
sudo docker build -t project -f Dockerfile.keras .
sudo docker rmi $(sudo docker images -f "dangling=true" -q)

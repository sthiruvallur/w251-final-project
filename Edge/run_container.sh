#!/usr/bin/env bash

sudo docker run -it --rm --name hd -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY --privileged -device=/dev/video0:/dev/video0 -v /data/project/hands:/tmp/hands -p 8888:8888 project

sudo docker exec -it hd bash

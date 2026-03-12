#!/bin/bash
IMAGE_NAME=sam3d
DOCKER_BUILDKIT=1 docker build --progress plain -f Dockerfile -t pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME .
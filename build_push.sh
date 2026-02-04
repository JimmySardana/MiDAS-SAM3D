#!/bin/bash
IMAGE_NAME=sam3_test
docker login pc3163.igd.fraunhofer.de:4567
docker build -f Dockerfile -t pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME .
docker push pc3163.igd.fraunhofer.de:4567/$IMAGE_NAME
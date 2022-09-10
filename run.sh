CONTAINER_NAME=bertflix_1

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

docker run -it --name ${CONTAINER_NAME} bertflix:1.0
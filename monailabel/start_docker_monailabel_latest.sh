version=${1-"latest"}
name="monailabel_$version"

if [ ! "$(docker ps -q -f name=$name)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$name)" ]; then
        # cleanup
        echo "  "
        echo "Found container <$name>. Removing..."
        docker rm $name
    fi
    # run your container
    echo "  "
    echo "Starting container <$name>."
    echo "Mapping all ports."
    docker run -it \
        --user $(id -u):$(id -g) \
        -v /etc/passwd:/etc/passwd:ro \
        -v /home/$(id -u -n):/home/$(id -u -n) \
        --gpus all \
        --shm-size=4g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v /workdir:/data/ \
        --network=host \
        --ipc=host \
        --workdir /data \
        --name $name \
        projectmonai/monailabel:$version
else
    echo "  "
    echo "Found container <$name>. Attaching..."
    docker exec -it $name --workdir /data/Projects /bin/bash
fi
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 name [container]"
    exit
elif [ $# -eq 1 ]; then
    name=$1
    container=tlab/cuda
else
    name=$1
    container=$2
fi


echo "running $container ... (the name of the container is $name)"

sudo docker run --gpus all --rm --name $name \
    --shm-size=1024m \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /home:/home \
    -v /mnt/tlab-nas:/mnt/tlab-nas \
    -v /mnt/data:/mnt/data \
    -u $(id -u $USER):$(id -g $USER)  -it  $container  bash

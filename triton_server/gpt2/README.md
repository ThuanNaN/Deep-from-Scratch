```
docker pull nvcr.io/nvidia/tritonserver:23.07-py3

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3



```
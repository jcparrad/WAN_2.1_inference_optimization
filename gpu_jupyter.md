# 🐳 Build and Run GPU JupyterLab Docker Container

This guide walks you through building a Docker image with JupyterLab and launching it with a mounted volume for persistent storage.

Download the models as set in the file  [gpu_fastapi_docker_inference ](gpu_fastapi_docker_inference.md)


---

## 🛠️ Step 1: Build the Docker Image

Build the Docker image using the `DockerfileJupyter` file:



```bash
sudo docker build -f DockerfileJupyter -t gpu-jupyter .
```

---

## ▶️ Step 2: Run the Container

Run the container with port mapping and volume binding:

```bash
sudo docker run --rm -it --gpus all --ipc=host \
    -p 8888:8888 \
    -p 8000:8000 \
    -v "$(pwd)/experiments":/workspace/experiments \
    -v "$(pwd)/inference_optimization":/workspace/inference_optimization \
    -v "$(pwd)/model_repo/Wan2.1-T2V-14B":/workspace/Wan2.1-T2V-14B \
    -v "$(pwd)/model_repo/Wan2.1-T2V-1.3B":/workspace/Wan2.1-T2V-1.3B \
    -v "$(pwd)/model_repo/Wan2.1-I2V-14B-480P":/workspace/Wan2.1-I2V-14B-480P \
    -v "$(pwd)/model_repo/Wan2.1-I2V-14B-720P":/workspace/Wan2.1-I2V-14B-720P \
    -v "$(pwd)/model_repo/Wan2.1-VACE-14B":/workspace/Wan2.1-VACE-14B \
    -v "$(pwd)/model_repo/Wan2.1":/workspace/Wan2.1 \
    gpu-jupyter

```


> 💡 This command:
> - Maps port `8888` so JupyterLab is accessible on `http://<ip>:8888/lab`
> - Mounts the local folder `experiments` to `/workspace/experiments` inside the container
> - Ensures any files created in Jupyter are saved persistently on your host

---

## ✅ Summary

| Feature           | Description                                                  |
|------------------|--------------------------------------------------------------|
| **Image name**    | `gpu-jupyter`                                                |
| **Jupyter port**  | `8888`                                                       |
| **Mounted folder**| `experiments/experiments_jupyter_notebook` → `/workspace/experiments_jupyter` |

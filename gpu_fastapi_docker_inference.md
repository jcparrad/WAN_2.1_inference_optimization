# 📽️ Wan2.1 Video Generation: Setup & Deployment Guide

This guide provides a step-by-step workflow to **deploy a FastAPI-based video generation service** using **Wan2.1** models from Hugging Face with **GPU acceleration** inside a Docker container on **Oracle Linux 9** (OL9).

---

## ✅ 1. Objective: What We’re Building

We aim to:

- Set up a GPU-enabled environment on Oracle Linux 9
- Install all required dependencies (Python, Docker, Hugging Face CLI, Git)
- Download Wan2.1 models (T2V and I2V)
- Run a FastAPI app (via Docker) that uses Wan2.1 to generate videos

---

## ⚙️ 2. Environment Setup: How to Prepare the Machine

### 🐍 Install Python 3.9, Pip, Git, and Hugging Face CLI

> Oracle Linux 9 usually comes with Python 3.9+ preinstalled. This section ensures compatibility.

```bash
# Update system packages
sudo dnf update -y

# Enable EPEL for OL9
sudo dnf install -y oracle-epel-release-el9
sudo dnf config-manager --set-enabled ol9_developer_EPEL

# Install Python 3.9 and pip
sudo dnf install -y python3.9 python3.9-pip

# Check versions
python3.9 --version
pip3.9 --version

# Set python3/pip to point to version 3.9
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
sudo ln -sf /usr/bin/pip3.9 /usr/bin/pip

# Upgrade pip and install Hugging Face CLI
pip install --upgrade pip
pip install "huggingface_hub[cli]"

# Install Git
sudo yum install -y git
```

### 🔥 Allow FastAPI Port Through Firewall

```bash
sudo firewall-cmd --zone=public --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

Also, make sure the **ingress rule** for port `8000` is open in the **VCN/subnet**.

---

## 📦 3. Download Wan2.1 Models and Code

Navigate to your working directory (e.g., inside `docker_oci_gpu`) and create a folder for the models:

```bash
mkdir -p docker_oci_gpu/model_repo
cd docker_oci_gpu/model_repo
```

### Download the models and repo:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir ./Wan2.1-VACE-14B


git clone https://github.com/Wan-Video/Wan2.1.git
```

---

## 🐳 4. Build Docker Image for FastAPI App

Navigate back to `docker_oci_gpu/` where the `Dockerfile` and `main.py` are located:

```bash
cd ../  # Go back to docker_oci_gpu
sudo docker build -t gpu-app .
```

This builds a Docker image named `gpu-app` with GPU + FastAPI + model code.

---

## 🚀 5. Run the Docker Container

Use the following command to start the FastAPI server with access to all GPUs and mount all required volumes:

```bash
sudo docker run --rm -it --gpus all --ipc=host \
  -p 8000:8000 \
  -v ./model_repo/Wan2.1-T2V-14B:/workspace/Wan2.1-T2V-14B \
  -v ./model_repo/Wan2.1-T2V-1.3B:/workspace/Wan2.1-T2V-1.3B \
  -v ./model_repo/Wan2.1-I2V-14B-480P:/workspace/Wan2.1-I2V-14B-480P \
  -v ./model_repo/Wan2.1-I2V-14B-720P:/workspace/Wan2.1-I2V-14B-720P \
  -v ./model_repo/Wan2.1:/workspace/Wan2.1 \
  -v ./main.py:/workspace/main.py \
  gpu-app uvicorn main:app --host 0.0.0.0 --port 8000
```

### Key Flags:

- `--gpus all`: Enables access to all available GPUs
- `--ipc=host`: Allows shared memory required by PyTorch
- `-v`: Mount local files/folders to the container
- `-p 8000:8000`: Maps FastAPI port to the host

---
## 🛠️ 6. Run Inference Using Predefined Scripts

You can use the provided shell scripts to run inference for specific models and resolutions:

```bash

chmod +x generate_video_480_i2v.sh
./generate_video_480_i2v.sh

chmod +x generate_video_480_t2v.sh
./generate_video_480_t2v.sh

chmod +x generate_video_720_i2v.sh
./generate_video_720_i2v.sh

chmod +x generate_video_720_t2v.sh
./generate_video_720_t2v.sh

chmod +x generate_video_480_t2v_1_3B.sh
./generate_video_480_t2v_1_3B.sh

```

> Each script sends a `curl` request to the FastAPI server with the appropriate parameters for model type and resolution.

⚠️ **Important**:  
- Be sure to **specify the correct server IP address** inside each script.  
- For image-to-video (`i2v`) scripts, make sure to **set the input image path or URL** in the payload.

---

## 🧾 Notes:

- Replace `<ip>` in scripts or requests with your public IP if calling from external machines.
- Output videos are saved using the name defined in `save_file` (default: `custom_filename.mp4`).

---

## 🔗 References

1. [Wan2.1-I2V-14B-480P on Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
2. [Wan2.1 GitHub Repository](https://github.com/Wan-Video/Wan2.1)
https://stable-diffusion-art.com/wan-vace-ref/

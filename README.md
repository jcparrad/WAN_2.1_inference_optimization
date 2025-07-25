# 📽️ Wan2.1 Video Generation: Setup & Deployment Guide

This guide provides a step-by-step workflow to optimice the inference of the model wan 2.1 for the task VACE over H100.8 GPUs inside a Docker container on **Oracle Linux 9** (OL9).

---

## ✅ 1. Objective: What We’re Building

We aim to:

- Set up a GPU-enabled environment on Oracle Linux 9
- Install all required dependencies (Python, Docker, Hugging Face CLI, Git)
- Download Wan2.1 models 
- Run experiments on jupyter notebook

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


```bash
mkdir -p model_repo
cd model_repo
```

### Download the models and repo:

Download only the VACE version

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir ./Wan2.1-VACE-14B


git clone https://github.com/Wan-Video/Wan2.1.git
```


copy the files generate_vace_1.py and generate_vace_2.py into model_repo/Wan2.1

```bash
cp generate_vace_1.py model_repo/Wan2.1/
cp generate_vace_2.py model_repo/Wan2.1/
```

---

## 🐳 4. Build Docker Image for FastAPI App

Build the Docker image using the `DockerfileJupyter` file:



```bash
sudo docker build -f DockerfileJupyter -t gpu-jupyter .
```

---

## ▶️ Step 5: Run the Container

Run the container with port mapping and volume binding:

```bash
sudo docker run --rm -it --gpus all --ipc=host \
    -p 8888:8888 \
    -p 8000:8000 \
    -v "$(pwd)/experiments":/workspace/experiments \
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



## Run the experiments in jupyter notebook

Follow the instructions in the notebook [inference_optimization.ipynb](inference_optimization.ipynb) to reproduce the experiments.

---

## 🧪 Best Experiment Summary

**Experiment 3** yielded the best results:

- **Processes**: 2 parallel processes  
- **GPUs Used**: 4 H100 GPUs per process (total of 8 GPUs)  
- **Tasks**: 10 video generation tasks per process  
- **Total Time**: 1850 seconds  
- **Machine Cost**: 80 USD/hour for H100.8 bare metal

### 💰 Cost Analysis

- **Total Cost**:  
  80 USD/hour ÷ 3600 seconds/hour × 1850 seconds = 41.1 USD

- **Videos Produced**: 20  
- **Cost per 5-second video**:  
  41.1 USD ÷ 20 videos = 2.055 USD per video

- **Cost per second of video**:  
  2.055 USD ÷ 5 seconds = 0.41 USD per second

---

## 🔬 Hypothesis for Future Testing

If using H200.8 GPU, the cost per second of video generation could be reduced by half:

- **Expected Cost per Second**:  
  0.41 USD ÷ 2 = 0.205 USD


## 🔗 References

1. [Wan2.1-I2V-14B-480P on Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
2. [Wan2.1 GitHub Repository](https://github.com/Wan-Video/Wan2.1)
https://stable-diffusion-art.com/wan-vace-ref/

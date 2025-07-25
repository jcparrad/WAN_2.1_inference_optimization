# Inference Optimization Setup and Experiments on WAN 2.1 – VACE Task

This folder contains the setup and experiments aimed at achieving the best possible price/performance for running 5-second videos on H100.8 bare metal machines on OCI.

---

## 🔧 Environment Setup

Set up the environment as defined in [gpu_jupyter.md](../gpu_jupyter.md) to prepare the inference environment.

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

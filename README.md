# wllm
从0到1学习大模型推理框架

AI Infra Guide官方推理框架教程
https://github.com/caomaolufei/AIInfraGuide

# Installation
This project was validated with Python 3.11, a CUDA-capable NVIDIA GPU, and ffmpeg.

- Create an environment.
```text
conda create -n wllm-omni python=3.11 -y
conda activate wllm-omni
```

- Install a CUDA-enabled PyTorch build that matches your system.
Example for CUDA 12.1:
```text
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your machine uses a different CUDA version, use the selector on the official PyTorch site instead of this example.

- Install system ffmpeg.
```text
sudo apt-get update
sudo apt-get install -y ffmpeg
```

- Install this project and the Hugging Face CLI.
This project currently depends on the Wan pipeline from the diffusers main branch, so pip install -e . will fetch diffusers from GitHub automatically.
```text
python -m pip install -e .
python -m pip install huggingface_hub
```

# Model Download
This repo expects the Wan model under ./models/Wan2.2-TI2V-5B-Diffusers.

Create the directory and download the official Diffusers weights:
```text
mkdir -p models
huggingface-cli download --resume-download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers \
  --local-dir-use-symlinks False
```
The repository already includes a demo image at ./assets/image.png, so no extra asset download is required for the default example.

# Quick Start
After the model is downloaded, this command should run end-to-end:
```text
CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/image.mp4
```

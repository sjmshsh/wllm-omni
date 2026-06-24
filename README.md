# wllm-omni

A lightweight AI inference framework playground for learning omni runtime design from first principles.

Current V0 focus:

- Wan2.2 image-to-video diffusion execution
- mini vLLM-Omni-style AR -> diffusion stage composition
- unified `ModelRunner` with paradigm-specific executors
- prepare-stage profiling and cache infrastructure

Reference direction:

- vLLM / SGLang: scheduler, runner, executor, KV/cache-oriented inference ideas
- vLLM-Omni / SGLang-Omni: stage graph, connector, multi-model runtime composition

## Architecture V0

```text
MiniOmniRuntime
  ├── ARStage
  │     └── AREngine
  │           └── RequestScheduler
  │                 └── ModelRunner
  │                       └── ARExecutor
  │                             └── TransformersARPipeline
  │
  ├── Connector
  │     AR text output + image + sampling params
  │     -> diffusion OmniRequest
  │
  └── DiffusionStage
        └── DiffusionEngine
              └── StepScheduler
                    └── ModelRunner
                          └── DiffusionExecutor
                                └── Wan22I2VPipeline
```

The runtime composes stages. Each stage owns its engine. Engines own scheduling and runner execution. The shared `ModelRunner` routes work to paradigm-specific executors.

## Installation

Validated with Python 3.11, a CUDA-capable NVIDIA GPU, and ffmpeg.

```text
conda create -n wllm-omni python=3.11 -y
conda activate wllm-omni
```

Install a CUDA-enabled PyTorch build that matches your system. Example for CUDA 12.1:

```text
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install ffmpeg:

```text
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Install the project:

```text
python -m pip install -e .
python -m pip install huggingface_hub
```

## Model Download

Wan diffusion model:

```text
mkdir -p models
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

Optional AR model for mini-omni mode:

```text
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./models/Qwen2.5-0.5B-Instruct
```

The repo includes a demo image at `./assets/image.png`.

## Diffusion-only Run

```text
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/image.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16
```

## Mini-Omni AR -> Diffusion Run

```text
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/mini_omni.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16 \
  --mini-omni \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-max-new-tokens 64
```

Expected trace:

```text
[wllm-omni][mini-omni] stages=ar.prompt_bridge -> diffusion.wan22_i2v
[wllm-omni][mini-omni] ar.model=... ar.mode=transformers_causal_lm ...
[wllm-omni][mini-omni] diffusion.bridge=ar_text_to_diffusion_prompt ...
```

## AR-only Run

```text
python example_wan22_i2v.py \
  --mini-omni \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-only \
  --prompt "A cat wearing sunglasses sits on a surfboard at the beach."
```

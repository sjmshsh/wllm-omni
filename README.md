# wllm-omni

`wllm-omni` 是一个用于学习 AI Infra / 多模态推理框架的轻量项目

## 当前 V0 定位

当前 V0 是一个 **mini vLLM-Omni 风格的单进程原型**

已经支持：

- Wan2.2 image-to-video diffusion 执行
- 可选 profiler
- Wan prepare-stage cache
- VAE dtype policy
- Qwen CausalLM AR stage
- `AR -> Diffusion` 的 mini-omni stage 编排
- 显式 `StageGraph`
- 顶层 `StageScheduler`
- 独立 `Connector`
- 统一 `ModelRunner`
- 差异化 `ARExecutor` / `DiffusionExecutor`

还没有支持：

- 动态多分支 `StageGraph`
- stage-level batching
- AR prefill / decode 分离
- KV cache
- streaming token 输出
- 多 session 调度
- pipeline overlap
- 多 GPU / 分布式 stage serving

## V0 架构

```text
MiniOmniRuntime
  └── StageScheduler
        └── StageGraph
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

## 安装

推荐环境：

- Python 3.11
- CUDA-capable NVIDIA GPU
- ffmpeg

创建环境：

```text
conda create -n wllm-omni python=3.11 -y
conda activate wllm-omni
```

安装匹配 CUDA 的 PyTorch。下面是 CUDA 12.1 示例：

```text
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装 ffmpeg：

```text
sudo apt-get update
sudo apt-get install -y ffmpeg
```

安装项目：

```text
python -m pip install -e .
python -m pip install huggingface_hub
```

## 模型下载

Wan diffusion 模型：

```text
mkdir -p models
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

mini-omni AR stage 使用的 Qwen 模型：

```text
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./models/Qwen2.5-0.5B-Instruct
```

## Diffusion-only 运行

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

## Mini-Omni AR -> Diffusion 运行

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

预期 trace：

```text
[wllm-omni][mini-omni] stages=ar.prompt_bridge -> diffusion.wan22_i2v
[wllm-omni][mini-omni] graph=ar.prompt_bridge -> diffusion.wan22_i2v
[wllm-omni][mini-omni] ar.model=... ar.mode=transformers_causal_lm ...
[wllm-omni][mini-omni] diffusion.bridge=ar_text_to_diffusion_prompt ...
```

其中：

- `ar.elapsed_ms` 表示 AR stage 执行耗时
- `diffusion.load_ms` 表示首次加载 diffusion engine 的耗时
- `diffusion.elapsed_ms` 表示 diffusion stage 真正执行耗时
- Wan profiler 里的 `forward.total` 是 Wan 内部请求执行耗时

## AR-only 运行

```text
python example_wan22_i2v.py \
  --mini-omni \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-only \
  --prompt "A cat wearing sunglasses sits on a surfboard at the beach."
```

# wllm-omni

`wllm-omni` 是一个用于学习 AI Infra / 多模态推理框架的轻量项目。当前目标不是复刻完整 vLLM-Omni，而是实现一个可运行、可扩展、便于学习的 mini vLLM-Omni 风格 runtime。

## 当前定位

当前版本是 **类 vLLM-Omni 的单进程 Stage Pipeline 原型**。

它已经从单一 Wan Diffusion runner 演进为：

```text
PipelineConfig / PipelineRegistry
  -> StageGraph
  -> StageScheduler
  -> Stage
  -> Engine
  -> Scheduler
  -> ModelRunner
  -> Executor
```

当前支持：

- Wan2.2 image-to-video diffusion 执行
- Qwen CausalLM AR stage
- AR prefill / decode 最小执行边界
- AR 内部 `past_key_values` KV cache 推进
- `ar_text` / `wan_i2v` / `qwen_to_wan_i2v` 三种固定 pipeline
- 显式 `StageGraph`
- 顶层 `StageScheduler`
- 独立 `Connector`
- 统一 `ModelRunner`
- 差异化 `ARExecutor` / `DiffusionExecutor`
- Wan profiler
- Wan prepare-stage cache
- VAE dtype policy

当前不支持：

- deploy YAML / stage override 配置
- 根据 model_type 自动选择 pipeline
- stage-level batching
- 调度层 AR KV cache 管理
- AR continuous batching
- streaming token 输出
- 多 session 调度
- pipeline overlap
- diffusion step execution
- 多 GPU / 多进程 / 分布式 stage serving

## 架构

```text
MiniOmniRuntime
  ├── PipelineRegistry
  │     ├── ar_text
  │     ├── wan_i2v
  │     └── qwen_to_wan_i2v
  │
  └── StageScheduler
        └── StageGraph
              ├── ARStage
              │     └── AREngine
              │           └── RequestScheduler
              │                 └── ModelRunner
              │                       └── ARExecutor
              │                             └── TransformersARPipeline
              │                                   ├── prefill
              │                                   ├── decode
              │                                   └── finalize
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

## 和 vLLM-Omni 的关系

当前项目在架构方向上对齐 vLLM-Omni，但只是单进程、学习版、最小闭环。

| vLLM-Omni 概念                | wllm-omni 当前对应                       | 当前状态                         |
| ----------------------------- | ---------------------------------------- | -------------------------------- |
| PipelineConfig                | `PipelineConfig`                       | 已有固定拓扑                     |
| Pipeline registry             | `PipelineRegistry`                     | 已有手动注册                     |
| Deploy config                 | 暂无                                     | 仍使用全局`EngineConfig`       |
| Stage topology                | `StageGraph`                           | 已有 DAG 骨架                    |
| Orchestrator / stage runtime  | `MiniOmniRuntime` + `StageScheduler` | 单进程顺序执行                   |
| Stage transition              | `StageConnector`                       | 已有最小 connector               |
| Stage input processor         | `ARToDiffusionConnector`               | 只有 AR text -> diffusion prompt |
| Per-stage engine              | `AREngine` / `DiffusionEngine`       | 已有                             |
| Per-stage scheduler           | `RequestScheduler` / `StepScheduler` | 已有请求级调度                   |
| Worker / runner               | `ModelRunner`                          | 已有统一 runner                  |
| Paradigm-specific model logic | `ARExecutor` / `DiffusionExecutor`   | 已有                             |
| Distributed connector         | 暂无                                     | 目前只在进程内传对象             |
| Stage-level batching          | 暂无                                     | 后续工作                         |
| AR KV cache / streaming       | `past_key_values` 内部推进              | 暂无调度层 KV / streaming        |
| Diffusion step execution      | 暂无                                     | 后续工作                         |

结论：当前架构已经是 **类 vLLM-Omni 的 pipeline runtime 雏形**，但还不是完整 vLLM-Omni。差距主要在配置系统、批处理、KV cache、streaming、分布式 connector 和 diffusion step execution。

## Pipeline 选择

当前 pipeline 是模型/配置级固定拓扑，不是根据 prompt 动态生成 graph。

```text
ar_text
  ar.prompt_bridge

wan_i2v
  diffusion.wan22_i2v

qwen_to_wan_i2v
  ar.prompt_bridge -> diffusion.wan22_i2v
```

其中：

- `ar_text` 对应单独 AR text pipeline
- `wan_i2v` 对应单独 Wan I2V diffusion pipeline
- `qwen_to_wan_i2v` 对应 AR prompt bridge + Wan I2V diffusion pipeline
- prompt / image / sampling params 是 graph 的输入，不负责临时规划 graph

## 安装

推荐环境：

- Python 3.11
- CUDA-capable NVIDIA GPU
- ffmpeg

创建环境：

```bash
conda create -n wllm-omni python=3.11 -y
conda activate wllm-omni
```

安装匹配 CUDA 的 PyTorch。下面是 CUDA 12.1 示例：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装 ffmpeg：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

安装项目：

```bash
python -m pip install -e .
python -m pip install huggingface_hub
```

## 模型下载

Wan diffusion 模型：

```bash
mkdir -p models
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

mini-omni AR stage 使用的 Qwen 模型：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./models/Qwen2.5-0.5B-Instruct
```

## 运行示例

### AR-only

```bash
python example_wan22_i2v.py \
  --pipeline ar_text \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --prompt "A cat wearing sunglasses sits on a surfboard at the beach."
```

### Diffusion-only

```bash
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/pipeline_wan_i2v.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16 \
  --pipeline wan_i2v
```

### AR -> Diffusion

```bash
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/pipeline_qwen_to_wan.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16 \
  --pipeline qwen_to_wan_i2v \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-max-new-tokens 64
```

预期 trace：

```text
[wllm-omni][mini-omni] request_id=... pipeline=qwen_to_wan_i2v stages=ar.prompt_bridge -> diffusion.wan22_i2v
[wllm-omni][mini-omni] graph=ar.prompt_bridge -> diffusion.wan22_i2v
[wllm-omni][mini-omni] ar.model=... ar.mode=transformers_causal_lm ...
[wllm-omni][mini-omni] diffusion.bridge=ar_text_to_diffusion_prompt ...
```

其中：

- `ar.elapsed_ms` 表示 AR stage 执行耗时
- `ar.prefill_ms` 表示 AR prompt prefill 耗时
- `ar.decode_ms` 表示 AR decode loop 耗时
- `ar.ttft_ms` 当前等价于 prefill 完成到首 token 可用的耗时
- `ar.kv_cache` 表示 AR backend 是否返回并使用 `past_key_values`
- `diffusion.load_ms` 表示首次加载 diffusion engine 的耗时
- `diffusion.elapsed_ms` 表示 diffusion stage 真正执行耗时
- Wan profiler 里的 `forward.total` 是 Wan 内部请求执行耗时

## 近期路线

### Step 1: AR runtime 细化

目标：把 request-level AR stage 拆成更接近 vLLM 的执行形态。

当前已完成：

- 明确 `ARPrefillOutput` / `ARDecodeOutput` / `ARTextOutput`
- 拆分 `prefill()` / `decode()` / `finalize()` 接口
- Transformers AR backend 使用显式 greedy decode loop
- decode 阶段复用 `past_key_values`
- trace 输出 `prefill_ms` / `decode_ms` / `ttft_ms` / `decode_steps` / `kv_cache`
- 保持 `Scheduler -> ModelRunner -> ARExecutor` 路径不变

当前边界：

- KV cache 仍由单请求 AR pipeline 内部持有
- Scheduler 尚不能感知 KV block、decode token、batchable decode step
- 还没有 streaming token 输出

### Step 2: AR 调度层 KV cache 与 streaming

目标：让 AR stage 从单请求内部 decode loop 走向可调度的 token runtime。

计划：

- 引入 AR request decode state
- 让 scheduler 能区分 prefill / decode request
- 将 KV cache 从 pipeline 内部状态提升到 executor/request state
- 支持单请求 streaming token 输出
- 再扩展到多请求 decode batching

### Step 3: AR 与 Diffusion 更好配合

目标：让 `qwen_to_wan_i2v` 不只是简单 prompt rewrite，而是更明确的跨 stage contract。

计划：

- 把 `ARToDiffusionConnector` 升级为更明确的 StageInputProcessor
- 记录 AR 输出 prompt、原 prompt、image、sampling params 的传递关系
- 支持可配置 prompt bridge 策略
- 为后续 AR 生成结构化 diffusion 参数预留接口

### Step 4: Stage 调度增强

目标：让 `StageScheduler` 从单请求顺序执行走向多请求管理。

计划：

- 多 session request state
- stage-level queue
- stage-level batching
- pipeline overlap

### Step 5: Diffusion step execution 与 cache

目标：在边界清晰后再做 diffusion 侧近似优化。

计划：

- 机械拆分 Wan prepare / denoise_step / step_scheduler / post_decode
- 验证 request-level 与 stepwise 输出一致性
- 再评估 DiTCache / TeaCache

## 参考方向

- vLLM-Omni: https://github.com/vllm-project/vllm-omni
- vLLM-Omni stage configs: https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/
- vLLM-Omni adding omni model: https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_omni_model/
- vLLM-Omni diffusion step execution: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_step_execution/

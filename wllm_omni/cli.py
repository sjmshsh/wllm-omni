import argparse
from pathlib import Path

import torch

from wllm_omni import DEFAULT_IMAGE, DEFAULT_MODEL, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, OmniLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Wan2.2 TI2V/I2V runner.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--preset", choices=["quality"], default="quality")
    parser.add_argument("--output", default="./download/example_wan22_i2v_quality.mp4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--profile", action="store_true", help="Print a per-request diffusion profiler summary.")
    parser.add_argument(
        "--vae-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="VAE load/decode dtype. fp32 is the stable default; bf16 can be profiled as an experimental speed policy.",
    )
    return parser.parse_args()


def _parse_vae_dtype(value: str) -> torch.dtype:
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported VAE dtype: {value}")


def main():
    args = parse_args()
    llm = OmniLLM(
        args.model,
        use_cpu_offload=not args.disable_cpu_offload,
        max_num_seqs=args.max_num_seqs,
        enable_profiling=args.profile,
        vae_dtype=_parse_vae_dtype(args.vae_dtype),
    )
    sampling_params = llm.preset(args.preset)
    if args.seed is not None:
        sampling_params.seed = args.seed
    generation = llm.generate(args.image, args.prompt, args.negative_prompt, sampling_params)
    output_path = Path(args.output)
    llm.save(output=generation, output_path=output_path)
    print(output_path.resolve())

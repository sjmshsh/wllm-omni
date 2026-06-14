import argparse
from pathlib import Path

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
    return parser.parse_args()


def main():
    args = parse_args()
    llm = OmniLLM(
        args.model,
        use_cpu_offload=not args.disable_cpu_offload,
        max_num_seqs=args.max_num_seqs,
        enable_profiling=args.profile,
    )
    sampling_params = llm.preset(args.preset)
    if args.seed is not None:
        sampling_params.seed = args.seed
    generation = llm.generate(args.image, args.prompt, args.negative_prompt, sampling_params)
    output_path = Path(args.output)
    llm.save(output=generation, output_path=output_path)
    print(output_path.resolve())

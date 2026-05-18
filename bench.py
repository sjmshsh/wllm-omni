import statistics
import time

from PIL import Image

from nanovllm_omni import DEFAULT_IMAGE, DEFAULT_PROMPT, OmniLLM


def main():
    llm = OmniLLM()
    sampling_params = llm.preset("quality")
    image = Image.open(DEFAULT_IMAGE).convert("RGB")

    # Warm up once after model load, following nano-vllm's bench style.
    warmup_output = llm.generate(image=image, prompt=DEFAULT_PROMPT, sampling_params=sampling_params)
    print(
        f"warmup: frames={len(warmup_output.frames)} size={warmup_output.width}x{warmup_output.height} "
        f"fps={warmup_output.fps}"
    )

    times = []
    for run_idx in range(5):
        start = time.perf_counter()
        output = llm.generate(image=image, prompt=DEFAULT_PROMPT, sampling_params=sampling_params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(
            f"run={run_idx + 1} time_s={elapsed:.4f} frames={len(output.frames)} "
            f"size={output.width}x{output.height}"
        )

    print("summary")
    print(f"mean_s={statistics.mean(times):.4f}")
    print(f"min_s={min(times):.4f}")
    print(f"max_s={max(times):.4f}")
    print("times_s=" + ",".join(f"{t:.4f}" for t in times))


if __name__ == "__main__":
    main()

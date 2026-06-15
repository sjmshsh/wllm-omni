from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from PIL import Image

from wllm_omni.cache import PromptCache, TensorCache
from wllm_omni.config import EngineConfig
from wllm_omni.outputs import OmniOutput
from wllm_omni.profiler import RequestProfiler
from wllm_omni.utils import resize_with_aspect
from wllm_omni.worker.utils import RunnerState


class Wan22I2VPipeline:
    """Step-wise Wan2.2 TI2V-5B image-to-video pipeline backed by diffusers."""

    @staticmethod
    def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # RunnerState is cached across steps under torch.inference_mode(); cloning avoids
        # reusing inference-mode tensors in later autograd-tracked forward passes.
        return tensor.clone()

    def __init__(self, config: EngineConfig):
        self.config = config
        self.prompt_cache = PromptCache(config.prompt_cache_size)
        self.image_cache = TensorCache(config.image_cache_size)
        vae = AutoencoderKLWan.from_pretrained(
            config.model,
            subfolder="vae",
            torch_dtype=config.vae_dtype,
            local_files_only=config.local_files_only,
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            config.model,
            vae=vae,
            torch_dtype=config.dtype,
            local_files_only=config.local_files_only,
        )
        if config.use_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(config.device)

    def _load_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def _image_cache_key(self, image: str | Path | Image.Image, height: int, width: int) -> tuple:
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser().resolve()
            stat = path.stat()
            image_key = ("path", str(path), stat.st_mtime_ns, stat.st_size)
        else:
            image_key = ("pil", id(image))
        return (image_key, height, width)

    def _align_resolution(self, height: int, width: int) -> tuple[int, int]:
        pipe = self.pipe
        patch_size = pipe.transformer.config.patch_size
        h_multiple_of = pipe.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = pipe.vae_scale_factor_spatial * patch_size[2]
        height = height // h_multiple_of * h_multiple_of
        width = width // w_multiple_of * w_multiple_of
        return height, width

    @staticmethod
    def _adjust_num_frames(num_frames: int, temporal_factor: int) -> int:
        if num_frames % temporal_factor != 1:
            num_frames = num_frames // temporal_factor * temporal_factor + 1
        return max(num_frames, 1)

    def _ensure_scheduler_on_device(self, device: torch.device) -> None:
        scheduler = self.pipe.scheduler
        for attr in ("sigmas", "timesteps", "last_sample"):
            value = getattr(scheduler, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(scheduler, attr, value.to(device))
        model_outputs = getattr(scheduler, "model_outputs", None)
        if isinstance(model_outputs, list):
            scheduler.model_outputs = [
                output.to(device) if isinstance(output, torch.Tensor) else output for output in model_outputs
            ]

    def _profile_sync(self) -> None:
        if not torch.cuda.is_available():
            return
        device = getattr(self.pipe, "_execution_device", None)
        if device is None:
            return
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _dtype_name(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    def runtime_info(self) -> dict[str, object]:
        pipe = self.pipe
        execution_device = getattr(pipe, "_execution_device", None)
        return {
            "cpu_offload": self.config.use_cpu_offload,
            "config_device": self.config.device,
            "execution_device": None if execution_device is None else str(execution_device),
            "config_dtype": self._dtype_name(self.config.dtype),
            "config_vae_dtype": self._dtype_name(self.config.vae_dtype),
            "transformer_dtype": self._dtype_name(getattr(pipe.transformer, "dtype", None)),
            "vae_dtype": self._dtype_name(getattr(pipe.vae, "dtype", None)),
            "scheduler": pipe.scheduler.__class__.__name__,
        }

    def profile_stage(self, state: RunnerState, name: str):
        profile = state.extra.get("profiler")
        if profile is None:
            return nullcontext()
        if not isinstance(profile, RequestProfiler):
            raise TypeError(f"Expected RequestProfiler payload, got {type(profile).__name__}.")
        return profile.stage(name, self._profile_sync)

    def prepare_encode(self, state: RunnerState) -> RunnerState:
        pipe = self.pipe
        sampling = state.sampling
        device = pipe._execution_device

        with self.profile_stage(state, "prepare.scheduler_config"):
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
                flow_shift=sampling.flow_shift,
            )

        with self.profile_stage(state, "prepare.image_load_resize"):
            height, width = self._align_resolution(sampling.height, sampling.width)
            num_frames = self._adjust_num_frames(sampling.num_frames, pipe.vae_scale_factor_temporal)
            image = resize_with_aspect(self._load_image(state.image), height, width)
        state.extra["height"] = height
        state.extra["width"] = width
        state.extra["num_frames"] = num_frames
        state.extra["guidance_scale"] = sampling.guidance_scale

        with self.profile_stage(state, "prepare.prompt_cache_lookup"):
            cache_key = (state.prompt, state.negative_prompt)
            cached = self.prompt_cache.get(cache_key)
        state.extra["prompt_cache_hit"] = cached is not None
        if cached is not None:
            with self.profile_stage(state, "prepare.prompt_cache_restore"):
                state.prompt_embeds, state.negative_prompt_embeds = cached
        else:
            with self.profile_stage(state, "prepare.prompt_encode"):
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=state.prompt,
                    negative_prompt=state.negative_prompt,
                    do_classifier_free_guidance=sampling.guidance_scale > 1.0,
                    num_videos_per_prompt=1,
                    max_sequence_length=512,
                    device=device,
                )
            with self.profile_stage(state, "prepare.prompt_to_dtype"):
                transformer_dtype = pipe.transformer.dtype
                state.prompt_embeds = prompt_embeds.to(transformer_dtype)
                state.negative_prompt_embeds = (
                    None if negative_prompt_embeds is None else negative_prompt_embeds.to(transformer_dtype)
                )
            with self.profile_stage(state, "prepare.prompt_cache_put"):
                self.prompt_cache.put(cache_key, (state.prompt_embeds, state.negative_prompt_embeds))

        with self.profile_stage(state, "prepare.image_cache_lookup"):
            image_cache_key = self._image_cache_key(state.image, height, width)
            cached_image_tensor = self.image_cache.get(image_cache_key)
        state.extra["image_cache_hit"] = cached_image_tensor is not None
        if cached_image_tensor is not None:
            with self.profile_stage(state, "prepare.image_cache_restore"):
                image_tensor = cached_image_tensor.to(device, dtype=torch.float32)
        else:
            with self.profile_stage(state, "prepare.image_preprocess"):
                image_tensor = pipe.video_processor.preprocess(image, height=height, width=width).to(
                    device, dtype=torch.float32
                )
            with self.profile_stage(state, "prepare.image_cache_put"):
                self.image_cache.put(image_cache_key, image_tensor)
        with self.profile_stage(state, "prepare.generator"):
            generator = torch.Generator(device=device).manual_seed(sampling.seed)

        with self.profile_stage(state, "prepare.latents"):
            latents_outputs = pipe.prepare_latents(
                image_tensor,
                batch_size=1,
                num_channels_latents=pipe.vae.config.z_dim,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )
        if pipe.config.expand_timesteps:
            latents, condition, first_frame_mask = latents_outputs
            state.extra["condition"] = condition
            state.extra["first_frame_mask"] = first_frame_mask
        else:
            latents, condition = latents_outputs
            state.extra["condition"] = condition
            state.extra["first_frame_mask"] = None

        with self.profile_stage(state, "prepare.clone_tensors"):
            state.latents = self._clone_tensor(latents)
            state.extra["condition"] = self._clone_tensor(state.extra["condition"])
            if state.extra["first_frame_mask"] is not None:
                state.extra["first_frame_mask"] = self._clone_tensor(state.extra["first_frame_mask"])
            state.prompt_embeds = self._clone_tensor(state.prompt_embeds)
            if state.negative_prompt_embeds is not None:
                state.negative_prompt_embeds = self._clone_tensor(state.negative_prompt_embeds)

        with self.profile_stage(state, "prepare.timesteps"):
            pipe.scheduler.set_timesteps(sampling.num_inference_steps, device=device)
            self._ensure_scheduler_on_device(device)
            state.timesteps = pipe.scheduler.timesteps
            pipe.scheduler.set_begin_index(0)
        state.step_index = 0
        return state

    def denoise_step(self, state: RunnerState) -> torch.Tensor:
        pipe = self.pipe
        with self.profile_stage(state, "denoise.prepare_inputs"):
            latents = state.latents
            device = latents.device
            t = state.timesteps[state.step_index].to(device)
            condition = state.extra["condition"].to(device)
            first_frame_mask = state.extra.get("first_frame_mask")
            if first_frame_mask is not None:
                first_frame_mask = first_frame_mask.to(device)
            guidance_scale = state.extra["guidance_scale"]
            transformer_dtype = pipe.transformer.dtype
            prompt_embeds = state.prompt_embeds.to(device)
            negative_prompt_embeds = (
                None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(device)
            )

            if pipe.config.expand_timesteps:
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

        with self.profile_stage(state, "denoise.transformer_cond"):
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

        if guidance_scale > 1.0:
            with self.profile_stage(state, "denoise.transformer_uncond"):
                with pipe.transformer.cache_context("uncond"):
                    noise_uncond = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]
            with self.profile_stage(state, "denoise.cfg_combine"):
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        return noise_pred

    def step_scheduler(self, state: RunnerState, noise_pred: torch.Tensor) -> None:
        with self.profile_stage(state, "scheduler.ensure_device"):
            device = state.latents.device
            self._ensure_scheduler_on_device(device)
            t = state.timesteps[state.step_index].to(device)
            noise_pred = noise_pred.to(device)
        with self.profile_stage(state, "scheduler.step_core"):
            latents = self.pipe.scheduler.step(noise_pred, t, state.latents, return_dict=False)[0]
        with self.profile_stage(state, "scheduler.clone_latents"):
            state.latents = self._clone_tensor(latents)
            state.step_index += 1

    def post_decode(self, state: RunnerState) -> OmniOutput:
        pipe = self.pipe
        with self.profile_stage(state, "post.latent_prepare"):
            latents = state.latents
            device = latents.device
            first_frame_mask = state.extra.get("first_frame_mask")
            condition = state.extra["condition"].to(device)

            if pipe.config.expand_timesteps:
                first_frame_mask = first_frame_mask.to(device)
                latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

            latents = latents.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
        with self.profile_stage(state, "post.vae_decode"):
            video = pipe.vae.decode(latents, return_dict=False)[0]
        with self.profile_stage(state, "post.video_postprocess"):
            frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]

        height = state.extra["height"]
        width = state.extra["width"]
        return OmniOutput(
            request_id=state.req_id,
            frames=frames,
            width=width,
            height=height,
            fps=state.sampling.fps,
            scheduler=self.pipe.scheduler.__class__.__name__,
        )

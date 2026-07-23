from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class ARTextOutput:
    request_id: str
    text: str
    tokens: list[str]
    token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARPrefillOutput:
    """Prompt-side AR state produced before decode."""

    prompt: str
    input_token_ids: list[int]
    input_tokens: list[str]
    next_token_id: int | None = None
    kv_cache: Any | None = None
    attention_mask: Any | None = None
    elapsed_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARDecodeOutput:
    token_ids: list[int]
    tokens: list[str]
    text: str
    elapsed_s: float = 0.0
    step_elapsed_s: list[float] = field(default_factory=list)
    stopped_by_eos: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return bool(self.metadata.get("finished", False))


class ARPipeline(ABC):
    """Text-generation stage interface for mini-Omni composition."""

    def generate(self, request: OmniRequest) -> ARTextOutput:
        prefill = self.prefill(request)
        decode = self.decode(request, prefill)
        return self.finalize(request, prefill, decode)

    @abstractmethod
    def prefill(self, request: OmniRequest) -> ARPrefillOutput:
        pass

    def init_decode(self) -> ARDecodeOutput:
        return ARDecodeOutput(token_ids=[], tokens=[], text="")

    def decode(self, request: OmniRequest, prefill: ARPrefillOutput) -> ARDecodeOutput:
        decode = self.init_decode()
        while not decode.finished:
            before_tokens = len(decode.token_ids)
            self.decode_step(request, prefill, decode)
            if len(decode.token_ids) == before_tokens and not decode.finished:
                decode.metadata["finished"] = True
                decode.stop_reason = "no_progress"
                decode.metadata["stop_reason"] = decode.stop_reason
        return decode

    @abstractmethod
    def decode_step(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARDecodeOutput:
        pass

    @abstractmethod
    def finalize(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARTextOutput:
        pass


class IdentityARPipeline(ARPipeline):
    """Deterministic AR placeholder used before a real causal LM backend."""

    def prefill(self, request: OmniRequest) -> ARPrefillOutput:
        start = perf_counter()
        text = self._normalize_prompt(request.prompt)
        tokens = self._tokenize(text)
        token_ids = [self._stable_token_id(token) for token in tokens]
        return ARPrefillOutput(
            prompt=text,
            input_token_ids=token_ids,
            input_tokens=tokens,
            elapsed_s=perf_counter() - start,
            metadata={"kv_cache_enabled": False},
        )

    def decode_step(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARDecodeOutput:
        decode.token_ids[:] = list(prefill.input_token_ids)
        decode.tokens[:] = list(prefill.input_tokens)
        decode.text = prefill.prompt
        decode.stop_reason = "identity"
        decode.metadata.update({
            "finished": True,
            "decode_model_steps": 0,
            "decode_scheduler_steps": 1,
            "stop_reason": decode.stop_reason,
        })
        return decode

    def finalize(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARTextOutput:
        return ARTextOutput(
            request_id=request.request_id,
            text=decode.text,
            tokens=decode.tokens,
            token_ids=decode.token_ids,
            metadata={
                "mode": "identity_prompt_bridge",
                "input_tokens": len(prefill.input_token_ids),
                "prefill_tokens": len(prefill.input_token_ids),
                "token_count": len(decode.token_ids),
                "generated_tokens": len(decode.token_ids),
                "prefill_elapsed_s": prefill.elapsed_s,
                "decode_elapsed_s": decode.elapsed_s,
                "ttft_s": prefill.elapsed_s,
                "kv_cache_enabled": False,
                "decode_model_steps": 0,
                "decode_model_calls": 0,
                "decode_scheduler_steps": decode.metadata.get("decode_scheduler_steps", 1),
                "stop_reason": decode.stop_reason,
            },
        )

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        text = " ".join(prompt.strip().split())
        return text or "high quality video"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _stable_token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)


class TransformersARPipeline(ARPipeline):
    """Minimal local Transformers CausalLM backend for the AR stage."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        dtype: Any = None,
        local_files_only: bool = True,
        max_new_tokens: int = 64,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=self.dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def prefill(self, request: OmniRequest) -> ARPrefillOutput:
        import torch

        prompt = self._build_prompt(request.prompt)
        inputs = self._tokenize_prompt(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            inputs["attention_mask"] = attention_mask

        self._sync_device()
        start = perf_counter()
        with torch.no_grad():
            model_output = self.model(
                **inputs,
                use_cache=True,
            )
        self._sync_device()
        elapsed_s = perf_counter() - start

        next_token_id = int(torch.argmax(model_output.logits[:, -1, :], dim=-1)[0].item())
        input_token_ids = [int(item) for item in input_ids[0].detach().cpu().tolist()]
        kv_cache = getattr(model_output, "past_key_values", None)
        return ARPrefillOutput(
            prompt=prompt,
            input_token_ids=input_token_ids,
            input_tokens=self.tokenizer.convert_ids_to_tokens(input_token_ids),
            next_token_id=next_token_id,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            elapsed_s=elapsed_s,
            metadata={
                "kv_cache_enabled": kv_cache is not None,
                "kv_cache_type": type(kv_cache).__name__ if kv_cache is not None else None,
                "prefill_tokens": len(input_token_ids),
            },
        )

    def decode_step(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARDecodeOutput:
        import torch

        decode.metadata["decode_scheduler_steps"] = int(decode.metadata.get("decode_scheduler_steps", 0)) + 1
        next_token_id = prefill.next_token_id
        if next_token_id is None:
            self._finish_decode(decode, "no_token")
            return decode
        if self._is_eos_token(next_token_id):
            decode.stopped_by_eos = True
            self._finish_decode(decode, "eos")
            return decode

        decode.token_ids.append(next_token_id)
        decode.tokens = self.tokenizer.convert_ids_to_tokens(decode.token_ids)
        decode.text = self.tokenizer.decode(decode.token_ids, skip_special_tokens=True).strip()

        if len(decode.token_ids) >= self.max_new_tokens:
            self._finish_decode(decode, "max_tokens")
            return decode

        input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
        attention_mask = prefill.attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, len(prefill.input_token_ids) + len(decode.token_ids) - 1),
                dtype=torch.long,
                device=self.device,
            )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=-1,
        )

        self._sync_device()
        step_start = perf_counter()
        with torch.no_grad():
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=prefill.kv_cache,
                use_cache=True,
            )
        self._sync_device()
        step_elapsed_s = perf_counter() - step_start
        decode.step_elapsed_s.append(step_elapsed_s)
        decode.elapsed_s += step_elapsed_s
        prefill.kv_cache = getattr(model_output, "past_key_values", None)
        prefill.attention_mask = attention_mask
        prefill.next_token_id = int(torch.argmax(model_output.logits[:, -1, :], dim=-1)[0].item())
        decode.metadata["decode_model_steps"] = len(decode.step_elapsed_s)
        decode.metadata["decode_tokens"] = len(decode.token_ids)

        if self._is_eos_token(prefill.next_token_id):
            decode.stopped_by_eos = True
            self._finish_decode(decode, "eos")
        return decode

    def finalize(
        self,
        request: OmniRequest,
        prefill: ARPrefillOutput,
        decode: ARDecodeOutput,
    ) -> ARTextOutput:
        text = decode.text
        if not text:
            text = request.prompt.strip()
        return ARTextOutput(
            request_id=request.request_id,
            text=text,
            tokens=decode.tokens,
            token_ids=decode.token_ids,
            metadata={
                "mode": "transformers_causal_lm",
                "model": self.model_path,
                "input_tokens": len(prefill.input_token_ids),
                "prefill_tokens": len(prefill.input_token_ids),
                "token_count": len(decode.token_ids),
                "generated_tokens": len(decode.token_ids),
                "prefill_elapsed_s": prefill.elapsed_s,
                "decode_elapsed_s": decode.elapsed_s,
                "decode_step_mean_ms": self._mean_ms(decode.step_elapsed_s),
                "decode_step_max_ms": self._max_ms(decode.step_elapsed_s),
                "decode_model_steps": decode.metadata.get("decode_model_steps", len(decode.step_elapsed_s)),
                "decode_model_calls": decode.metadata.get("decode_model_steps", len(decode.step_elapsed_s)),
                "decode_scheduler_steps": decode.metadata.get("decode_scheduler_steps"),
                "ttft_s": prefill.elapsed_s,
                "stopped_by_eos": decode.stopped_by_eos,
                "stop_reason": decode.stop_reason,
                "kv_cache_enabled": prefill.metadata.get("kv_cache_enabled", False),
                "kv_cache_type": prefill.metadata.get("kv_cache_type"),
            },
        )

    def _tokenize_prompt(self, prompt: str):
        messages = [
            {"role": "system", "content": "You rewrite user requests into concise visual prompts for image-to-video generation."},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    @staticmethod
    def _build_prompt(prompt: str) -> str:
        return (
            "Rewrite the following image-to-video request as a concise, visual video generation prompt. "
            "Keep the main subject, scene, motion, and style. Return only the rewritten prompt.\n\n"
            f"Request: {prompt.strip()}\nPrompt:"
        )

    def _finish_decode(self, decode: ARDecodeOutput, stop_reason: str) -> None:
        decode.stop_reason = stop_reason
        decode.metadata["finished"] = True
        decode.metadata["stop_reason"] = stop_reason
        decode.metadata["decode_model_steps"] = len(decode.step_elapsed_s)
        decode.metadata["decode_tokens"] = len(decode.token_ids)

    def _sync_device(self) -> None:
        if self.device.type != "cuda":
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _is_eos_token(self, token_id: int) -> bool:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, list):
            return token_id in eos_token_id
        return token_id == int(eos_token_id)

    @staticmethod
    def _mean_ms(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) * 1000.0 / len(values)

    @staticmethod
    def _max_ms(values: list[float]) -> float | None:
        if not values:
            return None
        return max(values) * 1000.0

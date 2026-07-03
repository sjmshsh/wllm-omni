from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
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


class ARPipeline(ABC):
    """Text-generation stage interface for mini-Omni composition."""

    @abstractmethod
    def generate(self, request: OmniRequest) -> ARTextOutput:
        pass


class IdentityARPipeline(ARPipeline):
    """Deterministic AR placeholder used before a real causal LM backend."""

    def generate(self, request: OmniRequest) -> ARTextOutput:
        text = self._normalize_prompt(request.prompt)
        tokens = self._tokenize(text)
        return ARTextOutput(
            request_id=request.request_id,
            text=text,
            tokens=tokens,
            token_ids=[self._stable_token_id(token) for token in tokens],
            metadata={
                "mode": "identity_prompt_bridge",
                "token_count": len(tokens),
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

    def generate(self, request: OmniRequest) -> ARTextOutput:
        import torch

        prompt = self._build_prompt(request.prompt)
        inputs = self._tokenize_prompt(prompt)
        input_length = int(inputs.input_ids.shape[-1])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0, input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = request.prompt.strip()
        token_ids = [int(item) for item in generated_ids.detach().cpu().tolist()]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return ARTextOutput(
            request_id=request.request_id,
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            metadata={
                "mode": "transformers_causal_lm",
                "model": self.model_path,
                "input_tokens": input_length,
                "token_count": len(token_ids),
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

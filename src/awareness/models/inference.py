"""Inference utilities for Awareness models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .awareness_decoder import AwarenessDecoder
from .encoder import ContextEncoder


class AwarenessInference:
    """Wrapper that loads trained weights into full-precision models for inference."""

    def __init__(
        self,
        encoder: ContextEncoder,
        decoder: AwarenessDecoder,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

        self.encoder.to(device=self.device, dtype=self.dtype)
        self.decoder.to(device=self.device, dtype=self.dtype)

    @classmethod
    def from_trained(
        cls,
        checkpoint_path: str,
        encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        decoder_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "AwarenessInference":
        """
        Load encoder/decoder weights from a checkpoint produced by AwarenessTrainer.
        """
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")

        encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_name, trust_remote_code=True
        )
        encoder_base = AutoModel.from_pretrained(
            encoder_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        encoder = ContextEncoder(
            model_name=encoder_name,
            base_model=encoder_base,
            tokenizer=encoder_tokenizer,
            torch_dtype=dtype,
        )
        encoder.load_state_dict(checkpoint["encoder"], strict=False)
        encoder.merge_lora_weights()

        decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_name, trust_remote_code=True
        )
        decoder_base = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        decoder = AwarenessDecoder(
            model_name=decoder_name,
            base_model=decoder_base,
            tokenizer=decoder_tokenizer,
            torch_dtype=dtype,
        )
        decoder.load_state_dict(checkpoint["decoder"], strict=False)

        return cls(encoder, decoder, device=device, dtype=dtype)

    def encode_context(self, context_documents: List[str]):
        """Encode context documents into padded memory tensors."""
        all_k, all_v, all_mask = [], [], []
        with torch.no_grad():
            for doc in context_documents:
                k, v, mask = self.encoder.encode_document(
                    doc, return_mask=True, use_grad=False
                )
                all_k.append(k)
                all_v.append(v)
                all_mask.append(mask)

        max_mem = max(t.size(1) for t in all_k) if all_k else 1
        batch_size = len(all_k) or 1

        memory_key = torch.zeros(
            batch_size,
            max_mem,
            self.encoder.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        memory_value = torch.zeros_like(memory_key)
        memory_mask = torch.zeros(batch_size, max_mem, device=self.device)

        for idx, (k, v, m) in enumerate(zip(all_k, all_v, all_mask)):
            seq_len = k.size(1)
            memory_key[idx, :seq_len] = k.squeeze(0)
            memory_value[idx, :seq_len] = v.squeeze(0)
            memory_mask[idx, :seq_len] = m.squeeze(0)

        return memory_key, memory_value, memory_mask

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        context_documents: List[str],
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> str:
        """Generate text conditioned on prompt + encoded memory."""
        tokenizer = self.decoder.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        memory_key, memory_value, memory_mask = self.encode_context(context_documents)

        output_ids = self.decoder.generate(
            input_ids=inputs.input_ids,
            memory_key=memory_key,
            memory_value=memory_value,
            memory_mask=memory_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
        """Encode context documents into memory tensors.

        All documents are concatenated along the sequence dimension into a
        single [1, total_tokens, hidden] tensor so the batch dimension matches
        a single-prompt forward pass.
        """
        all_k, all_v = [], []
        with torch.no_grad():
            for doc in context_documents:
                k, v, mask = self.encoder.encode_document(
                    doc, return_mask=True, use_grad=False
                )
                # Strip padding using the attention mask
                real_len = mask.sum().int().item()
                all_k.append(k.squeeze(0)[:real_len])
                all_v.append(v.squeeze(0)[:real_len])

        if not all_k:
            empty = torch.zeros(
                1, 1, self.encoder.hidden_size,
                device=self.device, dtype=self.dtype,
            )
            return empty, empty.clone(), torch.zeros(1, 1, device=self.device)

        # Concatenate all docs along the sequence dimension: [total_tokens, hidden]
        cat_k = torch.cat(all_k, dim=0)
        cat_v = torch.cat(all_v, dim=0)

        # Add batch dimension: [1, total_tokens, hidden]
        memory_key = cat_k.unsqueeze(0)
        memory_value = cat_v.unsqueeze(0)
        memory_mask = torch.ones(1, cat_k.size(0), device=self.device)

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

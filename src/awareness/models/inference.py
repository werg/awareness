"""Inference utilities for Awareness models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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

        # Validate hidden size compatibility for pipeline coarse attention
        enc_backbone = getattr(encoder, "backbone_hidden_size", None)
        dec_hidden = getattr(decoder, "hidden_size", None)
        if enc_backbone is not None and dec_hidden is not None and enc_backbone != dec_hidden:
            raise ValueError(
                f"Encoder backbone hidden size ({enc_backbone}) != decoder hidden size "
                f"({dec_hidden}). Pipeline coarse attention requires these to match."
            )

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
        pipeline_cfg = checkpoint.get("pipeline_config", {})

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
            **pipeline_cfg,
        )
        decoder.load_state_dict(checkpoint["decoder"], strict=False)

        return cls(encoder, decoder, device=device, dtype=dtype)

    def encode_context(
        self, context_documents: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Encode context documents into pipeline memory tensors.

        Returns:
            Dict with keys: doc_summary_key, doc_summary_value, doc_summary_mask,
            token_key, token_value, token_mask, doc_token_map.
            All tensors have batch dimension 1.
        """
        all_k: List[torch.Tensor] = []
        all_v: List[torch.Tensor] = []
        all_eos: List[torch.Tensor] = []
        doc_map_parts: List[torch.Tensor] = []

        with torch.no_grad():
            for doc_idx, doc in enumerate(context_documents):
                inputs = self.encoder.tokenizer(
                    doc,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.encoder.max_length,
                    padding=False,
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                k_mem, v_mem, eos_hidden = self.encoder(
                    input_ids, attention_mask, return_eos=True,
                )

                # Strip padding
                real_len = attention_mask.sum().int().item()
                all_k.append(k_mem.squeeze(0)[:real_len])
                all_v.append(v_mem.squeeze(0)[:real_len])
                all_eos.append(eos_hidden.squeeze(0))  # [hidden]
                doc_map_parts.append(
                    torch.full((real_len,), doc_idx, dtype=torch.long, device=self.device)
                )

        num_docs = len(context_documents)
        backbone_hidden = self.encoder.backbone_hidden_size
        hidden_size = self.encoder.hidden_size

        if num_docs == 0:
            return {
                "doc_summary_key": torch.zeros(1, 1, backbone_hidden, device=self.device, dtype=self.dtype),
                "doc_summary_value": torch.zeros(1, 1, backbone_hidden, device=self.device, dtype=self.dtype),
                "doc_summary_mask": torch.zeros(1, 1, device=self.device),
                "token_key": torch.zeros(1, 1, hidden_size, device=self.device, dtype=self.dtype),
                "token_value": torch.zeros(1, 1, hidden_size, device=self.device, dtype=self.dtype),
                "token_mask": torch.zeros(1, 1, device=self.device),
                "doc_token_map": torch.zeros(1, 1, dtype=torch.long, device=self.device),
            }

        # Doc summaries: [1, num_docs, backbone_hidden]
        doc_summary = torch.stack(all_eos, dim=0).unsqueeze(0)

        # Token-level KV: [1, total_tokens, hidden]
        cat_k = torch.cat(all_k, dim=0).unsqueeze(0)
        cat_v = torch.cat(all_v, dim=0).unsqueeze(0)
        cat_map = torch.cat(doc_map_parts, dim=0).unsqueeze(0)
        total_tokens = cat_k.size(1)

        return {
            "doc_summary_key": doc_summary,
            "doc_summary_value": doc_summary.clone(),
            "doc_summary_mask": torch.ones(1, num_docs, device=self.device),
            "token_key": cat_k,
            "token_value": cat_v,
            "token_mask": torch.ones(1, total_tokens, device=self.device),
            "doc_token_map": cat_map,
        }

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

        pipeline_memory = self.encode_context(context_documents)

        output_ids = self.decoder.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pipeline_memory=pipeline_memory,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

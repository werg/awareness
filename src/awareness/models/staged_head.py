"""StagedHead: one coarse-fine pair for pipelined hierarchical attention.

Coarse half: attends to per-document summary vectors (EOS embeddings),
injects summary V into residual stream, produces document selection scores.

Fine half: attends to token-level KV with log-multiplicative document bias
derived from coarse scores. Reuses GatedCrossAttention unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Float32RMSNorm, GatedCrossAttention


class StagedHead(nn.Module):
    """One coarse-fine head pair for pipelined staged attention.

    Owns all parameters for both coarse and fine stages.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float = 1e-6,
        attn_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # --- Coarse half ---
        self.coarse_norm = Float32RMSNorm(hidden_size, eps=rms_norm_eps)
        self.coarse_q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.coarse_k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.coarse_v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.coarse_o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.coarse_attn_dropout = nn.Dropout(attn_dropout)
        self.coarse_output_dropout = nn.Dropout(output_dropout)
        # Gate: sigmoid(-1) ~= 0.27, same init rationale as GCA
        self.coarse_gate = nn.Parameter(torch.tensor([-1.0]))

        # --- Fine half ---
        self.fine_norm = Float32RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fine_gca = GatedCrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_dropout=attn_dropout,
            output_dropout=output_dropout,
        )

    def _apply(self, fn):
        """Keep coarse_gate in float32 through dtype conversions."""
        super()._apply(fn)
        self.coarse_gate.data = self.coarse_gate.data.float()
        return self

    def coarse_forward(
        self,
        hidden_states: torch.Tensor,
        doc_summary_key: torch.Tensor,
        doc_summary_value: torch.Tensor,
        doc_summary_mask: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run coarse attention over document summaries.

        Args:
            hidden_states: Decoder hidden states [batch, seq_len, hidden].
            doc_summary_key: Per-doc EOS embeddings [batch, num_docs, hidden].
            doc_summary_value: Same as key [batch, num_docs, hidden].
            doc_summary_mask: 1=real doc, 0=padding [batch, num_docs].
            residual: Original (pre-norm) hidden states for residual connection.

        Returns:
            (updated_hidden_states, doc_scores):
            - updated_hidden_states: [batch, seq_len, hidden]
            - doc_scores: [batch, num_docs] attention probs averaged over heads+positions
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_docs = doc_summary_key.size(1)

        # Guard: no documents → passthrough with uniform scores
        if num_docs == 0:
            return residual, torch.zeros(
                batch_size, 0, device=hidden_states.device, dtype=hidden_states.dtype,
            )

        # Pre-norm
        normed = self.coarse_norm(hidden_states)

        # Project Q from decoder states
        q = self.coarse_q_proj(normed)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]

        # Project K, V from document summaries
        k = self.coarse_k_proj(doc_summary_key)
        k = k.view(batch_size, num_docs, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # [batch, num_kv_heads, num_docs, head_dim]

        v = self.coarse_v_proj(doc_summary_value)
        v = v.view(batch_size, num_docs, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention (manual path for score extraction)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, num_heads, seq_len, num_docs]

        # Apply document mask
        if doc_summary_mask is not None:
            # [batch, num_docs] -> [batch, 1, 1, num_docs]
            mask_bias = (1.0 - doc_summary_mask.unsqueeze(1).unsqueeze(2).to(attn_weights.dtype)) * -1e9
            attn_weights = attn_weights + mask_bias

        # Softmax in float32 for stability
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Document scores: average attention probs over heads and sequence positions
        # [batch, num_heads, seq_len, num_docs] -> [batch, num_docs]
        doc_scores = attn_probs.mean(dim=(1, 2))

        attn_probs = self.coarse_attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        # [batch, num_heads, seq_len, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.coarse_o_proj(attn_output)
        attn_output = self.coarse_output_dropout(attn_output)

        # Gated residual
        gate_value = torch.sigmoid(self.coarse_gate).to(dtype=attn_output.dtype)
        out = residual + gate_value * attn_output

        return out, doc_scores

    def fine_forward(
        self,
        hidden_states: torch.Tensor,
        token_key: torch.Tensor,
        token_value: torch.Tensor,
        token_mask: torch.Tensor,
        doc_scores: torch.Tensor,
        doc_token_map: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Run fine attention over token-level KV with document bias.

        Args:
            hidden_states: Decoder hidden states [batch, seq_len, hidden].
            token_key: Token-level keys [batch, total_tokens, hidden].
            token_value: Token-level values [batch, total_tokens, hidden].
            token_mask: 1=real token, 0=padding [batch, total_tokens].
            doc_scores: From coarse stage [batch, num_docs].
            doc_token_map: Maps each token to source doc index [batch, total_tokens] (long).
            residual: Original (pre-norm) hidden states for residual connection.

        Returns:
            Updated hidden states [batch, seq_len, hidden].
        """
        batch_size = hidden_states.size(0)
        total_tokens = token_key.size(1)

        # Build combined additive attention bias [batch, 1, 1, total_tokens]
        # 1) Document weighting: log(doc_scores[doc_of_token] + eps)
        eps = 1e-6
        log_doc_scores = torch.log(doc_scores + eps)  # [batch, num_docs]
        # Gather per-token document scores
        doc_bias = torch.gather(
            log_doc_scores, dim=1, index=doc_token_map
        )  # [batch, total_tokens]

        # 2) Padding mask: (1 - token_mask) * -1e9
        pad_bias = (1.0 - token_mask.to(doc_bias.dtype)) * -1e9

        # Combined bias
        combined_bias = (doc_bias + pad_bias).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, total_tokens]

        # Pre-norm then delegate to fine_gca
        normed = self.fine_norm(hidden_states)
        out = self.fine_gca(
            normed,
            token_key,
            token_value,
            memory_mask=combined_bias,
            residual=residual,
        )
        return out

"""Reasoning Kernel / Decoder (D_φ): Decoder-only LLM with Gated Cross-Attention.

From PLAN.md Section 2.3:
- Base Architecture: Standard Causal Self-Attention (CSA) handles immediate instruction
- Augmentation: In upper 1/3 of network, CSA blocks are interleaved with GCA blocks
- Mechanism: Attention(Q, K, V) = softmax(Q_loc @ K_mem^T / sqrt(d_k)) @ V_mem
  - Q_loc: Queries from Decoder's current prompt
  - K_mem, V_mem: Pre-computed tensors fetched from Memory Store
- Gradient Flow: During training, gradients propagate from D_φ through cross-attention
  into E_θ, forcing encoder to learn representations useful for decoder's reasoning
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention (GCA) block.

    Allows the decoder to attend to pre-computed memory tensors (K_mem, V_mem).
    The gating mechanism controls how much external knowledge to incorporate.

    This is interleaved with standard causal self-attention in the upper
    portion of the decoder network.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
    ):
        """
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of query attention heads
            num_kv_heads: Number of key/value heads (for GQA). Defaults to num_heads.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads

        # Query projection (from decoder hidden states)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # K/V projections (from encoder memory)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Learnable gate (initialized so projections get meaningful gradients)
        # Using sigmoid to bound in (0, 1) - represents fraction of memory to incorporate
        # Initialize to -1 so sigmoid(-1) ≈ 0.27 with gradient ≈ 0.20
        # This breaks the chicken-and-egg deadlock: projections receive ~27% of
        # full gradient (vs ~1.8% at -4), letting them learn useful patterns
        # which in turn gives the gate a consistent signal to grow.
        self.gate = nn.Parameter(torch.tensor([-1.0]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention to memory.

        Args:
            hidden_states: Decoder hidden states [batch, seq_len, hidden]
                (typically RMSNorm'd for query projection)
            memory_key: K_mem from encoder [batch, mem_len, hidden]
            memory_value: V_mem from encoder [batch, mem_len, hidden]
            memory_mask: Optional attention mask [batch, 1, 1, mem_len] or broadcastable
            residual: Original (pre-norm) hidden states for the residual connection.
                If None, hidden_states is used (backward-compatible).

        Returns:
            Updated hidden states with memory-aware representations (gated residual)
        """
        batch_size, seq_len, _ = hidden_states.shape
        mem_len = memory_key.size(1)

        # Project queries from decoder states
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [batch, num_heads, seq_len, head_dim]

        # Project K, V from encoder memory
        k = self.k_proj(memory_key)
        k = k.view(batch_size, mem_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k: [batch, num_kv_heads, mem_len, head_dim]

        v = self.v_proj(memory_value)
        v = v.view(batch_size, mem_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: [batch, num_kv_heads, mem_len, head_dim]

        # GQA: expand KV heads to match Q heads if needed
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        # attn_weights: [batch, num_heads, seq_len, mem_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply memory mask if provided (e.g., for padding)
        if memory_mask is not None:
            attn_weights = attn_weights + memory_mask

        # Softmax in float32 for numerical stability, then cast back to query dtype
        # (Standard HuggingFace pattern from Llama, Mistral, etc.)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Attend to values
        # attn_output: [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        # Gated residual connection
        # Gate uses sigmoid, bounded to (0, 1) representing fraction of memory to use
        # Residual adds to original (pre-norm) hidden states, not the normed version
        gate_value = torch.sigmoid(self.gate)
        base = residual if residual is not None else hidden_states
        return base + gate_value * attn_output


class ReasoningDecoder(nn.Module):
    """
    The Reasoning Kernel / Decoder (D_φ) - Abstract base class.

    A decoder-only LLM augmented with Gated Cross-Attention blocks
    in the upper 1/3 of the network.

    See AwarenessDecoder for the concrete Qwen3-based implementation.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int):
        """
        Args:
            hidden_size: Model hidden dimension
            num_layers: Total number of decoder layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # GCA applied to upper 1/3 of network (per PLAN.md)
        self.gca_start_layer = num_layers * 2 // 3

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional cross-attention to memory.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask for input [batch, seq_len]
            memory_key: K_mem from encoder [batch, mem_len, hidden]
            memory_value: V_mem from encoder [batch, mem_len, hidden]
            labels: Target token IDs for loss computation [batch, seq_len]

        Returns:
            Model outputs (implementation-specific)
        """
        raise NotImplementedError(
            "Decoder implementation depends on chosen backbone model. "
            "Use AwarenessDecoder for Qwen3-based implementation."
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        memory_key: Optional[torch.Tensor] = None,
        memory_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text with awareness of memory.

        During generation, each forward pass cross-attends to the memory store,
        giving the model "awareness" of the full repository context.
        """
        raise NotImplementedError

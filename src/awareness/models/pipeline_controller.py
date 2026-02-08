"""PipelineController: per-forward-pass orchestrator for staged heads.

NOT an nn.Module — holds no parameters. Created fresh each forward pass
by the decoder's _pipeline_context manager.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .pipeline_schedule import PipelineSchedule


class PipelineController:
    """Dispatches coarse/fine operations to staged heads based on the schedule.

    Created fresh each forward pass. Stores intermediate doc_scores
    produced by coarse stages for consumption by their paired fine stages.
    """

    def __init__(
        self,
        staged_heads: nn.ModuleList,
        schedule: PipelineSchedule,
        doc_summary_key: torch.Tensor,
        doc_summary_value: torch.Tensor,
        doc_summary_mask: Optional[torch.Tensor],
        token_key: torch.Tensor,
        token_value: torch.Tensor,
        token_mask: torch.Tensor,
        doc_token_map: torch.Tensor,
    ):
        self.staged_heads = staged_heads
        self.schedule = schedule
        self.doc_summary_key = doc_summary_key
        self.doc_summary_value = doc_summary_value
        self.doc_summary_mask = doc_summary_mask
        self.token_key = token_key
        self.token_value = token_value
        self.token_mask = token_mask
        self.doc_token_map = doc_token_map

        # Populated by coarse ops, consumed by fine ops
        self._doc_scores: Dict[int, torch.Tensor] = {}

    def process_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Process a decoder layer, running any scheduled coarse/fine ops.

        Args:
            layer_idx: The decoder layer index.
            hidden_states: Current hidden states (post-layer output).
            residual: Original hidden states for residual connection
                (typically same as hidden_states from the decoder layer output).

        Returns:
            Updated hidden states (unchanged if no ops scheduled at this layer).
        """
        if layer_idx not in self.schedule.layer_ops:
            return hidden_states

        op = self.schedule.layer_ops[layer_idx]

        if op.op_type == "coarse":
            hidden_states, doc_scores = self.staged_heads[op.coarse_head_idx].coarse_forward(
                hidden_states,
                self.doc_summary_key,
                self.doc_summary_value,
                self.doc_summary_mask,
                residual=residual,
            )
            self._doc_scores[op.coarse_head_idx] = doc_scores

        elif op.op_type == "fine":
            doc_scores = self._doc_scores[op.fine_head_idx]
            hidden_states = self.staged_heads[op.fine_head_idx].fine_forward(
                hidden_states,
                self.token_key,
                self.token_value,
                self.token_mask,
                doc_scores,
                self.doc_token_map,
                residual=residual,
            )

        elif op.op_type == "both":
            # Fine first (consumes prior coarse's scores), then coarse (starts next stage)
            # Fine uses the original residual
            doc_scores = self._doc_scores[op.fine_head_idx]
            hidden_states = self.staged_heads[op.fine_head_idx].fine_forward(
                hidden_states,
                self.token_key,
                self.token_value,
                self.token_mask,
                doc_scores,
                self.doc_token_map,
                residual=residual,
            )
            # Coarse uses fine's output as both input and residual
            hidden_states, new_doc_scores = self.staged_heads[op.coarse_head_idx].coarse_forward(
                hidden_states,
                self.doc_summary_key,
                self.doc_summary_value,
                self.doc_summary_mask,
                residual=hidden_states,
            )
            self._doc_scores[op.coarse_head_idx] = new_doc_scores

        return hidden_states

    def get_all_doc_scores(self) -> Dict[int, torch.Tensor]:
        """Return all accumulated document scores (head_idx -> scores tensor)."""
        return dict(self._doc_scores)

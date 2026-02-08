"""Pipeline schedule: maps decoder layer indices to coarse/fine operations.

Each staged head is a coarse-fine pair placed at layers (L, L+gap):
- Coarse (layer L): attends to per-document summary vectors
- Fine (layer L+gap): attends to token-level KV with document bias

On shared layers (where one head's fine meets the next head's coarse),
fine runs first, then coarse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class LayerOp:
    """Operation(s) to run at a given decoder layer."""

    op_type: Literal["coarse", "fine", "both"]
    fine_head_idx: Optional[int] = None
    coarse_head_idx: Optional[int] = None

    def __post_init__(self):
        if self.op_type == "coarse":
            if self.coarse_head_idx is None:
                raise ValueError("coarse op requires coarse_head_idx")
        elif self.op_type == "fine":
            if self.fine_head_idx is None:
                raise ValueError("fine op requires fine_head_idx")
        elif self.op_type == "both":
            if self.fine_head_idx is None or self.coarse_head_idx is None:
                raise ValueError("both op requires fine_head_idx and coarse_head_idx")


class PipelineSchedule:
    """Maps layer indices to coarse/fine operations for pipelined staged heads.

    Example layout (4 heads, gap=3, start=6, 28 layers):
        Layer  6: Coarse_0
        Layer  9: Fine_0 + Coarse_1
        Layer 12: Fine_1 + Coarse_2
        Layer 15: Fine_2 + Coarse_3
        Layer 18: Fine_3
    """

    def __init__(self, layer_ops: Dict[int, LayerOp]):
        self.layer_ops = layer_ops
        self.all_layer_indices = sorted(layer_ops.keys())

    @classmethod
    def build(
        cls,
        num_heads: int = 4,
        gap: int = 3,
        start_layer: int = 6,
        num_layers: int = 28,
    ) -> PipelineSchedule:
        """Build a pipeline schedule for the given configuration.

        Args:
            num_heads: Number of coarse-fine head pairs.
            gap: Layer spacing between coarse and its paired fine.
            start_layer: Layer index for the first coarse operation.
            num_layers: Total number of decoder layers (for validation).

        Returns:
            PipelineSchedule mapping layer indices to operations.

        Raises:
            ValueError: If schedule doesn't fit within num_layers.
        """
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if gap < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        if start_layer < 0:
            raise ValueError(f"start_layer must be >= 0, got {start_layer}")

        # Compute coarse and fine layer indices for each head
        coarse_layers = [start_layer + i * gap for i in range(num_heads)]
        fine_layers = [c + gap for c in coarse_layers]

        # Validate all layers fit
        max_layer = fine_layers[-1]
        if max_layer >= num_layers:
            raise ValueError(
                f"Pipeline schedule exceeds model layers: "
                f"last fine layer is {max_layer} but model has {num_layers} layers. "
                f"Reduce num_heads, gap, or start_layer."
            )

        # Build layer_ops dict, merging shared layers
        layer_ops: Dict[int, LayerOp] = {}

        for head_idx in range(num_heads):
            c_layer = coarse_layers[head_idx]
            f_layer = fine_layers[head_idx]

            # Register coarse
            if c_layer in layer_ops:
                # This layer already has a fine op from a previous head
                existing = layer_ops[c_layer]
                layer_ops[c_layer] = LayerOp(
                    op_type="both",
                    fine_head_idx=existing.fine_head_idx,
                    coarse_head_idx=head_idx,
                )
            else:
                layer_ops[c_layer] = LayerOp(
                    op_type="coarse",
                    coarse_head_idx=head_idx,
                )

            # Register fine
            if f_layer in layer_ops:
                # This layer already has a coarse op from a later head
                existing = layer_ops[f_layer]
                layer_ops[f_layer] = LayerOp(
                    op_type="both",
                    fine_head_idx=head_idx,
                    coarse_head_idx=existing.coarse_head_idx,
                )
            else:
                layer_ops[f_layer] = LayerOp(
                    op_type="fine",
                    fine_head_idx=head_idx,
                )

        return cls(layer_ops)

    @property
    def num_heads(self) -> int:
        """Number of staged head pairs in this schedule."""
        head_indices = set()
        for op in self.layer_ops.values():
            if op.coarse_head_idx is not None:
                head_indices.add(op.coarse_head_idx)
            if op.fine_head_idx is not None:
                head_indices.add(op.fine_head_idx)
        return len(head_indices)

    def __repr__(self) -> str:
        lines = []
        for layer_idx in self.all_layer_indices:
            op = self.layer_ops[layer_idx]
            lines.append(f"  Layer {layer_idx}: {op}")
        return f"PipelineSchedule(\n" + "\n".join(lines) + "\n)"

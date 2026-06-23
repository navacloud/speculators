"""Configuration for the depth-routed MoE EAGLE-3 speculator (eagle3_moe).

Extends the EAGLE-3 config with a small set of MoE fields. Routing is by rollout
depth (TTT step) and is deterministic, so there is no learned router and no
load-balancing loss. With ``num_experts == 1`` this is byte-for-byte EAGLE-3.
"""

from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.eagle3.config import Eagle3SpeculatorConfig

__all__ = ["Eagle3MoESpeculatorConfig"]


@SpeculatorModelConfig.register("eagle3_moe")
class Eagle3MoESpeculatorConfig(Eagle3SpeculatorConfig):
    """EAGLE-3 with per-rollout-depth expert MLPs.

    :param num_experts: Number of expert MLPs in each decoder layer's MoE bank.
        ``1`` reproduces vanilla EAGLE-3.
    :param depth_to_expert: Optional explicit map from TTT-step (rollout depth) to
        expert index. If ``None``, uses ``min(depth, num_experts - 1)`` so the last
        expert absorbs all deeper steps.
    :param expert_share_attention: Variant A (default): share attention + norms across
        depths, specialize only the MLP (keeps the cross-step K/V space consistent).
        ``False`` (fully independent layers) is not implemented yet.
    """

    speculators_model_type: Literal["eagle3_moe"] = "eagle3_moe"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle3MoEDraftModel"],
        description="Model architectures that can load these weights",
    )

    num_experts: int = Field(
        default=1,
        ge=1,
        description="Number of expert MLPs per decoder layer (1 = vanilla EAGLE-3)",
    )
    depth_to_expert: list[int] | None = Field(
        default=None,
        description="Explicit TTT-step -> expert-index map; None = min(depth, E-1)",
    )
    expert_share_attention: bool = Field(
        default=True,
        description="Variant A: share attention, specialize MLP only (only mode impl.)",
    )

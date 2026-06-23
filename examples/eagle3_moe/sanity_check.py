"""Sanity checks for the eagle3_moe scaffold (no GPU / no checkpoint required).

Run in an env with speculators installed (editable from this fork):
    pip install -e .
    python examples/eagle3_moe/sanity_check.py

Verifies:
  1. eagle3_moe is registered and resolvable by name.
  2. Each decoder layer's mlp is an ExpertMLP bank of size num_experts.
  3. Deterministic depth->expert routing.
  4. num_experts=1 is structurally equivalent to vanilla EAGLE-3 (expert.0 mirrors
     the baseline mlp; all non-MoE params identical in shape/count).
"""

from transformers import LlamaConfig

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models import Eagle3SpeculatorConfig  # noqa: F401  (ensures registration)
from speculators.models.eagle3_moe import (
    Eagle3MoEDraftModel,
    Eagle3MoESpeculatorConfig,
    ExpertMLP,
)


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=256,
        max_position_embeddings=128,
    )


def _moe_config(num_experts: int, depth_to_expert=None) -> Eagle3MoESpeculatorConfig:
    return Eagle3MoESpeculatorConfig(
        transformer_layer_config=_tiny_llama_config(),
        draft_vocab_size=128,
        num_experts=num_experts,
        depth_to_expert=depth_to_expert,
    )


def test_registered():
    assert SpeculatorModel.registry["eagle3_moe"] is Eagle3MoEDraftModel
    assert SpeculatorModelConfig.registry["eagle3_moe"] is Eagle3MoESpeculatorConfig
    print("[1/4] registration OK")


def test_expert_bank():
    model = Eagle3MoEDraftModel(_moe_config(num_experts=3))
    for layer in model.layers:
        assert isinstance(layer.mlp, ExpertMLP), "mlp not replaced by ExpertMLP"
        assert len(layer.mlp.experts) == 3
    print("[2/4] expert bank OK (3 experts/layer)")


def test_routing():
    m = Eagle3MoEDraftModel(_moe_config(num_experts=3))
    assert [m.route(d) for d in (0, 1, 2, 3, 7)] == [0, 1, 2, 2, 2]
    m2 = Eagle3MoEDraftModel(_moe_config(num_experts=3, depth_to_expert=[0, 0, 1]))
    assert [m2.route(d) for d in (0, 1, 2, 5)] == [0, 0, 1, 1]
    print("[3/4] routing OK")


def test_num_experts_1_parity():
    moe = Eagle3MoEDraftModel(_moe_config(num_experts=1))
    base = SpeculatorModel.registry["eagle3"](
        Eagle3SpeculatorConfig(
            transformer_layer_config=_tiny_llama_config(), draft_vocab_size=128
        )
    )
    # expert.0 mirrors the baseline mlp param shapes
    moe_mlp = dict(moe.layers[0].mlp.experts[0].named_parameters())
    base_mlp = dict(base.layers[0].mlp.named_parameters())
    assert moe_mlp.keys() == base_mlp.keys()
    assert all(moe_mlp[k].shape == base_mlp[k].shape for k in moe_mlp)
    # total trainable param count equal at num_experts=1
    n_moe = sum(p.numel() for p in moe.parameters())
    n_base = sum(p.numel() for p in base.parameters())
    assert n_moe == n_base, f"param mismatch: {n_moe} vs {n_base}"
    print("[4/4] num_experts=1 parity OK")


if __name__ == "__main__":
    test_registered()
    test_expert_bank()
    test_routing()
    test_num_experts_1_parity()
    print("\nAll eagle3_moe sanity checks passed.")

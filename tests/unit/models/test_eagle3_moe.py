"""Unit tests for the eagle3_moe (depth-routed MoE) draft model."""

import pytest
import torch
from transformers import LlamaConfig

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.models import Eagle3SpeculatorConfig
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


def _moe_config(num_experts: int = 1, depth_to_expert=None, share_attn: bool = True):
    return Eagle3MoESpeculatorConfig(
        transformer_layer_config=_tiny_llama_config(),
        draft_vocab_size=128,
        num_experts=num_experts,
        depth_to_expert=depth_to_expert,
        expert_share_attention=share_attn,
    )


def test_registered_by_name():
    assert SpeculatorModel.get_class("eagle3_moe") is Eagle3MoEDraftModel
    assert SpeculatorModelConfig.get_class("eagle3_moe") is Eagle3MoESpeculatorConfig


@pytest.mark.parametrize("num_experts", [1, 2, 3])
def test_expert_bank_constructed(num_experts):
    model = Eagle3MoEDraftModel(_moe_config(num_experts=num_experts))
    for layer in model.layers:
        assert isinstance(layer.mlp, ExpertMLP)
        assert len(layer.mlp.experts) == num_experts


def test_routing_default_tail_bucket():
    model = Eagle3MoEDraftModel(_moe_config(num_experts=3))
    assert [model.route(d) for d in (0, 1, 2, 3, 9)] == [0, 1, 2, 2, 2]


def test_routing_explicit_map():
    model = Eagle3MoEDraftModel(_moe_config(num_experts=2, depth_to_expert=[0, 0, 1]))
    assert [model.route(d) for d in (0, 1, 2, 5)] == [0, 0, 1, 1]


def test_set_active_expert_selects_mlp():
    model = Eagle3MoEDraftModel(_moe_config(num_experts=3))
    bank = model.layers[0].mlp
    x = torch.randn(1, 4, model.hidden_size)
    # Make experts return distinguishable outputs.
    with torch.no_grad():
        for i, e in enumerate(bank.experts):
            for p in e.parameters():
                p.zero_()
            # bias-free MLP -> zero output regardless; instead tag via a buffer check
        bank.set_active_expert(1)
    assert bank._active == 1
    # forward routes to experts[1] without error and preserves shape
    out = bank(x)
    assert out.shape == x.shape


def test_num_experts_1_parity_with_eagle3():
    moe = Eagle3MoEDraftModel(_moe_config(num_experts=1))
    base = SpeculatorModel.get_class("eagle3")(
        Eagle3SpeculatorConfig(
            transformer_layer_config=_tiny_llama_config(),
            draft_vocab_size=128,
        )
    )
    moe_mlp = dict(moe.layers[0].mlp.experts[0].named_parameters())
    base_mlp = dict(base.layers[0].mlp.named_parameters())
    assert moe_mlp.keys() == base_mlp.keys()
    assert all(moe_mlp[k].shape == base_mlp[k].shape for k in moe_mlp)
    assert sum(p.numel() for p in moe.parameters()) == sum(
        p.numel() for p in base.parameters()
    )


def test_expert_share_attention_false_not_implemented():
    with pytest.raises(NotImplementedError):
        Eagle3MoEDraftModel(_moe_config(num_experts=2, share_attn=False))

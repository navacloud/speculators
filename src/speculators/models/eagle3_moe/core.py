"""Depth-routed MoE EAGLE-3 draft model (eagle3_moe).

Subclasses :class:`Eagle3DraftModel` and replaces each decoder layer's ``mlp`` with
an :class:`ExpertMLP` bank. Exactly one expert is active per rollout depth (TTT step),
selected by a deterministic, parameter-free router. Everything else (fc fusion,
embedding, attention, norms, lm_head, draft-vocab mapping, losses) is inherited
unchanged.

NOTE: ``forward`` mirrors ``Eagle3DraftModel.forward`` so we can set the active expert
per TTT step. It is intentionally **not** ``torch.compile``-wrapped (the parent's is),
to keep the per-step expert selection straightforward; compile can be revisited later.
Keep this in sync with the parent's forward if upstream changes.
"""

import copy
from typing import ClassVar

import torch
from torch import nn
from transformers import DynamicCache, PretrainedConfig
from torch.nn.attention.flex_attention import create_block_mask

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.attention import (
    create_combined_mask_mod,
    extend_mask_for_draft_tokens,
)
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.eagle3.metrics import compute_metrics
from speculators.models.metrics import kl_div_loss, resolve_loss_fn
from speculators.models.utils import conditional_torch_compile, resolve_target_layer_ids
from speculators.proposals.greedy import GreedyTokenProposalConfig

from .config import Eagle3MoESpeculatorConfig

__all__ = ["Eagle3MoEDraftModel", "ExpertMLP"]


class ExpertMLP(nn.Module):
    """A bank of ``num_experts`` MLPs; one is active per call.

    Preserves the wrapped MLP's ``forward(x) -> x`` interface so the host decoder
    layer is unchanged. Experts are deep copies of the base MLP (identical at init;
    they diverge during training). The active expert is set by the model's forward
    loop via :meth:`set_active_expert` before each TTT step.
    """

    def __init__(self, base_mlp: nn.Module, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [copy.deepcopy(base_mlp) for _ in range(num_experts)]
        )
        self._active: int = 0

    def set_active_expert(self, idx: int) -> None:
        self._active = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts[self._active](x)


@SpeculatorModel.register("eagle3_moe")
class Eagle3MoEDraftModel(Eagle3DraftModel):
    config_class: ClassVar[type[Eagle3MoESpeculatorConfig]] = (  # type: ignore[misc,assignment]
        Eagle3MoESpeculatorConfig
    )

    def __init__(self, config: Eagle3MoESpeculatorConfig):
        if not config.expert_share_attention:
            raise NotImplementedError(
                "Only variant A (share attention, specialize MLP) is implemented; "
                "set expert_share_attention=True."
            )
        super().__init__(config)
        self.num_experts = config.num_experts
        self._depth_to_expert = config.depth_to_expert

        # Replace each layer's MLP with an expert bank (deep copies preserve the
        # already-initialized weights; do NOT call post_init again).
        for layer in self.layers:
            layer.mlp = ExpertMLP(layer.mlp, config.num_experts)

    def route(self, depth: int) -> int:
        """Deterministic depth -> expert index (no parameters)."""
        if self._depth_to_expert is not None:
            d = min(depth, len(self._depth_to_expert) - 1)
            return self._depth_to_expert[d]
        return min(depth, self.num_experts - 1)

    def _set_active_expert(self, expert_id: int) -> None:
        for layer in self.layers:
            layer.mlp.set_active_expert(expert_id)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, 3 * hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        ttt_steps: int = 3,
        ttt_step_loss_decay: float = 1.0,
        use_off_policy_tokens: bool = False,
        loss_fn=kl_div_loss,
        loss_only_depth: int | None = None,
        **kwargs,
    ):
        # Mirrors Eagle3DraftModel.forward, adding per-TTT-step expert routing.
        loss_fn = loss_fn or kl_div_loss
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]

        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        past_key_values = DynamicCache(config=self.config.transformer_layer_config)

        combined_mask_mod = create_combined_mask_mod(
            document_ids.squeeze(0).to(device), total_seq_len
        )
        attention_mask = create_block_mask(
            combined_mask_mod,
            B=None,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )

        if self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        hidden_states = self.fc(hidden_states)

        original_input_ids = input_ids.detach().clone()
        return_loss = verifier_last_hidden_states is not None
        if return_loss:
            with torch.no_grad():
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
            loss = torch.tensor(0.0, device=device)
            prev_correct = (
                loss_mask.clone()
                if loss_mask is not None
                else torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            )
            metrics = {}

        draft_tokens = []
        for ttt_step in range(ttt_steps):
            # ---- depth-routed expert selection (the only addition vs. base) ----
            self._set_active_expert(self.route(ttt_step))
            # ---------------------------------------------------------------------
            with torch.no_grad():
                input_embeds = self.embed_tokens(input_ids)
            cache_position = torch.arange(
                ttt_step * total_seq_len,
                (ttt_step + 1) * total_seq_len,
                dtype=torch.long,
                device=device,
            )

            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            logits = self.lm_head(self.norm(hidden_states))

            if return_loss:
                s_loss, s_metrics = compute_metrics(
                    logits,
                    targets,
                    loss_mask,
                    prev_correct,
                    ttt_step,
                    ttt_step_loss_decay,
                    loss_fn=loss_fn,
                )
                # Staged/curriculum training: accumulate loss ONLY from the target
                # depth. Because hidden_states carry differentiably across TTT steps,
                # a deeper depth's loss would otherwise backprop into (and corrupt)
                # the trained expert. None => joint (all depths), unchanged behavior.
                if loss_only_depth is None or ttt_step == loss_only_depth:
                    loss += s_loss
                metrics.update(s_metrics)

            input_ids = torch.argmax(logits, dim=-1)
            draft_tokens.append(input_ids.detach().clone())
            if self.d2t is not None:
                input_ids = input_ids + self.d2t[input_ids]  # type: ignore[index]

            if use_off_policy_tokens:
                input_ids = torch.cat(
                    [
                        original_input_ids[:, 1 + ttt_step :],
                        original_input_ids.new_zeros(1, 1 + ttt_step),
                    ],
                    dim=-1,
                )

            attention_mask = extend_mask_for_draft_tokens(attention_mask)
            position_ids = position_ids + 1

        if return_loss:
            metrics["loss_sum"] = loss.detach().clone()
            metrics["loss_total"] = torch.tensor(1.0, device=device)
            return draft_tokens, loss, metrics
        return draft_tokens

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "Eagle3MoEDraftModel":
        """Build from CLI args (mirrors Eagle3DraftModel.from_training_args + MoE)."""
        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"), kwargs["verifier_name_or_path"]
        )

        config = Eagle3MoESpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            norm_before_fc=kwargs.get("norm_before_fc", False),
            embed_requires_grad=kwargs.get("embed_requires_grad", False),
            eagle_aux_hidden_state_layer_ids=target_layer_ids,
            num_experts=kwargs.get("num_experts", 1),
            depth_to_expert=kwargs.get("depth_to_expert"),
            expert_share_attention=kwargs.get("expert_share_attention", True),
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3_moe",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

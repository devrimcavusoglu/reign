"""Tests for the chunk-position signal behind the App. E ablation.

REIGN's published design treats the chunk sequence as a permutation-equivariant
set and adds no positional information. That choice is justified empirically in
App. E rather than by assertion, which needs working `absolute` and `sinusoidal`
arms to compare against.

Two properties matter and are easy to get wrong:

* the default must stay inert, so every existing checkpoint keeps its published
  behaviour and legacy configs carrying leftovers of the removed positional
  surface still load;
* the two live arms must actually break permutation invariance -- an ablation
  between arms that all behave identically would silently prove nothing.
"""

from __future__ import annotations

import pytest
import torch

from reign.configuration import ReignConfig
from reign.modeling import ReignModel


def _model(position_embedding_type=None, max_position_embeddings=None, seed=0):
    torch.manual_seed(seed)
    config = ReignConfig(
        vocab_size=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        gn_projection_dim=32,
        position_embedding_type=position_embedding_type,
        max_position_embeddings=max_position_embeddings,
    )
    return ReignModel(config).eval()


def _pooled(model, embeds):
    mask = torch.ones(embeds.shape[:2], dtype=torch.long)
    with torch.no_grad():
        return model(inputs_embeds=embeds, attention_mask=mask).pooler_output


def test_default_is_inert():
    """A default config carries no position signal and stays order-blind."""
    model = _model()
    assert model.embeddings.position_embedding_type is None


@pytest.mark.parametrize("legacy", [None, "none", "relative_key", "relative_key_query"])
def test_legacy_and_none_values_stay_inert(legacy):
    """Leftovers of the removed positional surface must load without activating."""
    model = _model(position_embedding_type=legacy)
    assert model.embeddings.position_embedding_type is None


def test_none_arm_is_permutation_invariant():
    """The published design: shuffling chunks must not move the document vector."""
    model = _model()
    embeds = torch.randn(2, 6, 32)
    perm = torch.randperm(6)
    assert torch.allclose(_pooled(model, embeds), _pooled(model, embeds[:, perm]), atol=1e-5)


@pytest.mark.parametrize("arm", ["absolute", "sinusoidal"])
def test_live_arms_break_permutation_invariance(arm):
    """If an arm were order-blind too, the ablation would compare nothing."""
    model = _model(arm, max_position_embeddings=16)
    embeds = torch.randn(2, 6, 32)
    perm = torch.randperm(6)
    assert not torch.allclose(_pooled(model, embeds), _pooled(model, embeds[:, perm]), atol=1e-5)


@pytest.mark.parametrize("arm", ["absolute", "sinusoidal"])
def test_sequences_longer_than_the_table_clamp_rather_than_crash(arm):
    """DAPFAM documents run to hundreds of chunks; exceeding the table must not raise."""
    model = _model(arm, max_position_embeddings=8)
    out = _pooled(model, torch.randn(1, 40, 32))
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


def test_absolute_adds_a_table_and_sinusoidal_adds_no_parameters():
    base = sum(p.numel() for p in _model().parameters())
    absolute = sum(p.numel() for p in _model("absolute", max_position_embeddings=8).parameters())
    sinusoidal = sum(p.numel() for p in _model("sinusoidal", max_position_embeddings=8).parameters())
    assert absolute - base == 8 * 64  # max_position_embeddings x hidden_size
    assert sinusoidal == base  # fixed table is a buffer, not a parameter


@pytest.mark.parametrize("arm", ["absolute", "sinusoidal"])
def test_setting_survives_save_and_load(arm, tmp_path):
    """Eval reads the arm off the checkpoint, so it has to round-trip."""
    model = _model(arm, max_position_embeddings=8)
    model.save_pretrained(tmp_path)
    reloaded = ReignModel.from_pretrained(tmp_path).eval()
    assert reloaded.config.position_embedding_type == arm
    assert reloaded.embeddings.position_embedding_type == arm
    embeds = torch.randn(1, 5, 32)
    assert torch.allclose(_pooled(model, embeds), _pooled(reloaded, embeds), atol=1e-6)

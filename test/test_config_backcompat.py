"""Backwards-compat tests for ``ReignConfig`` after the dead positional-embedding
surface was removed.

Older REIGN configs persisted to disk may still carry ``max_position_embeddings``
and ``position_embedding_type`` keys. ``transformers.PretrainedConfig`` absorbs
any unknown kwargs as attributes, so they should load without raising. These
tests pin that behaviour so the removal is non-breaking for existing checkpoints.

Includes a small smoke test that instantiates ``ReignModel`` and runs a forward
pass over fake chunk embeddings to confirm the cleanup didn't regress the
forward path.
"""

from __future__ import annotations

import pytest
import torch

from reign.configuration import ReignConfig, ReignTinyL1Config
from reign.modeling import ReignModel


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"max_position_embeddings": 128, "position_embedding_type": None},
        {"max_position_embeddings": 512, "position_embedding_type": "absolute"},
        {"max_position_embeddings": 64, "position_embedding_type": "relative_key"},
    ],
)
def test_reign_config_accepts_legacy_position_kwargs(legacy_kwargs):
    """Legacy configs that include the removed kwargs still construct cleanly."""
    config = ReignConfig(
        vocab_size=3,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        gn_projection_dim=8,
        **legacy_kwargs,
    )
    # PretrainedConfig absorbs unknown kwargs as attributes, so legacy fields
    # remain readable (but inert).
    for key, value in legacy_kwargs.items():
        assert getattr(config, key) == value


def test_reign_config_has_no_position_defaults():
    """The defaults of the cleaned-up config no longer carry the position knobs."""
    config = ReignConfig()
    assert (
        not hasattr(config, "max_position_embeddings")
        or getattr(config, "max_position_embeddings", None) is None
    )
    assert (
        not hasattr(config, "position_embedding_type")
        or getattr(config, "position_embedding_type", None) is None
    )


def test_reign_model_forward_smoke():
    """Instantiate a tiny REIGN model and verify a forward pass over fake chunk
    embeddings runs end-to-end without touching the removed position surface.
    """
    config = ReignTinyL1Config(
        gn_projection_dim=8, hidden_size=8, num_attention_heads=2, intermediate_size=16
    )
    model = ReignModel(config)
    model.eval()

    batch_size, seq_len = 2, 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.gn_projection_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.pooler_output.shape == (batch_size, config.hidden_size)


def test_reign_model_forward_smoke_long_chunk_sequence():
    """Without ``max_position_embeddings`` REIGN should accept arbitrarily long
    chunk sequences. Run a forward pass at a length that exceeds the old default
    cap (128) to prove the limit is gone.
    """
    config = ReignTinyL1Config(
        gn_projection_dim=8, hidden_size=8, num_attention_heads=2, intermediate_size=16
    )
    model = ReignModel(config)
    model.eval()

    batch_size, seq_len = 1, 256  # 2x the old default of 128
    inputs_embeds = torch.randn(batch_size, seq_len, config.gn_projection_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

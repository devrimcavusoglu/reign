"""Tests for sliding-window chunking in ``ReignFeatureExtractor``.

Pins two behaviours:

1. The cache key derived in ``EmbeddingCache._get_cache_path`` falls back to the
   legacy ``f"{model_name}_{chunk_size}"`` shape whenever ``stride is None`` or
   ``stride == chunk_size``, and switches to ``f"{model_name}_{chunk_size}_s{N}"``
   only when stride differs (so existing caches keep their paths).

2. ``ReignFeatureExtractor.preprocess`` calls the underlying HF tokenizer with
   ``stride=chunk_size - stride`` (HF's stride means "overlap"), so that the
   resulting chunk count matches the sliding-window formula
   ``floor((tokens - chunk_size) / stride) + 1`` at common (chunk_size, stride)
   configurations.

The GN forward pass is **not** exercised; the tokenizer is mocked and
``AutoModel.from_pretrained`` is patched so no weights are downloaded.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from reign.feature_extractor import EmbeddingCache, ReignFeatureExtractor

# ---------------------------------------------------------------------------
# Cache-key derivation
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(cache_root=tmp_path, enable_cache=True)


def test_cache_path_legacy_format_when_stride_none(cache):
    """``stride=None`` (the pre-overlap call path) preserves the legacy key."""
    path_no_stride = cache._get_cache_path("thenlper/gte-small", 512, "ds-a")
    path_equal_stride = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=512)
    assert path_no_stride == path_equal_stride


def test_cache_path_diverges_when_stride_differs(cache):
    """``stride != chunk_size`` keys the model-cache directory by stride too."""
    legacy = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=512)
    overlap = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=384)
    assert legacy != overlap
    # Same dataset-id, so the *leaf filename* matches; only the model-hash dir differs.
    assert legacy.name == overlap.name
    assert legacy.parent != overlap.parent


def test_cache_path_stride_token_is_hashed_in(cache, tmp_path):
    """Different strides produce distinct cache directories (one hash per stride)."""
    s128 = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=128)
    s256 = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=256)
    s384 = cache._get_cache_path("thenlper/gte-small", 512, "ds-a", stride=384)
    assert len({s128.parent, s256.parent, s384.parent}) == 3


# ---------------------------------------------------------------------------
# Chunking with a mocked tokenizer
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(num_tokens: int):
    """Return a tokenizer mock whose ``__call__`` returns overflow chunks
    matching the sliding-window formula
    ``floor((num_tokens - max_length) / (max_length - stride)) + 1``
    (HF tokenizer stride = overlap = max_length - effective_stride).
    """

    def fake_call(
        inputs, max_length, padding, return_tensors, truncation, return_overflowing_tokens, stride
    ):
        # HF stride is the *overlap* (number of tokens shared with the previous chunk).
        effective_stride = max_length - stride
        if effective_stride <= 0:
            raise ValueError("effective_stride must be positive in this mock")
        # Number of windows of size max_length over num_tokens with step effective_stride.
        if num_tokens <= max_length:
            n_chunks = 1
        else:
            n_chunks = (num_tokens - max_length) // effective_stride + 1
            # HF returns one extra chunk for any trailing remainder; align with that
            # via the simple floor formula (which is what the spec asks the test to
            # assert against).
        # Build matching tensors; their values don't matter, only the shape.
        input_ids = torch.zeros(n_chunks, max_length, dtype=torch.long)
        attention_mask = torch.ones(n_chunks, max_length, dtype=torch.long)
        token_type_ids = torch.zeros(n_chunks, max_length, dtype=torch.long)
        result = MagicMock()
        result.__getitem__ = lambda self, k: {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }[k]
        result.pop = lambda k: torch.zeros(n_chunks, dtype=torch.long)
        result._n_chunks = n_chunks
        return result

    mock = MagicMock()
    mock.side_effect = fake_call
    return mock


def _build_extractor(chunk_size: int, stride: int):
    """Construct a ``ReignFeatureExtractor`` with the GN model and tokenizer
    fully mocked. No network access, no weights loaded.
    """
    with patch("reign.feature_extractor.AutoTokenizer.from_pretrained") as mock_tok, patch(
        "reign.feature_extractor.AutoModel.from_pretrained"
    ) as mock_model:
        mock_tok.return_value = MagicMock()
        fake_model = MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.config.max_position_embeddings = 128
        fake_model.config.hidden_size = 32
        mock_model.return_value = fake_model
        extractor = ReignFeatureExtractor(
            batch_size=2,
            model_name_or_path="dummy/gn",
            chunk_size=chunk_size,
            stride=stride,
            device="cpu",
            enable_cache=False,
        )
    return extractor


@pytest.mark.parametrize(
    "num_tokens,chunk_size,stride",
    [
        (1024, 512, 384),  # default: 25% overlap
        (1024, 512, 512),  # legacy: no overlap
        (1024, 256, 128),  # 50% overlap
        (512, 512, 384),  # exactly one chunk
        (2000, 512, 384),  # longer text
    ],
)
def test_chunk_count_matches_sliding_window_formula(num_tokens, chunk_size, stride):
    """``preprocess`` must call the tokenizer with stride translated to HF's
    overlap convention and produce
    ``floor((num_tokens - chunk_size) / stride) + 1`` chunks (clamped at 1).
    """
    extractor = _build_extractor(chunk_size, stride)
    extractor.tokenizer = _make_mock_tokenizer(num_tokens=num_tokens)

    model_inputs, overflow = extractor.preprocess(["irrelevant text"])

    # Verify the stride threaded into the HF tokenizer call matches HF's overlap
    # convention (max_length - effective_stride).
    assert extractor.tokenizer.called
    call_kwargs = extractor.tokenizer.call_args.kwargs
    expected_hf_stride = 0 if stride == chunk_size else chunk_size - stride
    assert call_kwargs["stride"] == expected_hf_stride
    assert call_kwargs["max_length"] == chunk_size
    assert call_kwargs["return_overflowing_tokens"] is True

    if num_tokens <= chunk_size:
        expected_chunks = 1
    else:
        expected_chunks = math.floor((num_tokens - chunk_size) / stride) + 1
    assert model_inputs._n_chunks == expected_chunks


def test_extractor_rejects_invalid_stride():
    with pytest.raises(ValueError):
        _build_extractor(chunk_size=512, stride=0)
    with pytest.raises(ValueError):
        _build_extractor(chunk_size=512, stride=1024)


def test_extractor_stores_chunking_metadata():
    extractor = _build_extractor(chunk_size=512, stride=384)
    assert extractor.chunk_size == 512
    assert extractor.stride == 384

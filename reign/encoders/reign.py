"""REIGN as a ``BaseEncoder``: load a trained REIGN checkpoint, run the
guidance network to produce chunk embeddings, then aggregate via the REIGN
encoder to a single document vector. Plugs into the same eval harness the
dense baselines use, so REIGN-on-X and X-alone rows in paper Tables 3-4 are
produced symmetrically.

Usage (programmatic):

    from reign.encoders.reign import ReignBaselineEncoder

    encoder = ReignBaselineEncoder(
        checkpoint_path="path/to/reign-base-on-gte-large",
        gn_model="thenlper/gte-large",
        chunk_size=512,
    )
    embeddings = encoder.encode(["doc 1 text", "doc 2 text"], batch_size=8)
    # → np.ndarray of shape (2, hidden_size), L2-normalised

The runner is `scripts/evaluate_reign.py`.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

import numpy as np

logger = logging.getLogger(__name__)


class ReignBaselineEncoder:
    """Trained REIGN model behind a ``BaseEncoder``-compatible surface.

    Combines:
      - a Guidance Network (GN) loaded by ``ReignFeatureExtractor`` that turns raw
        text into a chunk-level embedding sequence, and
      - a ``ReignModel`` checkpoint that aggregates that sequence into a single
        document embedding via its pooler head.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gn_model: str,
        chunk_size: int = 512,
        stride: int = 384,
        device: str | None = None,
        gn_batch_size: int = 12,
        normalize: bool = True,
        name: str | None = None,
    ):
        import torch

        from reign import ReignModel
        from reign.feature_extractor import ReignFeatureExtractor

        self.checkpoint_path = checkpoint_path
        self.gn_model = gn_model
        self.chunk_size = chunk_size
        self.stride = stride
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        gn_short = gn_model.split("/")[-1]
        self.name = name or f"reign-on-{gn_short}"

        logger.info("Loading REIGN checkpoint %s", checkpoint_path)
        self.model = ReignModel.from_pretrained(checkpoint_path).to(self.device).eval()

        logger.info(
            "Initialising guidance network %s (chunk_size=%d, stride=%d)",
            gn_model,
            chunk_size,
            stride,
        )
        self.feature_extractor = ReignFeatureExtractor(
            batch_size=gn_batch_size,
            model_name_or_path=gn_model,
            chunk_size=chunk_size,
            stride=stride,
            device=self.device,
            enable_cache=False,
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def hidden_size(self) -> int:
        return getattr(self.model.config, "hidden_size", -1)

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        side: Literal["query", "document"] = "document",  # REIGN is symmetric
    ) -> "np.ndarray":
        import torch

        texts = list(texts)
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        bs = batch_size or 8
        out: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            features = self.feature_extractor(batch, use_cache=False)
            features = {k: v.to(self.device) for k, v in features.items() if hasattr(v, "to")}
            with torch.no_grad():
                result = self.model(**features)
            pooled = getattr(result, "pooler_output", None)
            if pooled is None:
                # Fallback for tuple-style returns (e.g. mean-pooled last_hidden_state).
                pooled = result[1] if isinstance(result, tuple) and len(result) > 1 else result[0]
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)


__all__ = ["ReignBaselineEncoder"]

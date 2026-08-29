"""
Dense long-context retrieval baselines.

Wraps any HuggingFace transformer-based encoder behind the `BaseEncoder`
protocol. Targets the native long-context dense retrievers used as
baselines: BGE-M3, Jina-Embeddings-v3, Nomic-Embed-Text-v1.5,
E5-Mistral-7B, and Stella-en-1.5B.

Two inference protocols are supported per baseline:
  - "truncate":      tokenize with truncation to native context window
  - "chunk_pool":    split into K-token chunks, encode each, mean-pool

The "chunk_pool" path matches the protocol REIGN compares against in
Tables 3-4 ("Chunked inference (512 tokens)") so the comparison stays
apples-to-apples across the accuracy column.
"""

from __future__ import annotations

import time

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

logger = logging.getLogger(__name__)


PoolingMode = Literal["mean", "cls", "last_token"]
InferenceProtocol = Literal["truncate", "chunk_pool"]


@dataclass
class DenseEncoderConfig:
    """Hyperparameters for a dense baseline run.

    The defaults match a "neutral" sentence-transformer-style encoder; per-model
    overrides (CLS pooling for BGE, last-token for Mistral-style, instruction
    prefixes for Nomic) are wired in `BASELINES` below.
    """

    model_name_or_path: str
    pooling: PoolingMode = "mean"
    max_length: int = 512
    inference_protocol: InferenceProtocol = "truncate"
    chunk_size: int = 512
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "float32"  # "float16" / "bfloat16" for big models

    @property
    def display_name(self) -> str:
        return self.model_name_or_path.split("/")[-1]


# Curated dense baseline registry. The four native long-context models
# (BGE-M3, Jina-v3, Nomic-v1.5, Stella-1.5B) are the ones reported in the
# paper's main tables; E5-Mistral is the large-scale LoCo reference point.
BASELINES: dict[str, DenseEncoderConfig] = {
    "bge-m3": DenseEncoderConfig(
        model_name_or_path="BAAI/bge-m3",
        pooling="cls",
        max_length=8192,
        inference_protocol="truncate",
    ),
    "jina-v3": DenseEncoderConfig(
        model_name_or_path="jinaai/jina-embeddings-v3",
        pooling="mean",
        max_length=8192,
        inference_protocol="truncate",
        trust_remote_code=True,
        torch_dtype="bfloat16",  # 572M params @ 8K ctx OOMs in fp32 on a 24 GB card
    ),
    "nomic-v1.5": DenseEncoderConfig(
        model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
        pooling="mean",
        max_length=8192,
        inference_protocol="truncate",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        trust_remote_code=True,
    ),
    "e5-mistral-7b": DenseEncoderConfig(
        model_name_or_path="intfloat/e5-mistral-7b-instruct",
        pooling="last_token",
        max_length=4096,
        inference_protocol="truncate",
        torch_dtype="bfloat16",
    ),
    "stella-1.5b": DenseEncoderConfig(
        model_name_or_path="dunzhang/stella_en_1.5B_v5",
        pooling="mean",
        max_length=8192,
        inference_protocol="truncate",
        trust_remote_code=True,
        torch_dtype="bfloat16",
    ),
    # Short-context (512 tok) base models that REIGN has been / will be applied
    # on top of. These are the "before" side of the REIGN-lift comparison: each
    # row stands alongside a REIGN-on-X row for the same model. chunk_pool
    # protocol matches the "Chunked inference (512 tokens)" baseline reported in
    # paper Tables 3-4.
    "bge-large-chunked": DenseEncoderConfig(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        pooling="cls",
        max_length=512,
        inference_protocol="chunk_pool",
        chunk_size=512,
    ),
    "bge-base-chunked": DenseEncoderConfig(
        model_name_or_path="BAAI/bge-base-en-v1.5",
        pooling="cls",
        max_length=512,
        inference_protocol="chunk_pool",
        chunk_size=512,
    ),
    "gte-large-chunked": DenseEncoderConfig(
        model_name_or_path="thenlper/gte-large",
        pooling="mean",
        max_length=512,
        inference_protocol="chunk_pool",
        chunk_size=512,
    ),
    "gte-base-chunked": DenseEncoderConfig(
        model_name_or_path="thenlper/gte-base",
        pooling="mean",
        max_length=512,
        inference_protocol="chunk_pool",
        chunk_size=512,
    ),
    "gte-small-chunked": DenseEncoderConfig(
        model_name_or_path="thenlper/gte-small",
        pooling="mean",
        max_length=512,
        inference_protocol="chunk_pool",
        chunk_size=512,
    ),
}


def _pool(last_hidden: "np.ndarray", attention_mask: "np.ndarray", mode: PoolingMode):
    """Pool a (batch, seq, hidden) tensor according to `mode`."""
    import torch

    if mode == "cls":
        return last_hidden[:, 0]
    if mode == "last_token":
        # find last non-pad position per row
        seq_lens = attention_mask.sum(dim=1) - 1
        idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden.size(-1))
        return last_hidden.gather(1, idx).squeeze(1)
    # mean
    masked = last_hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)


class DenseEncoder:
    """HuggingFace dense retriever wrapped to honour `BaseEncoder`."""

    def __init__(self, config: DenseEncoderConfig, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.name = config.display_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map[config.torch_dtype]

        logger.info(
            "Loading dense encoder %s (dtype=%s)", config.model_name_or_path, config.torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def hidden_size(self) -> int:
        return getattr(self.model.config, "hidden_size", -1)

    def _encode_truncated(self, texts: list[str], batch_size: int) -> "np.ndarray":
        import torch

        embeddings: list[np.ndarray] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        # Long-context baselines take hours over a 45K-document corpus; without a
        # heartbeat a run is indistinguishable from a hang.
        _t0 = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            _b = i // batch_size
            if _b and _b % max(1, n_batches // 100) == 0:
                done = i / len(texts)
                el = time.perf_counter() - _t0
                logger.info(
                    "encode(%s) %5.1f%%  %d/%d docs  %.0fs elapsed  ETA %.0fs",
                    self.name, 100 * done, i, len(texts), el, el / done - el if done else 0,
                )
            enc = self.tokenizer(
                batch,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**enc)
            pooled = _pool(outputs.last_hidden_state, enc["attention_mask"], self.config.pooling)
            if self.config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.float().cpu().numpy())
        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, self.hidden_size))

    def _encode_chunk_pool(self, texts: list[str], batch_size: int) -> "np.ndarray":
        """Tokenize each doc into K-token chunks, encode each chunk, mean-pool the chunk embeddings."""
        import torch

        out: list[np.ndarray] = []
        _t0 = time.perf_counter()
        for _di, text in enumerate(texts):
            if _di and _di % max(1, len(texts) // 100) == 0:
                done = _di / len(texts)
                el = time.perf_counter() - _t0
                logger.info(
                    "encode(%s) %5.1f%%  %d/%d docs  %.0fs elapsed  ETA %.0fs",
                    self.name, 100 * done, _di, len(texts), el, el / done - el if done else 0,
                )
            chunks = self._chunk_tokenize(text, self.config.chunk_size)
            if not chunks:
                out.append(np.zeros(self.hidden_size, dtype=np.float32))
                continue
            chunk_embs: list[np.ndarray] = []
            for j in range(0, len(chunks), batch_size):
                batch_chunks = chunks[j : j + batch_size]
                enc = self.tokenizer.pad(
                    [{"input_ids": c} for c in batch_chunks],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**enc)
                pooled = _pool(outputs.last_hidden_state, enc["attention_mask"], self.config.pooling)
                chunk_embs.append(pooled.float().cpu().numpy())
            doc_emb = np.concatenate(chunk_embs, axis=0).mean(axis=0)
            if self.config.normalize:
                doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-12)
            out.append(doc_emb)
        return np.stack(out, axis=0)

    def _chunk_tokenize(self, text: str, chunk_size: int) -> list[list[int]]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        usable = chunk_size - (1 if cls_id is not None else 0) - (1 if sep_id is not None else 0)
        chunks: list[list[int]] = []
        for i in range(0, len(ids), usable):
            chunk = ids[i : i + usable]
            if cls_id is not None:
                chunk = [cls_id] + chunk
            if sep_id is not None:
                chunk = chunk + [sep_id]
            chunks.append(chunk)
        return chunks

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        side: Literal["query", "document"] = "document",
    ) -> "np.ndarray":
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.hidden_size))
        prefix = self.config.query_prefix if side == "query" else self.config.document_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        bs = batch_size or 8
        if self.config.inference_protocol == "chunk_pool":
            return self._encode_chunk_pool(texts, bs)
        return self._encode_truncated(texts, bs)


__all__ = ["DenseEncoder", "DenseEncoderConfig", "BASELINES"]

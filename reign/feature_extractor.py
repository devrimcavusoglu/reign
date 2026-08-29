# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright 2023 Devrim Çavuşoğlu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Part of this file is taken and adapted from Huggingface transformer's
Pipeline class. We opted for not directly inheriting from the base `Pipeline`
class. The main reason behind this is that we want to use the Pipeline as a
feature extraction pipeline and that we require a more delicate batching for building
inputs for REIGN. The design of `Pipeline` class(es) of `transformers`, on the other
hand, are built with the e2e inference logic kept in mind. Source:
https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py
"""

import hashlib
import logging
import math
import os
from collections import Counter, UserDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)

# Import profiling utilities
try:
    from reign.utils import profile_function
except ImportError:
    # Fallback if profiling not available
    def profile_function(name=None):
        def decorator(func):
            return func

        return decorator


class EmbeddingCache:
    """
    A caching mechanism for embeddings using h5py for fast I/O operations.

    This version caches embeddings per individual text instance along with their
    chunking metadata, allowing proper reconstruction of batches with correct padding.

    The cache is organized as:
    - cache_root/
      - model_name_hash/
        - dataset_name_split_hash.h5

    Each h5 file contains groups for each text instance:
    - text_0/
      - embeddings: (num_chunks, hidden_dim) array
      - attention_mask: (num_chunks,) array indicating valid chunks
      - metadata: chunk count, original_length, etc.
    - text_1/
      - ...
    """

    def __init__(self, cache_root: Union[str, Path] = None, enable_cache: bool = True):
        """
        Initialize the embedding cache.

        Args:
            cache_root: Root directory for cache storage. Defaults to ~/.reign_cache
            enable_cache: Whether to enable caching functionality
        """
        self.enable_cache = enable_cache
        # Set before the early return: __del__ closes this unconditionally, and a
        # cache-disabled instance would otherwise raise AttributeError at teardown.
        self._open_files = {}
        if not enable_cache:
            return

        if cache_root is None:
            cache_root = Path.home() / ".reign_cache"

        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Add cache file handles to reduce I/O overhead
        self._open_files = {}

        logger.info(f"Embedding cache initialized at: {self.cache_root}")

    def __del__(self):
        """Close any open cache files when the cache is destroyed."""
        self._close_all_files()

    def _close_all_files(self):
        """Close all open cache files."""
        for file_handle in self._open_files.values():
            try:
                if file_handle is not None:
                    file_handle.close()
            except:
                pass
        self._open_files.clear()

    def _get_cache_file_handle(self, cache_path: Path, mode: str = "r"):
        """
        Get a file handle for the cache file, keeping it open for better performance.

        Args:
            cache_path: Path to the cache file
            mode: File mode ('r' for read, 'w' for write)

        Returns:
            h5py.File handle
        """
        cache_key = str(cache_path)

        # Close and remove if we need a different mode
        if cache_key in self._open_files:
            current_file = self._open_files[cache_key]
            if current_file is None or current_file.mode != mode:
                try:
                    if current_file is not None:
                        current_file.close()
                except:
                    pass
                del self._open_files[cache_key]

        # Open new file if not in cache
        if cache_key not in self._open_files:
            try:
                self._open_files[cache_key] = h5py.File(cache_path, mode)
            except Exception as e:
                logger.error(f"Failed to open cache file {cache_path}: {e}")
                return None

        return self._open_files[cache_key]

    def _get_cache_path(
        self,
        model_name: str,
        chunk_size: int,
        dataset_identifier: str,
        stride: Optional[int] = None,
    ) -> Path:
        """Get the full path to a cache file.

        When ``stride`` is ``None`` or equal to ``chunk_size`` (legacy
        non-overlapping chunking), the hash input is the original
        ``f"{model_name}_{chunk_size}"`` so existing caches keep their paths.
        When ``stride != chunk_size`` (sliding-window chunking), the hash
        input becomes ``f"{model_name}_{chunk_size}_s{stride}"`` so caches at
        different strides are kept separate.
        """
        if not self.enable_cache:
            return None

        if stride is None or stride == chunk_size:
            key = f"{model_name}_{chunk_size}"
        else:
            key = f"{model_name}_{chunk_size}_s{stride}"
        model_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        model_cache_dir = self.cache_root / model_hash
        model_cache_dir.mkdir(exist_ok=True)

        dataset_hash = hashlib.md5(dataset_identifier.encode()).hexdigest()[:16]
        cache_file = model_cache_dir / f"{dataset_hash}.h5"

        return cache_file

    def has_cache(
        self,
        model_name: str,
        chunk_size: int,
        dataset_identifier: str,
        stride: Optional[int] = None,
    ) -> bool:
        """
        Check if a cache file exists for the given parameters.

        Args:
            model_name: Name or path of the model
            chunk_size: Chunk size used for tokenization
            dataset_identifier: Unique identifier for the dataset
            stride: Stride used for sliding-window chunking (``None`` or
                equal to ``chunk_size`` selects the legacy non-overlapping
                cache layout)

        Returns:
            True if cache exists and is valid, False otherwise
        """
        if not self.enable_cache:
            return False

        cache_path = self._get_cache_path(model_name, chunk_size, dataset_identifier, stride=stride)

        if not cache_path.exists():
            return False

        # Verify cache integrity
        try:
            with h5py.File(cache_path, "r") as f:
                # Check if we have at least one text group
                text_groups = [key for key in f.keys() if key.startswith("text_")]
                return len(text_groups) > 0
        except (OSError, KeyError):
            logger.warning(f"Cache file corrupted, will regenerate: {cache_path}")
            return False

    def save_cache(
        self,
        embeddings_list: List[torch.Tensor],
        attention_masks_list: List[torch.Tensor],
        model_name: str,
        chunk_size: int,
        dataset_identifier: str,
        metadata: Dict = None,
        per_instance_metadata: List[Dict] = None,
        stride: Optional[int] = None,
    ) -> None:
        """
        Save per-instance embeddings to cache (without storing original texts for performance).

        Args:
            embeddings_list: List of embedding tensors, one per text instance
            attention_masks_list: List of attention masks, one per text instance
            model_name: Name or path of the model
            chunk_size: Chunk size used for tokenization
            dataset_identifier: Unique identifier for the dataset
            metadata: Additional global metadata to store
            per_instance_metadata: List of metadata dicts, one per text instance (only essential fields stored)
        """
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(model_name, chunk_size, dataset_identifier, stride=stride)

        # Define essential metadata fields to preserve for training/evaluation
        essential_fields = {
            "article_id",
            "reference_article_id",
            "other_article_id",
            "article_type",
            "dataset_idx",
        }

        try:
            # Close any existing file handle first for write mode
            cache_key = str(cache_path)
            if cache_key in self._open_files:
                try:
                    self._open_files[cache_key].close()
                except:
                    pass
                del self._open_files[cache_key]

            f = self._get_cache_file_handle(cache_path, "w")
            if f is None:
                logger.error(f"Failed to open cache file for writing: {cache_path}")
                return

            # Store global metadata
            f.attrs["model_name"] = model_name
            f.attrs["chunk_size"] = chunk_size
            f.attrs["dataset_identifier"] = dataset_identifier
            f.attrs["num_texts"] = len(embeddings_list)

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value

            # Store each text instance separately (without text content)
            for i, (embeddings, attention_mask) in enumerate(
                zip(embeddings_list, attention_masks_list)
            ):
                text_group = f.create_group(f"text_{i}")

                # Convert to numpy for h5py storage
                embeddings_np = (
                    embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
                )
                attention_mask_np = (
                    attention_mask.cpu().numpy()
                    if isinstance(attention_mask, torch.Tensor)
                    else attention_mask
                )

                # Store embeddings and attention mask
                text_group.create_dataset("embeddings", data=embeddings_np, compression="gzip")
                text_group.create_dataset("attention_mask", data=attention_mask_np, compression="gzip")

                # Store only embedding-related metadata
                text_group.attrs["num_chunks"] = len(embeddings_np)

                # Store only essential per-instance metadata fields
                if per_instance_metadata and i < len(per_instance_metadata):
                    instance_metadata = per_instance_metadata[i]
                    if instance_metadata:
                        for key, value in instance_metadata.items():
                            if key in essential_fields:
                                if isinstance(value, (str, int, float, bool)):
                                    text_group.attrs[f"metadata_{key}"] = value
                                elif value is None:
                                    text_group.attrs[f"metadata_{key}"] = "None"

            # Close the file after writing to free up handle
            f.close()
            if str(cache_path) in self._open_files:
                del self._open_files[str(cache_path)]

            logger.info(
                f"Saved {len(embeddings_list)} text instances to cache (embeddings + essential metadata only): {cache_path}"
            )

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            # Clean up corrupted file
            if cache_path.exists():
                cache_path.unlink()

    def load_cache(
        self,
        model_name: str,
        chunk_size: int,
        dataset_identifier: str,
        text_indices: Optional[List[int]] = None,
        stride: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Load per-instance embeddings and essential metadata from cache.

        Args:
            model_name: Name or path of the model
            chunk_size: Chunk size used for tokenization
            dataset_identifier: Unique identifier for the dataset
            text_indices: Optional list of text indices to load (if None, load all)

        Returns:
            Tuple of (embeddings_list, attention_masks_list, metadata_list)
        """
        if not self.enable_cache:
            return None, None, None

        cache_path = self._get_cache_path(model_name, chunk_size, dataset_identifier, stride=stride)

        if not cache_path.exists():
            return None, None, None

        try:
            # Use file handle for better performance
            f = self._get_cache_file_handle(cache_path, "r")
            if f is None:
                return None, None, None

            num_texts = f.attrs.get("num_texts", 0)

            # Determine which indices to load
            if text_indices is None:
                indices_to_load = list(range(num_texts))
            else:
                indices_to_load = text_indices

            embeddings_list = []
            attention_masks_list = []
            metadata_list = []

            for i in indices_to_load:
                if f"text_{i}" not in f:
                    logger.warning(f"Missing text_{i} in cache file")
                    continue

                text_group = f[f"text_{i}"]

                # Load embeddings and attention mask
                embeddings_np = text_group["embeddings"][:]
                attention_mask_np = text_group["attention_mask"][:]

                embeddings_list.append(torch.from_numpy(embeddings_np))
                attention_masks_list.append(torch.from_numpy(attention_mask_np))

                # Always load essential metadata
                instance_metadata = {}
                for key in text_group.attrs.keys():
                    if key.startswith("metadata_"):
                        metadata_key = key[9:]  # Remove 'metadata_' prefix
                        value = text_group.attrs[key]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        if value == "None":
                            value = None
                        instance_metadata[metadata_key] = value
                metadata_list.append(instance_metadata)

            logger.debug(f"Loaded {len(embeddings_list)} text instances from cache: {cache_path}")

            return embeddings_list, attention_masks_list, metadata_list

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None, None, None

    @profile_function("cache.load_cache")
    def load_cache_profiled(self, *args, **kwargs):
        """Profiled version of load_cache."""
        return self.load_cache(*args, **kwargs)

    def get_cache_size(
        self,
        model_name: str,
        chunk_size: int,
        dataset_identifier: str,
        stride: Optional[int] = None,
    ) -> int:
        """Get the number of cached text instances."""
        if not self.enable_cache:
            return 0

        cache_path = self._get_cache_path(model_name, chunk_size, dataset_identifier, stride=stride)

        if not cache_path.exists():
            return 0

        try:
            with h5py.File(cache_path, "r") as f:
                return f.attrs.get("num_texts", 0)
        except Exception:
            return 0

    def clear_cache(self, model_name: str = None) -> None:
        """
        Clear cache files.

        Args:
            model_name: If provided, only clear cache for this model. Otherwise clear all.
        """
        if not self.enable_cache:
            return

        if model_name:
            model_hash = hashlib.md5(model_name.encode()).hexdigest()[:16]
            model_cache_dir = self.cache_root / model_hash
            if model_cache_dir.exists():
                for cache_file in model_cache_dir.glob("*.h5"):
                    cache_file.unlink()
                    logger.info(f"Cleared cache file: {cache_file}")
        else:
            for cache_file in self.cache_root.glob("**/*.h5"):
                cache_file.unlink()
                logger.info(f"Cleared cache file: {cache_file}")

    def get_cache_info(self) -> Dict:
        """Get information about cached files."""
        if not self.enable_cache:
            return {}

        cache_info = {
            "cache_root": str(self.cache_root),
            "total_files": 0,
            "total_size_mb": 0,
            "models": {},
        }

        for cache_file in self.cache_root.glob("**/*.h5"):
            cache_info["total_files"] += 1
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            cache_info["total_size_mb"] += file_size_mb

            try:
                with h5py.File(cache_file, "r") as f:
                    model_name = f.attrs.get("model_name", "unknown")
                    if model_name not in cache_info["models"]:
                        cache_info["models"][model_name] = {"files": 0, "size_mb": 0, "total_texts": 0}

                    cache_info["models"][model_name]["files"] += 1
                    cache_info["models"][model_name]["size_mb"] += file_size_mb
                    cache_info["models"][model_name]["total_texts"] += f.attrs.get("num_texts", 0)

            except Exception:
                pass  # Skip corrupted files

        return cache_info


class ReignFeatureExtractor:
    def __init__(
        self,
        batch_size: int,
        model_name_or_path: str = "thenlper/gte-large",
        chunk_size: int = 512,
        stride: int = 384,
        device: str | int = "cpu",
        cache_root: Union[str, Path] = None,
        enable_cache: bool = True,
    ):
        """Initialise the feature extractor.

        Args:
            chunk_size: Number of tokens per chunk passed to the GN.
            stride: Step between successive chunks. ``stride == chunk_size``
                gives the legacy non-overlapping chunking; ``stride <
                chunk_size`` produces ToBERT-style overlapping chunks where
                each pair of adjacent chunks shares ``chunk_size - stride``
                tokens. Defaults to 384 (25% overlap at ``chunk_size=512``).
        """
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.max_num_chunks = self.model.config.max_position_embeddings
        self.chunk_size = chunk_size
        if stride <= 0 or stride > chunk_size:
            raise ValueError(
                f"stride must satisfy 0 < stride <= chunk_size; got stride={stride}, "
                f"chunk_size={chunk_size}"
            )
        self.stride = stride

        # Initialize cache
        self.cache = EmbeddingCache(cache_root=cache_root, enable_cache=enable_cache)
        logger.info(
            "ReignFeatureExtractor initialized with model: %s, chunk_size=%d, stride=%d, cache_enabled: %s",
            model_name_or_path,
            chunk_size,
            stride,
            enable_cache,
        )

    def __call__(self, inputs: list[str], dataset_identifier: str = None, use_cache: bool = True):
        """
        Extract features with caching support.

        Args:
            inputs: List of input texts
            dataset_identifier: Unique identifier for the dataset (for caching)
            use_cache: Whether to use cache if available

        Returns:
            Extracted embeddings tensor with proper batching and padding
        """
        # Check if we can use cache
        if (
            use_cache
            and dataset_identifier
            and self.cache.has_cache(
                self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
            )
        ):
            logger.info(f"Loading embeddings from cache for dataset: {dataset_identifier}")
            # Load individual embeddings and reconstruct batch
            return self._load_and_batch_from_cache(inputs, dataset_identifier)

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(inputs)} texts...")
        model_inputs, overflow_to_sample_mapping = self.preprocess(inputs=inputs)
        model_outputs = self.forward(model_inputs)
        encodings = self.postprocess(model_outputs, overflow_to_sample_mapping)

        # Save to cache if requested
        if use_cache and dataset_identifier:
            logger.info(f"Saving embeddings to cache for dataset: {dataset_identifier}")
            self._save_to_cache(
                inputs, model_inputs, model_outputs, overflow_to_sample_mapping, dataset_identifier
            )

        return encodings.to(self.device)

    def _load_and_batch_from_cache(self, inputs: List[str], dataset_identifier: str) -> BatchEncoding:
        """Load individual cached embeddings and reconstruct batch with proper padding."""

        # For now, we need to determine which indices these texts correspond to
        # This is a limitation - we need a way to map texts to their cached indices
        # For the current implementation, we'll assume the texts are requested in order
        cached_size = self.cache.get_cache_size(
            self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
        )

        if len(inputs) > cached_size:
            logger.warning(f"Requested {len(inputs)} texts but cache only has {cached_size}")
            # Fall back to computing embeddings
            return self._compute_embeddings_no_cache(inputs)

        # Load the individual embeddings
        embeddings_list, attention_masks_list, _ = self.cache.load_cache(
            self.model_name_or_path,
            self.chunk_size,
            dataset_identifier,
            text_indices=list(range(len(inputs))),
            stride=self.stride,
        )

        if embeddings_list is None:
            logger.warning("Failed to load from cache, computing embeddings")
            return self._compute_embeddings_no_cache(inputs)

        # Reconstruct batch with proper padding
        return self._reconstruct_batch_encoding(embeddings_list, attention_masks_list)

    def _compute_embeddings_no_cache(self, inputs: List[str]) -> BatchEncoding:
        """Compute embeddings without caching."""
        model_inputs, overflow_to_sample_mapping = self.preprocess(inputs=inputs)
        model_outputs = self.forward(model_inputs)
        return self.postprocess(model_outputs, overflow_to_sample_mapping).to(self.device)

    def _reconstruct_batch_encoding(
        self, embeddings_list: List[torch.Tensor], attention_masks_list: List[torch.Tensor]
    ) -> BatchEncoding:
        """Reconstruct BatchEncoding from individual cached embeddings."""

        if not embeddings_list:
            # Return empty tensor with correct shape
            hidden_size = self.model.config.hidden_size
            dummy_encoding = BatchEncoding(
                dict(
                    inputs_embeds=torch.zeros((1, 1, hidden_size), device=torch.device("cpu")),
                    attention_mask=torch.zeros((1, 1), dtype=torch.int64, device=torch.device("cpu")),
                )
            )
            return dummy_encoding

        # Filter out empty embeddings and their corresponding masks
        valid_embeddings = []
        valid_attention_masks = []

        for emb, mask in zip(embeddings_list, attention_masks_list):
            if emb.shape[0] > 0:  # Only include non-empty embeddings
                valid_embeddings.append(emb)
                valid_attention_masks.append(mask)

        # If all embeddings are empty, return a minimal dummy encoding
        if not valid_embeddings:
            hidden_size = self.model.config.hidden_size
            dummy_encoding = BatchEncoding(
                dict(
                    inputs_embeds=torch.zeros((1, 1, hidden_size), device=torch.device("cpu")),
                    attention_mask=torch.zeros((1, 1), dtype=torch.int64, device=torch.device("cpu")),
                )
            )
            return dummy_encoding

        # Find the maximum number of chunks for padding
        max_chunks = max(emb.shape[0] for emb in valid_embeddings)
        hidden_size = valid_embeddings[0].shape[1]
        batch_size = len(valid_embeddings)

        # Create padded tensors
        batched_embeddings = torch.zeros(
            batch_size, max_chunks, hidden_size, device=torch.device("cpu"), requires_grad=False
        )
        batched_attention_mask = torch.zeros(
            batch_size, max_chunks, requires_grad=False, dtype=torch.int64
        )

        # Fill in the actual embeddings and attention masks
        for i, (embeddings, attention_mask) in enumerate(zip(valid_embeddings, valid_attention_masks)):
            num_chunks = embeddings.shape[0]
            batched_embeddings[i, :num_chunks, :] = embeddings
            # Ensure attention_mask has the correct shape
            if attention_mask.shape[0] == num_chunks:
                batched_attention_mask[i, :num_chunks] = attention_mask
            else:
                # Create a new attention mask with the correct shape
                batched_attention_mask[i, :num_chunks] = torch.ones(num_chunks, dtype=torch.int64)

        return BatchEncoding(
            dict(
                inputs_embeds=batched_embeddings,
                attention_mask=batched_attention_mask,
            )
        )

    def _save_to_cache(
        self,
        inputs: List[str],
        model_inputs: BatchEncoding,
        model_outputs: torch.Tensor,
        overflow_to_sample_mapping: torch.Tensor,
        dataset_identifier: str,
    ) -> None:
        """Save individual text embeddings to cache."""

        # Reconstruct individual embeddings from the batch
        embeddings_list = []
        attention_masks_list = []

        # Group chunks by original text using overflow mapping
        counts = Counter(overflow_to_sample_mapping.tolist())
        sorted_counts = dict(sorted(counts.items()))

        s = 0
        for text_idx, num_chunks in sorted_counts.items():
            # Extract embeddings for this text
            text_embeddings = model_outputs[s : s + num_chunks, :].cpu()
            text_attention_mask = torch.ones(num_chunks, dtype=torch.int64)

            embeddings_list.append(text_embeddings)
            attention_masks_list.append(text_attention_mask)
            s += num_chunks

        # Save to cache
        self.cache.save_cache(
            embeddings_list=embeddings_list,
            attention_masks_list=attention_masks_list,
            model_name=self.model_name_or_path,
            chunk_size=self.chunk_size,
            dataset_identifier=dataset_identifier,
            stride=self.stride,
            metadata={
                "num_texts": len(inputs),
                "device": str(self.device),
            },
        )

    def compute_and_cache_dataset_embeddings(
        self, texts: List[str], dataset_identifier: str, force_recompute: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute and cache embeddings for an entire dataset, storing per-instance.

        Args:
            texts: List of all texts in the dataset
            dataset_identifier: Unique identifier for the dataset
            force_recompute: Whether to force recomputation even if cache exists

        Returns:
            List of (embeddings, attention_mask) tuples, one per text
        """
        # Check if cache exists and we don't need to force recompute
        if not force_recompute and self.cache.has_cache(
            self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
        ):
            cached_size = self.cache.get_cache_size(
                self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
            )
            if cached_size == len(texts):
                logger.info(f"Loading existing cache for dataset: {dataset_identifier}")
                embeddings_list, attention_masks_list, _ = self.cache.load_cache(
                    self.model_name_or_path,
                    self.chunk_size,
                    dataset_identifier,
                    stride=self.stride,
                )
                if embeddings_list is not None:
                    return list(zip(embeddings_list, attention_masks_list))
            else:
                logger.warning("Cache size mismatch, recomputing embeddings...")

        logger.info(f"Computing embeddings for {len(texts)} texts in dataset: {dataset_identifier}")

        all_embeddings_list = []
        all_attention_masks_list = []

        # Process in batches to avoid memory issues
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with tqdm(total=num_batches, desc=f"Computing embeddings for {dataset_identifier}") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                # Compute embeddings for this batch
                model_inputs, overflow_to_sample_mapping = self.preprocess(inputs=batch_texts)
                model_outputs = self.forward(model_inputs)

                # Extract individual embeddings from batch
                counts = Counter(overflow_to_sample_mapping.tolist())
                sorted_counts = dict(sorted(counts.items()))

                s = 0
                for text_idx, num_chunks in sorted_counts.items():
                    # Extract embeddings for this text
                    text_embeddings = model_outputs[s : s + num_chunks, :].cpu()
                    text_attention_mask = torch.ones(num_chunks, dtype=torch.int64)

                    all_embeddings_list.append(text_embeddings)
                    all_attention_masks_list.append(text_attention_mask)
                    s += num_chunks

                pbar.update(1)

        # Save to cache
        logger.info(f"Saving {len(all_embeddings_list)} text instances to cache...")
        self.cache.save_cache(
            embeddings_list=all_embeddings_list,
            attention_masks_list=all_attention_masks_list,
            model_name=self.model_name_or_path,
            chunk_size=self.chunk_size,
            dataset_identifier=dataset_identifier,
            stride=self.stride,
            metadata={
                "num_texts": len(texts),
                "computed_in_batches": True,
            },
        )

        return list(zip(all_embeddings_list, all_attention_masks_list))

    def compute_and_cache_dataset_embeddings_with_metadata(
        self,
        texts: List[str],
        dataset_identifier: str,
        metadata_list: List[Dict],
        force_recompute: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute and cache embeddings for an entire dataset with metadata, storing per-instance.

        Args:
            texts: List of all texts in the dataset
            dataset_identifier: Unique identifier for the dataset
            metadata_list: List of metadata dicts, one per text
            force_recompute: Whether to force recomputation even if cache exists

        Returns:
            List of (embeddings, attention_mask) tuples, one per text
        """
        # Check if cache exists and we don't need to force recompute
        if not force_recompute and self.cache.has_cache(
            self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
        ):
            cached_size = self.cache.get_cache_size(
                self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
            )
            if cached_size == len(texts):
                logger.info(f"Loading existing cache for dataset: {dataset_identifier}")
                embeddings_list, attention_masks_list, _ = self.cache.load_cache(
                    self.model_name_or_path,
                    self.chunk_size,
                    dataset_identifier,
                    stride=self.stride,
                )
                if embeddings_list is not None:
                    return list(zip(embeddings_list, attention_masks_list))
            else:
                logger.warning("Cache size mismatch, recomputing embeddings...")

        logger.info(f"Computing embeddings for {len(texts)} texts in dataset: {dataset_identifier}")

        all_embeddings_list = []
        all_attention_masks_list = []

        # Process in batches to avoid memory issues
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with tqdm(total=num_batches, desc=f"Computing embeddings for {dataset_identifier}") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                # Compute embeddings for this batch
                model_inputs, overflow_to_sample_mapping = self.preprocess(inputs=batch_texts)
                model_outputs = self.forward(model_inputs)

                # Extract individual embeddings from batch
                counts = Counter(overflow_to_sample_mapping.tolist())
                sorted_counts = dict(sorted(counts.items()))

                s = 0
                for text_idx, num_chunks in sorted_counts.items():
                    # Extract embeddings for this text
                    text_embeddings = model_outputs[s : s + num_chunks, :].cpu()
                    text_attention_mask = torch.ones(num_chunks, dtype=torch.int64)

                    all_embeddings_list.append(text_embeddings)
                    all_attention_masks_list.append(text_attention_mask)
                    s += num_chunks

                pbar.update(1)

        # Save to cache with metadata
        logger.info(f"Saving {len(all_embeddings_list)} text instances to cache...")
        self.cache.save_cache(
            embeddings_list=all_embeddings_list,
            attention_masks_list=all_attention_masks_list,
            model_name=self.model_name_or_path,
            chunk_size=self.chunk_size,
            dataset_identifier=dataset_identifier,
            stride=self.stride,
            metadata={
                "num_texts": len(texts),
                "computed_in_batches": True,
            },
            per_instance_metadata=metadata_list,
        )

        return list(zip(all_embeddings_list, all_attention_masks_list))

    def get_cached_embeddings(
        self, dataset_identifier: str, indices: Optional[List[int]] = None
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get cached embeddings for a dataset, optionally filtering by indices.

        Args:
            dataset_identifier: Unique identifier for the dataset
            indices: Optional list of indices to retrieve

        Returns:
            List of (embeddings, attention_mask) tuples or None if not cached
        """
        if not self.cache.has_cache(
            self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
        ):
            return None

        embeddings_list, attention_masks_list, _ = self.cache.load_cache(
            self.model_name_or_path,
            self.chunk_size,
            dataset_identifier,
            text_indices=indices,
            stride=self.stride,
        )

        if embeddings_list is None:
            return None

        # CPU tensors only — see ``get_cached_embeddings_with_metadata`` for why.
        return list(zip(embeddings_list, attention_masks_list))

    def get_cached_embeddings_with_metadata(
        self, dataset_identifier: str, indices: Optional[List[int]] = None
    ) -> Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Dict]]]:
        """
        Get cached embeddings and metadata for a dataset, optionally filtering by indices.

        Args:
            dataset_identifier: Unique identifier for the dataset
            indices: Optional list of indices to retrieve

        Returns:
            Tuple of (List of (embeddings, attention_mask) tuples, List of metadata dicts) or None if not cached
        """
        if not self.cache.has_cache(
            self.model_name_or_path, self.chunk_size, dataset_identifier, stride=self.stride
        ):
            return None

        embeddings_list, attention_masks_list, metadata_list = self.cache.load_cache(
            self.model_name_or_path,
            self.chunk_size,
            dataset_identifier,
            text_indices=indices,
            stride=self.stride,
        )

        if embeddings_list is None:
            return None

        # Return CPU tensors. ``__getitem__`` runs in DataLoader worker processes
        # which can't touch CUDA (forked workers fail re-init); Lightning's
        # ``transfer_batch_to_device`` moves the batch to GPU before
        # ``training_step``. This used to be ``emb.to(self.device, ...)`` which
        # crashed with ``Cannot re-initialize CUDA in forked subprocess`` when
        # ``--data-loader-num-workers > 0``.
        embeddings_with_masks = list(zip(embeddings_list, attention_masks_list))
        return embeddings_with_masks, metadata_list

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor, normalize: bool = True
    ) -> Tensor:
        """
        Average pooling for last hidden states.
        Taken and adapted from https://huggingface.co/thenlper/gte-large.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if normalize:
            return F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @contextmanager
    def device_placement(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                yield
        else:
            yield

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        elif isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def batch_iterator(self, inputs):
        start = 0
        end = self.batch_size
        step = self.batch_size

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        iterate = end <= len(input_ids) if start > 0 else True
        while iterate:
            yield BatchEncoding(
                data={
                    "input_ids": input_ids[start:end, ...],
                    "token_type_ids": token_type_ids[start:end, ...],
                    "attention_mask": attention_mask[start:end, ...],
                }
            )
            if end == len(input_ids):
                break
            start += step
            if end + step > len(input_ids):
                end = len(input_ids)
            else:
                end += step

    def _get_total_steps(self, model_inputs):
        return math.ceil(len(model_inputs["input_ids"]) / self.batch_size)

    def forward(self, model_inputs):
        self.model.eval()
        with self.device_placement():
            with torch.no_grad():
                model_outputs = []
                for batch in tqdm(
                    self.batch_iterator(model_inputs),
                    desc="Batches",
                    total=self._get_total_steps(model_inputs),
                ):
                    batch = self._ensure_tensor_on_device(batch, device=self.device)
                    outputs = self._forward(batch)
                    outputs = self._ensure_tensor_on_device(outputs, device=torch.device("cpu"))
                    model_outputs.append(outputs)

                # Check if model_outputs is empty
                if not model_outputs:
                    # Return an empty tensor with the correct shape
                    # Get the expected shape from the model's config
                    hidden_size = self.model.config.hidden_size
                    return torch.zeros((0, hidden_size), device=torch.device("cpu"))

        return torch.vstack(model_outputs)

    def preprocess(self, inputs: list[str]):
        # HF tokenizers' ``stride`` arg controls the overlap with overflow tokens:
        # at stride == chunk_size we recover the legacy non-overlapping behaviour
        # (no shared tokens between adjacent chunks), and stride < chunk_size
        # produces sliding-window chunks sharing ``chunk_size - stride`` tokens.
        tokenizer_stride = 0 if self.stride == self.chunk_size else self.chunk_size - self.stride
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.chunk_size,
            padding=True,
            return_tensors="pt",
            truncation=True,
            return_overflowing_tokens=True,
            stride=tokenizer_stride,
        )
        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        return model_inputs, overflow_to_sample_mapping

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        embeddings = self.average_pool(outputs.last_hidden_state, model_inputs["attention_mask"])
        return embeddings.to(self.device)

    def postprocess(self, model_outputs, overflow_to_sample_mapping) -> BatchEncoding:
        # Handle the case where model_outputs is empty or has zero dimension
        dummy_encoding = BatchEncoding(
            dict(
                inputs_embeds=torch.zeros(
                    (1, 1, self.model.config.hidden_size), device=torch.device("cpu")
                ),
                attention_mask=torch.zeros((1, 1), dtype=torch.int64, device=torch.device("cpu")),
            )
        )
        if isinstance(model_outputs, torch.Tensor) and model_outputs.shape[0] == 0:
            return dummy_encoding

        # If overflow_to_sample_mapping is empty, return a minimal valid encoding
        elif len(overflow_to_sample_mapping) == 0:
            return dummy_encoding

        counts = Counter(overflow_to_sample_mapping.tolist())
        _, max_length = counts.most_common(1)[0]
        N, L, H = (
            len(counts),
            max_length,
            self.model.config.hidden_size,
        )  # batch, sequence, dim
        batched_embeddings = torch.zeros(
            N, L, H, device=torch.device("cpu"), requires_grad=False
        )  # (N, L, H)
        sorted_counts = dict(sorted(counts.items()))
        s = 0
        for idx, length in sorted_counts.items():
            batched_embeddings[idx, 0 : 0 + length, :] += model_outputs[s : s + length, :]
            s += length

        # Prepare attention mask
        attention_mask = torch.ones(N, L, requires_grad=False, dtype=torch.int64)
        for i, count in sorted_counts.items():
            attention_mask[
                i, count:
            ] = 0  # Padding tokens for batching (set positions after valid tokens to 0)
        return BatchEncoding(
            dict(
                inputs_embeds=batched_embeddings,
                attention_mask=attention_mask,
            )
        )

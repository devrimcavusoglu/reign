"""
Test script to verify the caching functionality for ReignFeatureExtractor.

This script demonstrates:
1. Basic embedding computation and caching (per-instance)
2. Loading from cache with proper batching
3. Dataset-level caching with individual embeddings
4. Cache management operations
"""

import logging
import tempfile
from pathlib import Path

import pytest
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_caching():
    """Test basic embedding caching functionality with per-instance storage."""
    logger.info("=== Testing Basic Per-Instance Caching ===")

    # Import here to avoid import issues
    from reign.feature_extractor import ReignFeatureExtractor

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)

        # Initialize feature extractor with caching enabled
        extractor = ReignFeatureExtractor(
            batch_size=2,
            model_name_or_path="thenlper/gte-small",  # Use smaller model for testing
            device="cpu",  # Use CPU for testing
            cache_root=cache_root,
            enable_cache=True,
        )

        # Test texts with different lengths to verify per-instance caching
        test_texts = [
            "Short text.",
            "This is a longer text that should generate multiple chunks when tokenized with overflow to test the chunking mechanism properly.",
            "Medium length text for testing.",
        ]

        dataset_id = "test_basic_caching"

        # First call - should compute embeddings
        logger.info("First call: Computing embeddings...")
        encodings1 = extractor(test_texts, dataset_identifier=dataset_id, use_cache=True)
        logger.info(f"Computed embeddings shape: {encodings1.inputs_embeds.shape}")
        logger.info(f"Attention mask shape: {encodings1.attention_mask.shape}")

        # Second call - should load from cache and reconstruct batch
        logger.info("Second call: Loading from cache...")
        encodings2 = extractor(test_texts, dataset_identifier=dataset_id, use_cache=True)
        logger.info(f"Cached embeddings shape: {encodings2.inputs_embeds.shape}")
        logger.info(f"Attention mask shape: {encodings2.attention_mask.shape}")

        # Verify embeddings are identical (within tolerance due to reconstruction)
        assert torch.allclose(
            encodings1.inputs_embeds, encodings2.inputs_embeds, atol=1e-6
        ), "Cached embeddings do not match computed embeddings"

        # Test cache info
        cache_info = extractor.cache.get_cache_info()
        logger.info(f"Cache info: {cache_info}")


def test_dataset_caching():
    """Test dataset-level caching functionality with per-instance storage."""
    logger.info("=== Testing Dataset Per-Instance Caching ===")

    from reign.feature_extractor import ReignFeatureExtractor

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)

        # Initialize feature extractor
        extractor = ReignFeatureExtractor(
            batch_size=2,
            model_name_or_path="thenlper/gte-small",
            device="cpu",
            cache_root=cache_root,
            enable_cache=True,
        )

        # Test dataset texts with varying lengths
        dataset_texts = [
            "First article.",
            "Second article with much longer content that will likely be split into multiple chunks during tokenization process.",
            "Third article for testing purposes with medium length.",
            "Fourth article to test batch processing.",
            "Fifth and final article to complete the test set with some additional content.",
        ]

        dataset_id = "test_dataset_caching"

        # Compute and cache entire dataset
        logger.info("Computing and caching entire dataset...")
        embeddings_and_masks = extractor.compute_and_cache_dataset_embeddings(dataset_texts, dataset_id)
        logger.info(f"Dataset has {len(embeddings_and_masks)} text instances")

        # Verify each instance has embeddings and attention mask
        for i, (embeddings, attention_mask) in enumerate(embeddings_and_masks):
            logger.info(
                f"Text {i}: embeddings shape {embeddings.shape}, mask shape {attention_mask.shape}"
            )

        # Test retrieving specific indices
        logger.info("Testing index-based retrieval...")
        subset_cached = extractor.get_cached_embeddings(dataset_id, [0, 2, 4])
        assert subset_cached is not None, "Index-based retrieval failed"
        logger.info(f"Retrieved {len(subset_cached)} text instances")
        for i, (emb, mask) in enumerate(subset_cached):
            logger.info(f"Subset text {i}: embeddings {emb.shape}, mask {mask.shape}")


def test_cache_management():
    """Test cache management operations."""
    logger.info("=== Testing Cache Management ===")

    from reign.feature_extractor import ReignFeatureExtractor

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)

        # Initialize feature extractor
        extractor = ReignFeatureExtractor(
            batch_size=2,
            model_name_or_path="thenlper/gte-small",
            device="cpu",
            cache_root=cache_root,
            enable_cache=True,
        )

        # Create some cached data
        test_texts = ["Test sentence for cache management."]
        dataset_id = "test_cache_management"

        extractor(test_texts, dataset_identifier=dataset_id, use_cache=True)

        # Check cache exists. The cache is keyed by stride as well as chunk size
        # (sliding-window chunks live apart from legacy non-overlapping ones), so
        # the lookup has to use the stride the extractor actually wrote with.
        assert extractor.cache.has_cache(
            extractor.model_name_or_path,
            extractor.chunk_size,
            dataset_id,
            stride=extractor.stride,
        ), "Cache creation failed"

        # Check cache size
        cache_size = extractor.cache.get_cache_size(
            extractor.model_name_or_path,
            extractor.chunk_size,
            dataset_id,
            stride=extractor.stride,
        )
        logger.info(f"Cache contains {cache_size} text instances")
        assert cache_size == len(test_texts)

        # Test cache clearing
        logger.info("Testing cache clearing...")
        extractor.cache.clear_cache()

        assert not extractor.cache.has_cache(
            extractor.model_name_or_path,
            extractor.chunk_size,
            dataset_id,
            stride=extractor.stride,
        ), "Cache clearing failed"


def test_batching_consistency():
    """Test that different batch compositions give consistent results when using cache."""
    logger.info("=== Testing Batching Consistency ===")

    from reign.feature_extractor import ReignFeatureExtractor

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_root = Path(temp_dir)

        # Initialize feature extractor
        extractor = ReignFeatureExtractor(
            batch_size=3,  # Use batch size 3
            model_name_or_path="thenlper/gte-small",
            device="cpu",
            cache_root=cache_root,
            enable_cache=True,
        )

        # Test texts
        test_texts = [
            "Text A with short length.",
            "Text B that is much longer and will likely be tokenized into multiple chunks to test the overflow mechanism.",
            "Text C medium length content.",
            "Text D another short one.",
        ]

        dataset_id = "test_batching_consistency"

        # First: compute and cache all texts together
        logger.info("Caching all texts together...")
        extractor.compute_and_cache_dataset_embeddings(test_texts, dataset_id)

        # Now test different batch compositions
        logger.info("Testing different batch combinations...")

        # Batch 1: texts [0, 1]
        result1 = extractor(test_texts[:2], dataset_identifier=dataset_id, use_cache=True)

        # Batch 2: texts [0, 2] (different second text, should give same result for text 0)
        result2 = extractor(
            [test_texts[0], test_texts[2]], dataset_identifier=dataset_id, use_cache=True
        )

        # Extract the embedding for text 0 from both results
        # Text 0 should have identical embeddings regardless of batch composition
        text0_emb1 = result1.inputs_embeds[0]  # First text in first batch
        text0_emb2 = result2.inputs_embeds[0]  # First text in second batch

        assert torch.allclose(
            text0_emb1, text0_emb2, atol=1e-6
        ), "Text embeddings differ with different batch compositions"


def test_cached_dataset():
    """Test the ReignCachedDataset functionality."""
    logger.info("=== Testing ReignCachedDataset ===")

    try:
        from reign.dataset import ReignCachedDataset, create_data_loaders
        from reign.feature_extractor import ReignFeatureExtractor
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

    logger.info("Note: ReignCachedDataset requires actual HuggingFace datasets.")
    logger.info("This test would need real dataset access to run fully.")
    logger.info("The classes have been implemented with per-instance caching support.")

    assert callable(ReignCachedDataset)
    assert callable(create_data_loaders)
    assert callable(ReignFeatureExtractor)


def main():
    """Run all caching tests."""
    logger.info("Starting per-instance caching functionality tests...")

    tests = [
        test_basic_caching,
        test_dataset_caching,
        test_cache_management,
        test_batching_consistency,
        test_cached_dataset,
    ]

    # The tests assert rather than return a status, so a failure surfaces as an
    # exception here exactly as it does under pytest.
    for test_func in tests:
        test_func()
        logger.info(f"✓ {test_func.__name__} passed")

    logger.info(f"\nAll {len(tests)} per-instance caching tests passed.")


if __name__ == "__main__":
    main()

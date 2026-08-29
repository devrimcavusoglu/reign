"""
Evaluation module for REIGN models.

This module provides an Evaluator class for computing metrics on embedding models,
with special handling for partial matches in the evaluation set. Metric formulas
live in `reign.encoders.eval_utils`; this module loads texts via the cached/text
data loaders, builds a relevance matrix from training-side metadata, and then
delegates to `compute_metrics` so training-time and baseline-time evaluation use
the same implementation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from reign.dataset import create_data_loaders
from reign.encoders.eval_utils import compute_metrics as _compute_graded_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for embedding models that computes metrics on embedding quality.

    This evaluator supports both standard evaluation and evaluation with partial matches,
    where some articles are considered partially relevant (e.g., distractor articles).
    """

    def __init__(self, batch_size=32, data_loader=None, top_k=10, use_cached_embeddings=False):
        """
        Initialize the evaluator.

        Args:
            batch_size: Batch size for processing embeddings
            data_loader: DataLoader for evaluation data
            top_k: Top-k value for metrics calculation
            use_cached_embeddings: Whether to expect cached embeddings from the data loader
        """
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.top_k = top_k
        self.use_cached_embeddings = use_cached_embeddings

    def get_embeddings(self, model, data):
        """
        Get embeddings from model, handling both text and cached embedding inputs.

        Args:
            model: The model to get embeddings from
            data: Either text data or cached embedding data (embeddings, masks)

        Returns:
            Tensor of embeddings
        """
        with torch.no_grad():
            if self.use_cached_embeddings:
                # Handle cached embeddings format: (embeddings, masks)
                if isinstance(data, tuple) and len(data) == 2:
                    embeddings, masks = data
                    # Cache reads return CPU tensors (so DataLoader workers don't
                    # trip ``Cannot re-initialize CUDA in forked subprocess``).
                    # The model lives on GPU, so move the inputs here before the
                    # forward pass — Lightning auto-transfers ``training_step``
                    # batches but the eval loop iterates the dataloader manually.
                    device = next(model.parameters()).device
                    if embeddings.device != device:
                        embeddings = embeddings.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                    features = {"inputs_embeds": embeddings, "attention_mask": masks}
                    # Check if this is a ReignLitModel (has .model attribute) or direct REIGN model
                    if hasattr(model, "model"):
                        result = model.model(**features)
                    else:
                        result = model(**features)

                    if hasattr(result, "pooler_output"):
                        return result.pooler_output.cpu()
                    else:
                        return result.cpu()
                else:
                    raise ValueError(
                        f"Expected cached embeddings as (embeddings, masks) tuple, got {type(data)}"
                    )
            else:
                # Handle regular text inputs
                embeddings = model(data)
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
                return embeddings.cpu()

    def evaluate_with_integrated_dataset(self, model, no_detailed_results: bool = False):
        """
        Evaluate a model using query texts against search texts.

        Args:
            model: The model to evaluate
            no_detailed_results: Whether to skip detailed results in output

        Returns:
            Dictionary of metrics
        """
        logger.info("Extracting texts and metadata from evaluation dataset...")
        query_embeddings, query_metadata, search_embeddings, search_metadata = [], [], [], []

        for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Processing batches")):
            if self.use_cached_embeddings:
                if len(batch) == 7:
                    self._process_cached_batch_with_metadata(
                        batch,
                        batch_idx,
                        model,
                        query_embeddings,
                        query_metadata,
                        search_embeddings,
                        search_metadata,
                    )
                else:
                    self._process_cached_batch_legacy(
                        batch,
                        batch_idx,
                        model,
                        query_embeddings,
                        query_metadata,
                        search_embeddings,
                        search_metadata,
                    )
            else:
                self._process_text_batch(
                    batch,
                    batch_idx,
                    model,
                    query_embeddings,
                    query_metadata,
                    search_embeddings,
                    search_metadata,
                )

        query_embeddings = torch.cat(query_embeddings, dim=0)
        search_embeddings = torch.cat(search_embeddings, dim=0)

        # Normalize embeddings for cosine similarity
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        search_embeddings = torch.nn.functional.normalize(search_embeddings, p=2, dim=1)

        # Validate dimensions
        num_queries = query_embeddings.shape[0]
        num_searches = search_embeddings.shape[0]
        num_query_metadata = len(query_metadata)
        num_search_metadata = len(search_metadata)

        print(f"Validation: {num_queries} query embeddings, {num_query_metadata} query metadata")
        print(f"Validation: {num_searches} search embeddings, {num_search_metadata} search metadata")

        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = torch.matmul(query_embeddings, search_embeddings.transpose(0, 1))

        # Create relevance matrix based on metadata
        print("Creating relevance matrix...")
        (
            relevance_matrix,
            fully_relevant_count,
            partially_relevant_count,
        ) = self._create_relevance_matrix(query_metadata, search_metadata, similarity_matrix.shape)

        print(
            f"Relevance assignments - Fully relevant: {fully_relevant_count}, Partially relevant: {partially_relevant_count}"
        )

        if fully_relevant_count == 0 and partially_relevant_count == 0:
            print("ERROR: No relevance found! This will result in zero metrics.")
            print("Sample query metadata:", query_metadata[:3] if query_metadata else "None")
            print("Sample search metadata:", search_metadata[:5] if search_metadata else "None")
            if query_metadata and search_metadata:
                sample_query_id = query_metadata[0].get("article_id")
                sample_search_ref_id = search_metadata[0].get("reference_article_id")
                print(
                    f"Type mismatch check: query_id type={type(sample_query_id)}, search_ref_id type={type(sample_search_ref_id)}"
                )
                print(
                    f"First comparison: {sample_query_id} == {sample_search_ref_id} = {sample_query_id == sample_search_ref_id}"
                )
                str_query_id = str(sample_query_id) if sample_query_id is not None else None
                str_search_ref_id = (
                    str(sample_search_ref_id) if sample_search_ref_id is not None else None
                )
                print(
                    f"String comparison: {str_query_id} == {str_search_ref_id} = {str_query_id == str_search_ref_id}"
                )

        # Get the indices of the top-k most similar search articles for each query
        print(f"Computing top-{self.top_k} indices...")
        _, topk_indices = torch.topk(similarity_matrix, k=self.top_k, dim=1)

        # Compute metrics
        print("Computing metrics...")
        metrics = self._compute_metrics_from_matrices(
            similarity_matrix, relevance_matrix, topk_indices, no_detailed_results
        )

        return metrics

    def _process_cached_batch_with_metadata(
        self,
        batch,
        batch_idx,
        model,
        query_embeddings,
        query_metadata,
        search_embeddings,
        search_metadata,
    ):
        (
            original_embeddings,
            original_masks,
            synthetic_embeddings,
            synthetic_masks,
            distractor_embeddings,
            distractor_masks,
            batch_metadata,
        ) = batch

        # Add original articles as queries
        query_embeddings.append(self.get_embeddings(model, (original_embeddings, original_masks)))
        for meta in batch_metadata["original_metadata"]:
            query_metadata.append(
                {"article_id": meta.get("article_id", f"batch_{batch_idx}_{len(query_metadata)}")}
            )

        # Add synthetic articles to search set
        search_embeddings.append(self.get_embeddings(model, (synthetic_embeddings, synthetic_masks)))
        for meta in batch_metadata["synthetic_metadata"]:
            search_metadata.append(
                {
                    "reference_article_id": meta.get("reference_article_id", ""),
                    "other_article_id": meta.get("other_article_id", None),
                    "article_type": meta.get("article_type", "pair"),
                }
            )

        # Add distractor articles to search set if they exist
        if distractor_embeddings.numel() > 0:
            search_embeddings.append(
                self.get_embeddings(model, (distractor_embeddings, distractor_masks))
            )
            for meta in batch_metadata["distractor_metadata"]:
                search_metadata.append(
                    {
                        "reference_article_id": meta.get("reference_article_id", ""),
                        "other_article_id": meta.get("other_article_id", None),
                        "article_type": meta.get("article_type", "distractor"),
                    }
                )

    def _process_cached_batch_legacy(
        self,
        batch,
        batch_idx,
        model,
        query_embeddings,
        query_metadata,
        search_embeddings,
        search_metadata,
    ):
        (
            original_embeddings,
            original_masks,
            synthetic_embeddings,
            synthetic_masks,
            distractor_embeddings,
            distractor_masks,
        ) = batch

        batch_size = original_embeddings.shape[0]
        # Add original articles as queries
        query_embeddings.append(self.get_embeddings(model, (original_embeddings, original_masks)))
        for i in range(batch_size):
            query_metadata.append({"article_id": f"batch_{batch_idx}_{i}"})

        # Add synthetic articles to search set
        search_embeddings.append(self.get_embeddings(model, (synthetic_embeddings, synthetic_masks)))
        for i in range(batch_size):
            search_metadata.append(
                {
                    "reference_article_id": f"batch_{batch_idx}_{i}",
                    "other_article_id": None,
                    "article_type": "pair",
                }
            )

        # Add distractor articles to search set if they exist
        if distractor_embeddings.numel() > 0:
            search_embeddings.append(
                self.get_embeddings(model, (distractor_embeddings, distractor_masks))
            )
            for i in range(distractor_embeddings.shape[0]):
                orig_idx = i // 2  # Assuming 2 distractors per original article
                search_metadata.append(
                    {
                        "reference_article_id": f"batch_{batch_idx}_{orig_idx}",
                        "other_article_id": None,
                        "article_type": "distractor",
                    }
                )

    def _process_text_batch(
        self,
        batch,
        batch_idx,
        model,
        query_embeddings,
        query_metadata,
        search_embeddings,
        search_metadata,
    ):
        original_articles, synthetic_articles, distractor_articles_list = batch

        if isinstance(original_articles[0], str):
            # Old format - text only, no metadata available
            query_embeddings.append(self.get_embeddings(model, original_articles))
            for i, text in enumerate(original_articles):
                query_metadata.append({"article_id": f"batch_{batch_idx}_orig_{i}"})

            search_embeddings.append(self.get_embeddings(model, synthetic_articles))
            for i, text in enumerate(synthetic_articles):
                search_metadata.append(
                    {
                        "reference_article_id": f"batch_{batch_idx}_orig_{i}",
                        "other_article_id": None,
                        "article_type": "pair",
                    }
                )

            for sample_idx, distractors in enumerate(distractor_articles_list):
                if distractors:
                    search_embeddings.append(self.get_embeddings(model, distractors))
                    for dist_idx, distractor_text in enumerate(distractors):
                        search_metadata.append(
                            {
                                "reference_article_id": f"batch_{batch_idx}_orig_{sample_idx}",
                                "other_article_id": None,
                                "article_type": "distractor",
                            }
                        )
        else:
            # New format - text with metadata
            original_texts = [article["text"] for article in original_articles]
            synthetic_texts = [article["text"] for article in synthetic_articles]

            query_embeddings.append(self.get_embeddings(model, original_texts))
            for orig_article in original_articles:
                aid = orig_article.get("metadata", {}).get("article_id")
                query_metadata.append({"article_id": aid})

            search_embeddings.append(self.get_embeddings(model, synthetic_texts))
            for synth_article in synthetic_articles:
                synth_meta = synth_article.get("metadata", {})
                search_metadata.append(
                    {
                        "reference_article_id": synth_meta.get("reference_article_id", ""),
                        "other_article_id": synth_meta.get("other_article_id", None),
                        "article_type": synth_meta.get("article_type", "pair"),
                    }
                )

            for distractors in distractor_articles_list:
                if distractors:
                    distractor_texts = [d["text"] for d in distractors]
                    search_embeddings.append(self.get_embeddings(model, distractor_texts))
                    for distractor in distractors:
                        distractor_meta = distractor.get("metadata", {})
                        search_metadata.append(
                            {
                                "reference_article_id": distractor_meta.get("reference_article_id", ""),
                                "other_article_id": distractor_meta.get("other_article_id", None),
                                "article_type": distractor_meta.get("article_type", "distractor"),
                            }
                        )

    def _create_relevance_matrix(self, query_metadata, search_metadata, matrix_shape):
        relevance_matrix = torch.zeros(matrix_shape, dtype=torch.float32)
        fully_relevant_count = 0
        partially_relevant_count = 0

        try:
            for i, query_meta in enumerate(query_metadata):
                query_id = query_meta.get("article_id", i)
                for j, search_meta in enumerate(search_metadata):
                    search_reference_id = search_meta.get("reference_article_id")
                    search_other_id = search_meta.get("other_article_id")
                    article_type = search_meta.get("article_type")

                    if query_id == search_reference_id and article_type == "pair":
                        relevance_matrix[i, j] = 2.0  # Fully relevant
                        fully_relevant_count += 1
                    elif query_id in (search_reference_id, search_other_id):
                        relevance_matrix[i, j] = 1.0  # Partially relevant
                        partially_relevant_count += 1
        except IndexError as e:
            print(f"IndexError in relevance matrix creation:")
            print(f"  Matrix shape: {relevance_matrix.shape}")
            print(f"  Query metadata count: {len(query_metadata)}")
            print(f"  Search metadata count: {len(search_metadata)}")
            print(f"  Current query_meta: {query_meta}")
            print(f"  Current search_meta: {search_meta}")
            raise e

        return relevance_matrix, fully_relevant_count, partially_relevant_count

    def evaluate_from_jsonl(self, model, texts, metadata):
        """
        Evaluate a model using texts from a single dataset (traditional method).

        Args:
            model: The model to evaluate
            texts: List of texts
            metadata: List of metadata for texts

        Returns:
            Dictionary of metrics
        """
        # Get embeddings using the model's built-in batching
        print("Computing embeddings...")
        embeddings = model(texts, batch_size=self.batch_size)

        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1))

        # Create relevance matrix based on metadata
        relevance_matrix = torch.zeros_like(similarity_matrix)

        # Fill in relevance matrix based on metadata
        print("Creating relevance matrix...")
        for i, meta_i in enumerate(metadata):
            for j, meta_j in enumerate(metadata):
                if i == j:
                    # Skip self-similarity
                    continue

                # Check if articles are related
                if meta_i["dataset_idx"] == meta_j["dataset_idx"]:
                    # Assign relevance based on article type
                    if meta_j["article_type"] == "pair" or meta_j["article_type"] == "rephrase":
                        relevance_matrix[i, j] = 2.0  # Fully relevant
                    elif meta_j["article_type"] == "distractor" or meta_j["article_type"] == "partial":
                        relevance_matrix[i, j] = 1.0  # Partially relevant

        # Get the indices of the top-k most similar articles for each article
        print(f"Computing top-{self.top_k+1} indices...")
        _, topk_indices = torch.topk(similarity_matrix, k=self.top_k + 1, dim=1)  # +1 to exclude self

        # Remove self-similarity (first column) from topk_indices
        topk_indices = topk_indices[:, 1 : self.top_k + 1]

        # Compute metrics
        print("Computing metrics...")
        metrics = self._compute_metrics_from_matrices(similarity_matrix, relevance_matrix, topk_indices)

        return metrics

    def _compute_metrics_from_matrices(
        self,
        similarity_matrix,
        relevance_matrix,
        topk_indices,
        no_detailed_results: bool = False,
    ):
        """Compute the graded retrieval metric suite (P/R/MAP/nDCG @ top_k).

        Delegates the actual formulas to ``reign.encoders.eval_utils.compute_metrics``
        so this code path and the baseline runners share one implementation.
        Honours REIGN's 2 : 1 : 0 weighting on P/R/MAP and the BEIR-conventional
        exponential gain on nDCG. ``no_detailed_results`` is accepted for back-
        compatibility but currently has no effect.
        """
        import numpy as np

        rel_np = (
            relevance_matrix.detach().cpu().numpy()
            if isinstance(relevance_matrix, torch.Tensor)
            else np.asarray(relevance_matrix)
        ).astype(np.int64)
        top_np = (
            topk_indices.detach().cpu().numpy()
            if isinstance(topk_indices, torch.Tensor)
            else np.asarray(topk_indices)
        ).astype(np.int64)

        graded = _compute_graded_metrics(top_np, rel_np, k=self.top_k)
        # Re-key into the lower-cased names the training loop / Lightning logger expects.
        return {
            f"precision@{self.top_k}": graded[f"P@{self.top_k}"],
            f"recall@{self.top_k}": graded[f"R@{self.top_k}"],
            f"map@{self.top_k}": graded[f"MAP@{self.top_k}"],
            f"ndcg@{self.top_k}": graded[f"nDCG@{self.top_k}"],
        }

    def compute_metrics(self, model):
        """
        Compatibility method for the training script.
        This method should be implemented based on how the data is structured in the training script.

        Args:
            model: The model to evaluate

        Returns:
            Dictionary of metrics
        """
        # This is a placeholder implementation
        # In a real implementation, this would extract data from the model's data loaders
        # and call evaluate_with_integrated_dataset or evaluate_from_jsonl

        # For now, return empty metrics
        return {
            f"precision@{self.top_k}": 0.0,
            f"recall@{self.top_k}": 0.0,
            f"map@{self.top_k}": 0.0,
            f"ndcg@{self.top_k}": 0.0,
            "detailed_results": {},
        }

    def compute_metrics_with_partial_matches(self, model):
        """
        Compatibility method for the training script with partial matches.

        Args:
            model: The model to evaluate

        Returns:
            Dictionary of metrics
        """
        # This is a placeholder implementation
        # In a real implementation, this would extract data from the model's data loaders
        # and call evaluate_with_integrated_dataset

        # For now, return empty metrics
        return {
            f"precision@{self.top_k}": 0.0,
            f"recall@{self.top_k}": 0.0,
            f"map@{self.top_k}": 0.0,
            f"ndcg@{self.top_k}": 0.0,
            "detailed_results": {},
        }


class GTE:
    def __init__(self, model_name_or_path: str = "thenlper/gte-large", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

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

    def __call__(self, input_texts: list[str], **kwargs):
        with torch.no_grad():
            model_inputs = self.tokenizer(
                input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            outputs = self.model(**model_inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, model_inputs["attention_mask"])
        return embeddings.cpu()


def main() -> None:
    """
    Command-line interface for evaluating REIGN models.

    Usage:
        python -m reign.eval --model_path MODEL --data_path DATA [--batch_size N] [--top_k K] [--enable_cache]
    """
    parser = argparse.ArgumentParser(
        description="Evaluate REIGN embedding models with various metrics."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model or model checkpoint.",
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        required=True,
        help="Path to the evaluation data (JSONL or dataset).",
    )
    parser.add_argument(
        "--synthetic_data_path",
        type=str,
        required=True,
        help="Path to the synthetic data (JSONL or dataset).",
    )
    parser.add_argument(
        "--original_data_split",
        type=str,
        default="test",
        help="Split of the dataset provided (JSONL or dataset).",
    )
    parser.add_argument(
        "--synthetic_data_split",
        type=str,
        default="test",
        help="Split of the dataset provided (JSONL or dataset).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k value for metrics calculation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the evaluation results as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the evaluation on.",
    )
    parser.add_argument(
        "--no_detailed_results",
        action="store_true",
        help="Whether to return detailed results.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    # Caching parameters
    parser.add_argument(
        "--enable_cache",
        action="store_true",
        help="Enable caching of embeddings to speed up evaluation",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=(Path.home() / ".reign_cache").as_posix(),
        help="Root directory for cache storage (default: ~/.reign_cache)",
    )
    parser.add_argument(
        "--force_cache_refresh",
        action="store_true",
        help="Force refresh of cached embeddings even if they exist",
    )
    parser.add_argument(
        "--gn_model",
        type=str,
        default="thenlper/gte-small",
        help="Model name or path for ReignFeatureExtractor (default: thenlper/gte-small)",
    )
    parser.add_argument(
        "--gn-chunk-size",
        "--gn_chunk_size",
        dest="gn_chunk_size",
        type=int,
        default=512,
        help="Number of tokens per chunk fed to the Guidance Network (default: 512).",
    )
    parser.add_argument(
        "--gn-stride",
        "--gn_stride",
        dest="gn_stride",
        type=int,
        default=384,
        help=(
            "Stride between successive chunks fed to the Guidance Network (default: 384, "
            "i.e. 25%% overlap at gn_chunk_size=512). Set equal to --gn-chunk-size for "
            "legacy non-overlapping chunking."
        ),
    )

    args = parser.parse_args()

    # Initialize feature extractor if using cached embeddings
    feature_extractor = None
    if args.enable_cache:
        from reign.feature_extractor import ReignFeatureExtractor

        logger.info("Initializing ReignFeatureExtractor for caching...")
        feature_extractor = ReignFeatureExtractor(
            batch_size=args.batch_size,
            device=args.device,
            model_name_or_path=args.gn_model,
            chunk_size=args.gn_chunk_size,
            stride=args.gn_stride,
            cache_root=args.cache_root,
            enable_cache=args.enable_cache,
        )

        # Handle cache refresh
        if args.force_cache_refresh:
            logger.info("Forcing cache refresh...")
            feature_extractor.cache.clear_cache(args.gn_model)

    # Load model based on whether cached embeddings are enabled
    print(f"Loading model from {args.model_path} ...")
    if args.enable_cache:
        # When using cached embeddings, we need a REIGN model that supports inputs_embeds
        from reign.modeling import ReignModel

        try:
            model = ReignModel.from_pretrained(args.model_path)
            if args.device:
                model = model.to(args.device)
            model.eval()
            logger.info(f"Loaded REIGN model for cached embeddings evaluation")
        except Exception as e:
            logger.error(f"Failed to load REIGN model from {args.model_path}: {e}")
            logger.error(
                "When using --enable_cache, please provide a path to a trained REIGN model, not a generic transformer model."
            )
            raise
    else:
        # For regular evaluation, use GTE model
        model = GTE(model_name_or_path=args.model_path, device=args.device)

    _, data_loader = create_data_loaders(
        original_dataset_name=args.original_data_path,
        synthetic_dataset_name=args.synthetic_data_path,
        train_split=args.original_data_split,
        eval_split=args.synthetic_data_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        collate_fn=None,  # without collate
        use_cached_dataset=args.enable_cache,
        feature_extractor=feature_extractor,
    )

    evaluator = Evaluator(
        batch_size=args.batch_size,
        data_loader=data_loader,
        top_k=args.top_k,
        use_cached_embeddings=args.enable_cache,
    )

    # Run evaluation
    metrics = evaluator.evaluate_with_integrated_dataset(
        model, no_detailed_results=args.no_detailed_results
    )

    # Print results
    print(json.dumps(metrics, indent=2))

    # Optionally save results
    if args.output_path:
        output_path = Path(args.output_path)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to evaluate REIGN model on MTEB benchmark tasks.
Compatible with MTEB version 1.38.34.

For complete task information, see:
- MTEB GitHub: https://github.com/embeddings-benchmark/mteb
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- MTEB Paper: https://arxiv.org/abs/2210.07316
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mteb import MTEB
from tqdm import tqdm

# Try to import MTEB_MAIN_EN, fallback to manual task list
try:
    from mteb.benchmarks import MTEB_MAIN_EN

    MTEB_MAIN_TASKS = MTEB_MAIN_EN.tasks
except ImportError:
    # Fallback task list for MTEB 1.38.34
    MTEB_MAIN_TASKS = [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "ArguAna",
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "AskUbuntuDupQuestions",
        "BIOSSES",
        "Banking77Classification",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "ClimateFEVER",
        "DBPedia",
        "EmotionClassification",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "ImdbClassification",
        "MSMARCO",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "MindSmallReranking",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "RedditClustering",
        "RedditClusteringP2P",
        "SCIDOCS",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STS22",
        "STSBenchmark",
        "SciDocsRR",
        "SciFact",
        "SprintDuplicateQuestions",
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "StackOverflowDupQuestions",
        "SummEval",
        "TRECCOVID",
        "Touche2020",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
        "TwentyNewsgroupsClustering",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ]

from reign.feature_extractor import ReignFeatureExtractor
from reign.modeling import ReignModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# MTEB Task Categories and Lists
MTEB_TASKS_BY_CATEGORY = {
    "Classification": [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "BiorxivClusteringP2P",
        "BiorxivClusteringS2S",
        "MedrxivClusteringP2P",
        "MedrxivClusteringS2S",
        "RedditClustering",
        "RedditClusteringP2P",
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
    ],
    "PairClassification": [
        "AskUbuntuDupQuestions",
        "SprintDuplicateQuestions",
        "StackOverflowDupQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "Reranking": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR"],
    "Retrieval": [
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "DBPedia",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "TRECCOVID",
        "Touche2020",
    ],
    "STS": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STS22",
        "STSBenchmark",
    ],
    "Summarization": ["SummEval"],
}

# Popular task subsets for quick evaluation
POPULAR_TASK_SUBSETS = {
    "retrieval_small": ["ArguAna", "FiQA2018", "NFCorpus", "SCIDOCS", "SciFact"],
    "classification_small": ["Banking77Classification", "EmotionClassification", "ImdbClassification"],
    "clustering_small": ["ArxivClusteringP2P", "RedditClustering", "TwentyNewsgroupsClustering"],
    "sts_small": ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark"],
}


class ReignEmbedder:
    """
    REIGN model wrapper for MTEB evaluation compatible with MTEB 1.38.34.
    Implements the interface expected by MTEB.
    """

    def __init__(
        self,
        model_name_or_path: str,
        gn_model: str,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        enable_cache: bool = False,
        cache_root: Optional[str] = None,
    ):
        """
        Initialize the REIGN model for embedding generation.

        Args:
            model_name_or_path: Path to the REIGN model or model name
            gn_model: Model name or path for the guidance network (feature extractor).
            batch_size: Batch size for inference
            device: Device to use (cuda or cpu)
            max_seq_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings to unit length
            enable_cache: Whether to enable caching of embeddings
            cache_root: Root directory for cache storage
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.model_name_or_path = Path(model_name_or_path)

        # Load REIGN model
        logger.info(f"Loading REIGN model from: {model_name_or_path}")
        try:
            self.model = self._load_model(self.model_name_or_path)
        except OSError as e:
            self.model = self._load_model(self.model_name_or_path / "best")
        except Exception as e:
            logger.error(f"Failed to load model from {model_name_or_path}: {e}")
            raise e
        self.model.eval()

        # Initialize feature extractor
        logger.info(f"Initializing ReignFeatureExtractor with model: {gn_model}")
        self.feature_extractor = ReignFeatureExtractor(
            batch_size=batch_size,
            device=self.device,
            model_name_or_path=gn_model,
            chunk_size=max_seq_length,
            enable_cache=enable_cache,
            cache_root=cache_root,
        )

        # Model metadata for MTEB
        self.name = f"REIGN-{Path(model_name_or_path).name}"

    def _load_model(self, model_path: str) -> ReignModel:
        """Load REIGN model from checkpoint or pretrained path."""
        model_path = Path(model_path)
        logger.info(f"Loading model using from_pretrained: {model_path}")
        model = ReignModel.from_pretrained(model_path).to(self.device)
        logger.info(f"Successfully loaded model using from_pretrained")
        return model

    def encode(
        self,
        sentences: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding (overrides self.batch_size if provided)
            show_progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments (for MTEB compatibility)

        Returns:
            Tensor of embeddings
        """
        if batch_size is None:
            batch_size = self.batch_size

        if not sentences:
            # Return empty tensor with correct dimensions
            return torch.zeros((0, self.model.config.hidden_size)).numpy()

        # Process in batches
        all_embeddings = []

        # Create batches
        for i in tqdm(
            range(0, len(sentences), batch_size),
            desc="Generating embeddings",
            disable=not show_progress_bar,
        ):
            batch_sentences = sentences[i : i + batch_size]

            # Generate embeddings using feature extractor
            with torch.no_grad():
                # Use feature extractor to get embeddings
                model_inputs = self.feature_extractor(
                    batch_sentences,
                    dataset_identifier=f"mteb_batch_{i}",
                    use_cache=False,  # Disable caching for MTEB evaluation
                )

                # Pass through REIGN model
                model_outputs = self.model(
                    inputs_embeds=model_inputs.inputs_embeds, attention_mask=model_inputs.attention_mask
                )

                # Get pooled embeddings
                embeddings = model_outputs.pooler_output

                # Normalize if required
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Move to CPU
                all_embeddings.append(embeddings.cpu())

        # Concatenate all embeddings
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0).numpy()
        else:
            # Return empty tensor with correct dimensions if no sentences
            return torch.zeros((0, self.model.config.hidden_size)).numpy()


def run_mteb_evaluation(
    model_path: str,
    gn_model: str,
    output_dir: str,
    task_types: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    max_seq_length: int = 512,
    normalize_embeddings: bool = True,
    enable_cache: bool = False,
    cache_root: Optional[str] = None,
    eval_splits: Optional[List[str]] = None,
    verbosity: int = 2,
) -> Dict:
    """
    Run MTEB evaluation on specified tasks.

    Args:
        model_path: Path to the REIGN model
        gn_model: Model name or path for the guidance network (feature extractor)
        output_dir: Directory to save evaluation results
        task_types: List of task types to evaluate
        task_names: List of specific task names to evaluate
        batch_size: Batch size for inference
        device: Device to use (cuda or cpu)
        max_seq_length: Maximum sequence length
        normalize_embeddings: Whether to normalize embeddings to unit length
        enable_cache: Whether to enable caching of embeddings
        cache_root: Root directory for cache storage
        eval_splits: List of dataset splits to evaluate on
        verbosity: Verbosity level for MTEB logging

    Returns:
        Dictionary with evaluation results
    """
    # Initialize model
    logger.info("Initializing REIGN model for MTEB evaluation...")
    model = ReignEmbedder(
        model_name_or_path=model_path,
        gn_model=gn_model,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
        normalize_embeddings=normalize_embeddings,
        enable_cache=enable_cache,
        cache_root=cache_root,
    )

    # Determine tasks to evaluate
    if task_names:
        tasks_to_run = task_names
    elif task_types:
        tasks_to_run = []
        for task_type in task_types:
            if task_type in MTEB_TASKS_BY_CATEGORY:
                tasks_to_run.extend(MTEB_TASKS_BY_CATEGORY[task_type])
            else:
                logger.warning(f"Unknown task type: {task_type}")
    else:
        # Default to retrieval tasks
        tasks_to_run = MTEB_TASKS_BY_CATEGORY["Retrieval"]

    # Remove duplicates and validate tasks
    tasks_to_run = list(set(tasks_to_run))
    valid_tasks = [task for task in tasks_to_run if task in MTEB_MAIN_TASKS]
    invalid_tasks = [task for task in tasks_to_run if task not in MTEB_MAIN_TASKS]

    if invalid_tasks:
        logger.warning(f"Invalid tasks (will be skipped): {invalid_tasks}")

    if not valid_tasks:
        logger.error("No valid tasks to evaluate!")
        return {}

    logger.info(f"Evaluating on {len(valid_tasks)} tasks: {valid_tasks}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MTEB
    try:
        mteb = MTEB(tasks=valid_tasks, task_langs=["en"])
    except Exception as e:
        logger.error(f"Failed to initialize MTEB: {e}")
        # Fallback: try without task_langs parameter
        mteb = MTEB(tasks=valid_tasks)

    # Set evaluation splits
    if eval_splits is None:
        eval_splits = ["test"]

    # Run evaluation
    logger.info(f"Starting MTEB evaluation on {model.name}")
    start_time = time.time()

    try:
        results = mteb.run(
            model, output_folder=output_dir, eval_splits=eval_splits, verbosity=verbosity
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {}

    elapsed_time = time.time() - start_time

    # Add metadata to results
    """ results["metadata"] = {
        "model_name": model.name,
        "evaluation_time": elapsed_time,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "normalize_embeddings": normalize_embeddings,
        "device": device,
        "gn_model": gn_model,
        "enable_cache": enable_cache
    } """

    # Save consolidated results
    results_path = os.path.join(output_dir, "consolidated_results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, default=str, indent=2)
        logger.info(f"Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")

    return results


def print_available_tasks():
    """Print information about available MTEB tasks."""
    print("\n" + "=" * 60)
    print("MTEB TASKS INFORMATION")
    print("=" * 60)

    print(f"\nTotal tasks available: {len(MTEB_MAIN_TASKS)}")
    print(f"Tasks by category:")

    for category, tasks in MTEB_TASKS_BY_CATEGORY.items():
        print(f"\n{category} ({len(tasks)} tasks):")
        for task in tasks:
            print(f"  - {task}")

    print(f"\nPopular task subsets:")
    for subset_name, tasks in POPULAR_TASK_SUBSETS.items():
        print(f"\n{subset_name} ({len(tasks)} tasks):")
        for task in tasks:
            print(f"  - {task}")

    print(f"\nFor complete task information, see:")
    print(f"  - MTEB GitHub: https://github.com/embeddings-benchmark/mteb")
    print(f"  - MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard")
    print(f"  - MTEB Paper: https://arxiv.org/abs/2210.07316")
    print("=" * 60)


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate REIGN model on MTEB benchmark (v1.38.34)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on all retrieval tasks
  python scripts/evaluate_mteb.py --model-path /path/to/model --gn-model thenlper/gte-small --task-types Retrieval
  
  # Evaluate on specific tasks
  python scripts/evaluate_mteb.py --model-path /path/to/model --gn-model thenlper/gte-small --task-names ArguAna FiQA2018 NFCorpus
  
  # Evaluate on popular subset
  python scripts/evaluate_mteb.py --model-path /path/to/model --gn-model thenlper/gte-small --task-names retrieval_small
  
  # Show available tasks
  python scripts/evaluate_mteb.py --show-tasks
        """,
    )

    # Show tasks option
    parser.add_argument("--show-tasks", action="store_true", help="Show available MTEB tasks and exit")

    # Model parameters
    parser.add_argument("--model-path", type=str, default=None, help="Path to the REIGN model")
    parser.add_argument(
        "--gn-model",
        type=str,
        required=False,
        help="Model name or path for the guidance network (feature extractor)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--normalize-embeddings", action="store_true", help="Normalize embeddings to unit length"
    )

    # Caching parameters
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable caching of embeddings to speed up evaluation",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Root directory for cache storage (default: ~/.reign_cache)",
    )

    # Task selection
    parser.add_argument(
        "--task-types",
        type=str,
        nargs="+",
        default=None,
        choices=list(MTEB_TASKS_BY_CATEGORY.keys()),
        help="Task types to evaluate",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help="Specific task names to evaluate (overrides task-types). Use 'retrieval_small', 'classification_small', etc. for popular subsets",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-splits",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to evaluate on (default: test)",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Verbosity level for MTEB logging (0: silent, 1: minimal, 2: detailed)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default="./mteb_results", help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Show tasks if requested
    if args.show_tasks:
        print_available_tasks()
        return

    # Validate required arguments
    if args.gn_model is None:
        parser.error("--gn-model is required when not using --show-tasks")

    if args.model_path is None:
        from reign import MODEL_DIR

        args.model_path = str(MODEL_DIR / "test_save")
        logger.info(f"Using default model path: {args.model_path}")

    # Set default cache root if not provided
    if args.cache_root is None:
        args.cache_root = str(Path.home() / ".reign_cache")

    # Handle popular task subsets
    if args.task_names:
        expanded_tasks = []
        for task_name in args.task_names:
            if task_name in POPULAR_TASK_SUBSETS:
                expanded_tasks.extend(POPULAR_TASK_SUBSETS[task_name])
                logger.info(f"Expanding '{task_name}' to: {POPULAR_TASK_SUBSETS[task_name]}")
            else:
                expanded_tasks.append(task_name)
        args.task_names = expanded_tasks

    # Run evaluation
    results = run_mteb_evaluation(
        model_path=args.model_path,
        gn_model=args.gn_model,
        output_dir=args.output_dir,
        task_types=args.task_types,
        task_names=args.task_names,
        batch_size=args.batch_size,
        device=args.device,
        max_seq_length=args.max_seq_length,
        normalize_embeddings=args.normalize_embeddings,
        enable_cache=args.enable_cache,
        cache_root=args.cache_root,
        eval_splits=args.eval_splits,
        verbosity=args.verbosity,
    )

    if results:
        logger.info("Evaluation completed successfully!")
    else:
        logger.error("Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

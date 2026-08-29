# main.py
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import datasets
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from lightning import Callback, seed_everything
from torch import optim

from reign import MODEL_DIR, ReignModel
from reign.configuration import (
    ReignBaseConfig,
    ReignBaseL3Config,
    ReignBaseL4Config,
    ReignConfig,
    ReignLargeConfig,
    ReignLargeL4Config,
    ReignLargeL6Config,
    ReignSmallConfig,
    ReignSmallL2Config,
    ReignSmallL3Config,
    ReignTinyL1Config,
    ReignTinyL3Config,
    ReignXLargeConfig,
    ReignXLargeL4Config,
    ReignXLargeL6Config,
)
from reign.dataset import ReignCachedDataset, ReignDataset, create_data_loaders
from reign.eval import Evaluator
from reign.feature_extractor import ReignFeatureExtractor
from reign.loss import InfoNCELoss, ThreeWayCosineEmbeddingLoss
from reign.modeling import ReignForPreTraining
from reign.utils import CheckpointHandler, get_local_logger, profile_function, profiler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class ReignLitModel(L.LightningModule):
    """
    LightningModule for training the Reign model with contrastive learning.

    This module implements the training logic for the REIGN framework, which focuses
    on large-scale text embedding models using contrastive learning and synthetic
    dataset generation.

    Args:
        batch_size (int): Training batch size.
        lr (float): Learning rate for optimization.
        max_epochs (int): Maximum number of training epochs.
        weight_decay (float): Weight decay for regularization.
        temperature (float): Temperature parameter for InfoNCE loss (default: 1).
        weight_partial (float): Weight for partial relevance in cosine loss (default: 0.5).
        negative_batch_size_multiplier (Optional[int]): Multiplier for negative batch size (default: 2).
        device (Optional[Union[str, int]]): Device to run training on (default: "cuda").
        checkpoint_handler (Optional[CheckpointHandler]): Handler for model checkpointing.
        evaluator (Optional[Evaluator]): Evaluator for model performance assessment.
        loss_function (str): Type of loss function to use ("cosine" or "infonce").
        gn_model (str): Name or path of the guidance network model (default: "thenlper/gte-small").
        chunk_size (int): Number of tokens for chunks of a text in the feature extractor (default: 512).
        from_checkpoint (Optional[str]): Path to load model from checkpoint.
        use_cached_embeddings (bool): Whether to use cached embeddings (default: False).
        model_config (str): Model configuration size ("small", "base", or "large", default: "base").
        position_embedding_type (str): Chunk-position signal: "none" (default, the
            published permutation-equivariant design), "absolute", or "sinusoidal".
        metric_to_monitor (str): Metric to monitor for checkpointing (default: "map@10").
        lower_is_better (bool): Whether lower metric values are better for checkpointing (default: False).
    """

    def __init__(
        self,
        batch_size: int,
        lr: float,
        max_epochs: int,
        weight_decay: float,
        temperature: float = 1,
        weight_partial: float = 0.5,
        negative_batch_size_multiplier: Optional[int] = 2,
        device: Optional[str | int] = "cuda",
        checkpoint_handler: CheckpointHandler = None,
        evaluator: Evaluator = None,
        loss_function: str = "cosine",
        gn_model: str = "thenlper/gte-small",
        gn_device: Optional[str | int] = "cuda",
        gn_batch_size: Optional[int] = 12,
        chunk_size: int = 512,
        stride: int = 384,
        from_checkpoint: Optional[str] = None,
        use_cached_embeddings: bool = False,
        model_config: str = "base",
        position_embedding_type: str = "none",
        metric_to_monitor: str = "map@10",
        lower_is_better: bool = False,
    ):
        nbsm = (
            min(negative_batch_size_multiplier, batch_size - 1)
            if negative_batch_size_multiplier > 0
            else batch_size - 1
        )
        self.hparams.negative_batch_size_multiplier = nbsm
        if loss_function == "infonce":
            self.hparams.effective_batch_size = (nbsm + 1) * batch_size
            logger.info(
                f"Effective batch size: {(nbsm + 1) * batch_size} without distractors (partial matches), i.e. infonce loss."
            )
        else:
            self.hparams.effective_batch_size = (3 * nbsm + 1) * batch_size
            logger.info(
                f"Effective batch size: {(3*nbsm + 1) * batch_size} with distractors, i.e. cosine loss."
            )
        super().__init__()
        logger.info(
            f"Initializing ReignLitModel with batch_size={batch_size}, lr={lr}, device={device}, gn_model={gn_model}"
        )
        # Save only the primitive hyperparameters, not complex objects
        self.save_hyperparameters(ignore=["checkpoint_handler", "evaluator"])

        logger.info("Initializing ReignFeatureExtractor...")
        self.feature_extractor = ReignFeatureExtractor(
            gn_batch_size,
            device=gn_device,
            model_name_or_path=gn_model,
            chunk_size=chunk_size,
            stride=stride,
        )

        logger.info(f"Initializing REIGN model with config: {model_config}")
        if from_checkpoint is not None:
            # `device` is not a from_pretrained/ReignModel.__init__ kwarg — HF would
            # forward it into __init__ and raise. Load, then move (mirrors else-branch).
            self.model = ReignModel.from_pretrained(from_checkpoint).to(device)
        else:
            # Select appropriate config based on model_config parameter
            config_map = {
                # New layer-based variants
                "tiny-l1": ReignTinyL1Config,
                "tiny-l3": ReignTinyL3Config,
                "small-l2": ReignSmallL2Config,
                "small-l3": ReignSmallL3Config,
                "base-l3": ReignBaseL3Config,
                "base-l4": ReignBaseL4Config,
                "large-l4": ReignLargeL4Config,
                "large-l6": ReignLargeL6Config,
                "xlarge-l4": ReignXLargeL4Config,
                "xlarge-l6": ReignXLargeL6Config,
            }

            if model_config not in config_map:
                raise ValueError(
                    f"Invalid model_config: {model_config}. Choose from: {list(config_map.keys())}"
                )

            config_class = config_map[model_config]
            config = config_class(
                gn_projection_dim=self.feature_extractor.model.config.hidden_size,
                gn_chunk_size=chunk_size,
                gn_stride=stride,
                position_embedding_type=position_embedding_type,
            )

            self.model = ReignModel(config).to(device)
        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters"
        )

        # Initialize the loss function based on the specified type
        if loss_function == "cosine":
            logger.info(
                f"Setting up ThreeWayCosineEmbeddingLoss with weight_partial={weight_partial}..."
            )
            self.loss = ThreeWayCosineEmbeddingLoss(weight_partial=weight_partial)
        elif loss_function == "infonce":
            logger.info(
                f"Setting up InfoNCELoss with temperature={temperature}, "
                f"partial_weight={weight_partial}..."
            )
            self.loss = InfoNCELoss(temperature=temperature, partial_weight=weight_partial)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # ThreeWayCosineEmbeddingLoss always uses distractors. InfoNCELoss
        # uses them only when graded mode is on (``partial_weight > 0``).
        # Cache this on self so the optimised forward path in
        # ``_process_cached_batch`` can skip the distractor GN forward when
        # they would be ignored anyway.
        self._loss_uses_distractors = (
            not isinstance(self.loss, InfoNCELoss) or getattr(self.loss, "partial_weight", 0.0) > 0
        )

        self.checkpoint_handler = checkpoint_handler
        self._current_eval_metric = checkpoint_handler.metric_value if checkpoint_handler else 0.0
        self.evaluator = evaluator
        self.use_cached_embeddings = use_cached_embeddings
        logger.info("ReignLitModel initialization complete")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        if self.use_cached_embeddings:
            # When using cached embeddings, x should be embeddings
            if isinstance(x, tuple) and len(x) == 3:
                # Handle cached embeddings format: (pooler_output,)
                embeddings = x
            elif isinstance(x, torch.Tensor):
                # For single embedding tensor
                embeddings = torch.stack([x]) if x.dim() == 1 else x
            else:
                # Fallback to regular feature extraction if x is text
                features = self.feature_extractor(x)
                embeddings = self.model(**features)
        else:
            # Regular feature extraction
            features = self.feature_extractor(x)
            embeddings = self.model(**features)

        return embeddings

    def configure_optimizers(self):
        logger.info(
            f"Configuring optimizer with lr={self.hparams.lr}, weight_decay={self.hparams.weight_decay}"
        )
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={self.hparams.max_epochs}")
        return [optimizer], [lr_scheduler]

    def get_combined_batch(
        self,
        original_embeddings: torch.Tensor,
        synthetic_embeddings: torch.Tensor,
        distractor_embeddings: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Create combined batch with positive, partial, and negative pairs for contrastive learning.

        Args:
            original_embeddings: Embeddings of original articles
            synthetic_embeddings: Embeddings of paired synthetic articles
            distractor_embeddings: List of embeddings for distractor articles (optional)

        Returns:
            Tuple containing combined instances, paired instances, targets, and metrics
        """
        batch_size = original_embeddings.shape[0]
        device = original_embeddings.device

        # Create positive pairs (original and synthetic pairs)
        positive_instances = original_embeddings
        positive_paired_instances = synthetic_embeddings
        positive_target = torch.ones(batch_size, device=device)

        # If we have distractor embeddings and we're not using InfoNCE loss, use them as partial matches
        partial_instances = []
        partial_paired_instances = []
        partial_targets = []

        # Emit partial pairs (target=0) whenever distractors are available.
        # ThreeWayCosineEmbeddingLoss uses them with weight ``weight_partial``;
        # InfoNCELoss with ``partial_weight > 0`` treats them as graded soft
        # positives in the numerator; InfoNCELoss with ``partial_weight = 0``
        # (the GoodWiki default) silently ignores them. So the same combined
        # batch works for all three loss configurations.
        if distractor_embeddings is not None:
            # The collate flattens K distractors per sample into a single tensor;
            # tile originals by the actual ratio (typically K=2) and emit a target
            # vector of matching length. (The previous code hardcoded both pieces
            # and disagreed on the partial-target length, which crashed the loss
            # under cosine + cached batching.)
            n_distractors = distractor_embeddings.shape[0]
            if n_distractors == 0 or n_distractors % batch_size != 0:
                raise ValueError(
                    f"distractor count {n_distractors} is not a positive multiple of "
                    f"batch_size {batch_size}; refusing to pair ambiguously"
                )
            k = n_distractors // batch_size
            logger.debug(f"Processing {n_distractors} distractors ({k} per sample)")
            partial_instance_tensor = original_embeddings.repeat_interleave(k, dim=0)
            partial_instances.append(partial_instance_tensor)
            partial_paired_instances.append(distractor_embeddings)
            partial_targets.append(torch.zeros(n_distractors, device=device))

        # Create negative samples by shifting indices of synthetic_embeddings
        nbsm = int(self.hparams.negative_batch_size_multiplier)
        logger.debug(f"Creating {nbsm} negative samples per batch item")
        negative_instances = []
        negative_paired_instances = []
        negative_targets = []

        for i in range(nbsm):
            shifted_embeddings = torch.roll(synthetic_embeddings, shifts=-(i + 1), dims=0)
            # TODO: to be removed (obsolete)
            # Roll the synthetic embeddings to create negative pairs
            # shift 3 idx as (pair, distractor, distractor, negative, ..., negative)
            # shifted_embeddings = torch.roll(synthetic_embeddings, shifts=-(i+1)*3, dims=0)

            negative_instances.append(original_embeddings)
            negative_paired_instances.append(shifted_embeddings)
            negative_targets.append(torch.full((batch_size,), -1, device=device))

        # Combine all types of pairs
        combined_instance = torch.cat(
            [positive_instances]
            + ([inst for inst in partial_instances] if partial_instances else [])
            + [inst for inst in negative_instances],
            dim=0,
        )

        combined_paired_instance = torch.cat(
            [positive_paired_instances]
            + ([inst for inst in partial_paired_instances] if partial_paired_instances else [])
            + [inst for inst in negative_paired_instances],
            dim=0,
        )

        combined_target = torch.cat(
            [positive_target]
            + ([tgt for tgt in partial_targets] if partial_targets else [])
            + [tgt for tgt in negative_targets],
            dim=0,
        )

        logger.debug(
            f"Combined batch shape: {combined_instance.shape}, targets shape: {combined_target.shape}"
        )

        # Calculate metrics
        metrics = {
            "avg_pos_cos_sim": F.cosine_similarity(original_embeddings, synthetic_embeddings).mean(),
        }

        if negative_instances:
            all_neg_instances = torch.cat(negative_instances, dim=0)
            all_neg_paired = torch.cat(negative_paired_instances, dim=0)
            metrics["avg_neg_cos_sim"] = F.cosine_similarity(all_neg_instances, all_neg_paired).mean()

        if partial_instances:
            all_partial_instances = torch.cat(partial_instances, dim=0)
            all_partial_paired = torch.cat(partial_paired_instances, dim=0)
            metrics["avg_partial_cos_sim"] = F.cosine_similarity(
                all_partial_instances, all_partial_paired
            ).mean()

        return combined_instance, combined_paired_instance, combined_target, metrics

    @profile_function("train.compute_loss")
    def compute_loss(self, batch, mode="train"):
        logger.debug(f"Computing loss for {mode} batch")

        if self.use_cached_embeddings:
            # Handle cached embeddings with new format
            (
                original_embeddings,
                original_masks,
                synthetic_embeddings,
                synthetic_masks,
                distractor_embeddings,
                distractor_masks,
                _,
            ) = batch

            # Optimize: Batch all embeddings together for single model forward pass when possible
            # Process each embedding group separately to avoid shape mismatch in dim=1
            if self._loss_uses_distractors and distractor_embeddings.numel() > 0:
                # Forward pass for original embeddings
                original_outputs = self.model(
                    inputs_embeds=original_embeddings, attention_mask=original_masks
                ).pooler_output

                # Forward pass for synthetic embeddings
                synthetic_outputs = self.model(
                    inputs_embeds=synthetic_embeddings, attention_mask=synthetic_masks
                ).pooler_output

                # Forward pass for distractor embeddings
                distractor_outputs = self.model(
                    inputs_embeds=distractor_embeddings, attention_mask=distractor_masks
                ).pooler_output

                # Assign outputs
                original_embeddings = original_outputs
                synthetic_embeddings = synthetic_outputs
                distractor_embeddings = distractor_outputs

            else:
                # Only process original and synthetic embeddings
                original_outputs = self.model(
                    inputs_embeds=original_embeddings, attention_mask=original_masks
                ).pooler_output

                synthetic_outputs = self.model(
                    inputs_embeds=synthetic_embeddings, attention_mask=synthetic_masks
                ).pooler_output

                original_embeddings = original_outputs
                synthetic_embeddings = synthetic_outputs
                distractor_embeddings = None
        else:
            # Handle regular text inputs
            original_articles, synthetic_articles, distractor_articles_list = batch

            # Encode original and synthetic articles
            logger.debug(f"Encoding {len(original_articles)} original articles")
            original_embeddings = self(original_articles).pooler_output

            logger.debug(f"Encoding {len(synthetic_articles)} synthetic articles")
            synthetic_embeddings = self(synthetic_articles).pooler_output

            batch_distractors = [
                sample_distractors[0]
                for sample_distractors in distractor_articles_list
                if sample_distractors and len(sample_distractors) > 0
            ]

            if self._loss_uses_distractors and batch_distractors:
                logger.debug(f"Encoding {len(batch_distractors)} distractor articles")
                distractor_embeddings = self(batch_distractors).pooler_output
            else:
                distractor_embeddings = None

        # Get combined batch for contrastive learning
        logger.debug("Creating combined batch for contrastive learning")
        combined_instance, combined_paired_instance, combined_target, metrics = self.get_combined_batch(
            original_embeddings, synthetic_embeddings, distractor_embeddings
        )

        # Compute loss
        loss = self.loss(combined_instance, combined_paired_instance, combined_target)
        logger.debug(f"{mode} loss: {loss.item():.4f}")

        # Logging
        self.log(f"{mode}/loss", loss, batch_size=self.hparams.batch_size)
        for metric_name, metric_value in metrics.items():
            self.log(f"{mode}/{metric_name}", metric_value, batch_size=self.hparams.batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        logger.debug(f"Training step {batch_idx}")
        return self.compute_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step {batch_idx}")
        self.compute_loss(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        logger.info("Validation epoch completed, computing metrics...")

        # Print profiling stats every few epochs
        if self.current_epoch > 0 and self.current_epoch % 5 == 0:
            logger.info(f"Performance profiling stats for epoch {self.current_epoch}:")
            profiler.print_stats(min_percentage=0.5)
        if self.evaluator is not None:
            # Create a wrapper function to get embeddings from the model

            # Compute metrics using the evaluator
            logger.info("Computing evaluation metrics...")
            metrics = self.evaluator.evaluate_with_integrated_dataset(self)

            for name, metric in metrics.items():
                if name == "detailed_results":
                    continue
                else:
                    if name == self.hparams.metric_to_monitor:
                        self._current_eval_metric = metric
                        logger.info(f"Validation {self.hparams.metric_to_monitor}: {metric:.4f}")
                    self.log(f"val/{name}", metric)

        if self.checkpoint_handler is not None:
            logger.info("Saving checkpoint...")
            model_dict = {
                "optimizer": self.optimizers().state_dict(),
                "lr_scheduler": self.lr_schedulers().state_dict(),
                "epoch": self.current_epoch,
                "hparams_initial": self.hparams_initial,
                "hparams": self.hparams,
            }
            log_stats = {
                "eval_metric": float(self._current_eval_metric),
                "epoch": self.current_epoch,
            }
            self.checkpoint_handler.save(
                self.model, model_dict, log_stats, float(self._current_eval_metric), self.current_epoch
            )
            logger.info(f"Checkpoint saved. Current best metric: {self._current_eval_metric:.4f}")

    def save_model(self, output_dir, **kwargs):
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Train REIGN model")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=12, help="Batch size for evaluation")
    parser.add_argument(
        "--negative-batch-size-multiplier",
        type=int,
        default=1,
        help="Multiplier for negative samples in a batch",
    )
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument(
        "--data-loader-num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading (use 0 to avoid CUDA multiprocessing issues)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--weight-partial",
        type=float,
        default=0.5,
        help="Weight for partial matches in the loss function",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Mixed precision training parameters
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16-mixed", "bf16-mixed", "32"],
        help="Training precision: '16-mixed' for fp16 mixed precision, 'bf16-mixed' for bfloat16 mixed precision, '32' for full precision (default: 32)",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Gradient clipping value. Useful with mixed precision training (default: None)",
    )
    parser.add_argument(
        "--gradient-clip-algorithm",
        type=str,
        default="norm",
        choices=["norm", "value"],
        help="Gradient clipping algorithm: 'norm' for gradient norm clipping, 'value' for gradient value clipping (default: norm)",
    )

    # New arguments for loss function
    parser.add_argument(
        "--loss-function",
        type=str,
        default="cosine",
        choices=["cosine", "infonce"],
        help="Loss function to use: 'cosine' for ThreeWayCosineEmbeddingLoss or 'infonce' for InfoNCE",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss"
    )

    # New arguments for validation
    parser.add_argument(
        "--check-val-every-n-epoch", type=int, default=4, help="Run validation every N epochs"
    )
    parser.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=0,
        help="Number of validation steps to run before starting training",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="devrim/goodwiki_long_synthetic_ir",
        help="HF dataset id in BEIR/MTEB layout (corpus/queries/default configs)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="qrels split used for training (train | val | test)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="val",
        help="qrels split used for evaluation during training (train | val | test)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to use (for debugging)"
    )
    parser.add_argument(
        "--n-distractors-per-sample",
        type=int,
        default=None,
        help=(
            "Cap the per-instance distractor count to K (with deterministic "
            "per-instance subsampling). Required for IR datasets where queries "
            "have variable distractor counts; the training collate otherwise "
            "emits a non-multiple of batch_size and crashes. Leave unset for "
            "GoodWiki-Long-Synthetic (uniformly K=2)."
        ),
    )
    parser.add_argument(
        "--gn-chunk-size",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=512,
        help=(
            "Number of tokens per chunk fed to the Guidance Network (default: 512). "
            "``--chunk-size`` is accepted as a legacy alias."
        ),
    )
    parser.add_argument(
        "--gn-stride",
        dest="stride",
        type=int,
        default=384,
        help=(
            "Stride between successive chunks fed to the Guidance Network (default: 384, "
            "i.e. 25%% overlap at chunk_size=512). Set equal to --gn-chunk-size for "
            "legacy non-overlapping chunking."
        ),
    )

    # Caching parameters
    parser.add_argument(
        "--enable-cache", action="store_true", help="Enable caching of embeddings to speed up training"
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=(Path.home() / ".reign_cache").as_posix(),
        help="Root directory for cache storage (default: ~/.reign_cache)",
    )
    parser.add_argument(
        "--force-cache-refresh",
        action="store_true",
        help="Force refresh of cached embeddings even if they exist",
    )
    parser.add_argument(
        "--position-embedding-type",
        choices=["none", "absolute", "sinusoidal"],
        default="none",
        help=(
            "Chunk-position signal fed to the REIGN encoder. 'none' (default) is the "
            "published design: a permutation-equivariant set function over chunk "
            "embeddings. 'absolute' learns a per-position embedding; 'sinusoidal' uses "
            "a fixed table. Drives the positional-encoding ablation."
        ),
    )
    parser.add_argument("--cache-info", action="store_true", help="Show cache information and exit")

    # Evaluation parameters
    parser.add_argument("--top-k", type=int, default=10, help="Top-k value for evaluation metrics")
    parser.add_argument(
        "--metric-to-monitor",
        type=str,
        default="map@10",
        choices=["loss", "precision@10", "recall@10", "map@10", "ndcg@10"],
        help="Metric to monitor for checkpointing (default: map@10)",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Whether lower metric values are better for checkpointing",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints (default: auto-generated)",
    )
    parser.add_argument("--log-every-n-steps", type=int, default=5, help="Log metrics every N steps")
    parser.add_argument(
        "--logger",
        type=str,
        default="csv",
        choices=["csv", "tensorboard", "none"],
        help=(
            "Local Lightning logger for train/val metrics (default: csv, written to "
            "logs/<run-name>/). No external tracking service is contacted."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable performance profiling")

    # Device parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")

    # New argument for ReignFeatureExtractor
    parser.add_argument(
        "--gn-model",
        type=str,
        default="thenlper/gte-small",
        help="Model name or path for ReignFeatureExtractor (default: thenlper/gte-small)",
    )

    # New argument for continued pretraining from checkpoint
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to a Lightning checkpoint to resume training from (for continued pretraining)",
    )

    # Model configuration parameter
    parser.add_argument(
        "--model-config",
        type=str,
        default="base-l3",
        choices=[
            "tiny-l1",
            "tiny-l3",
            "small-l2",
            "small-l3",
            "base-l3",
            "base-l4",
            "large-l4",
            "large-l6",
            "xlarge-l4",
            "xlarge-l6",
        ],
        help=(
            "REIGN encoder size. Options: "
            "tiny-l1 (192d, 1 layer), tiny-l3 (192d, 3 layers), "
            "small-l2 (384d, 2 layers), small-l3 (384d, 3 layers), "
            "base-l3 (768d, 3 layers), base-l4 (768d, 4 layers), "
            "large-l4 (1024d, 4 layers), large-l6 (1024d, 6 layers), "
            "xlarge-l4 (1024d, 4 layers, 5x FFN), xlarge-l6 (1024d, 6 layers, 5x FFN). "
            "The released checkpoints use base-l3 for the main results and "
            "tiny-l1/small-l2/large-l4 for the size ablation."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Enable/disable profiling
    if args.enable_profiling:
        profiler.enable()
        logger.info("Performance profiling enabled")
    else:
        profiler.disable()
        logger.info("Performance profiling disabled")

    logger.info(f"Starting REIGN training with arguments: {args}")

    # Initialize feature extractor early to handle cache operations
    logger.info("Initializing ReignFeatureExtractor for caching...")
    feature_extractor = ReignFeatureExtractor(
        batch_size=args.batch_size,
        device=args.device,
        model_name_or_path=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        cache_root=args.cache_root,
        enable_cache=args.enable_cache,
    )

    # Handle cache info request
    if args.cache_info:
        cache_info = feature_extractor.cache.get_cache_info()
        logger.info("Cache Information:")
        logger.info(f"  Cache root: {cache_info.get('cache_root', 'N/A')}")
        logger.info(f"  Total files: {cache_info.get('total_files', 0)}")
        logger.info(f"  Total size: {cache_info.get('total_size_mb', 0):.2f} MB")

        for model_name, model_info in cache_info.get("models", {}).items():
            logger.info(f"  Model {model_name}:")
            logger.info(f"    Files: {model_info['files']}")
            logger.info(f"    Size: {model_info['size_mb']:.2f} MB")
            logger.info(f"    Embeddings: {model_info['total_embeddings']}")
        return

    # Handle cache refresh
    if args.force_cache_refresh:
        logger.info("Forcing cache refresh...")
        feature_extractor.cache.clear_cache(args.gn_model)

    # Set up output directory
    if args.output_dir is None:
        cache_suffix = "_cached" if args.enable_cache else ""
        precision_suffix = f"_{args.precision}" if args.precision != "32" else ""
        output_dir = (
            MODEL_DIR
            / f"reign-{args.model_config}_lr-{args.lr}_bs-{args.batch_size}_nbsm-{args.negative_batch_size_multiplier}_wp-{args.weight_partial}_chunk-{args.chunk_size}_epochs-{args.max_epochs}{cache_suffix}{precision_suffix}"
        )
    else:
        output_dir = MODEL_DIR / args.output_dir

    # If resuming from checkpoint, don't try to create the output directory again
    if args.from_checkpoint is None:
        logger.info(f"Output directory: {output_dir}")
        output_dir.mkdir(exist_ok=False, parents=True)
    else:
        logger.info(f"Resuming from checkpoint: {args.from_checkpoint}")
        logger.info(f"Output directory (may already exist): {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)

    # Log mixed precision configuration
    logger.info(f"Training precision: {args.precision}")
    if args.precision in ["16-mixed", "bf16-mixed"]:
        logger.info(
            "Mixed precision training enabled. This can improve training speed and reduce memory usage."
        )
        if args.gradient_clip_val is not None:
            logger.info(
                f"Gradient clipping enabled with value: {args.gradient_clip_val} (algorithm: {args.gradient_clip_algorithm})"
            )
        else:
            logger.info("Consider using gradient clipping with mixed precision training for stability.")

        # Check device compatibility
        if args.device == "cuda":
            if args.precision == "bf16-mixed":
                # Check if CUDA device supports bfloat16
                if torch.cuda.is_available():
                    device_capability = torch.cuda.get_device_capability()
                    if device_capability[0] < 8:  # Ampere (RTX 30xx) and newer support bfloat16
                        logger.warning(
                            f"bfloat16 mixed precision requires CUDA compute capability >= 8.0, "
                            f"but detected {device_capability}. Consider using fp16 mixed precision instead."
                        )
            logger.info("CUDA device detected - mixed precision training should work well.")
        else:
            logger.warning(
                "Mixed precision training is optimized for CUDA devices. CPU training may not see benefits."
            )
    else:
        logger.info("Using full precision (fp32) training.")

    # Disable Tokenizers' parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Disabled tokenizers parallelism")

    # For reproducibility
    logger.info(f"Setting random seed to {args.seed}")
    seed_everything(args.seed)

    # Handle CUDA multiprocessing issue
    effective_num_workers = args.data_loader_num_workers
    force_multiprocessing = os.environ.get("REIGN_FORCE_MULTIPROCESSING", "false").lower() == "true"

    if (
        args.device == "cuda"
        and args.data_loader_num_workers > 0
        and not args.enable_cache
        and not force_multiprocessing
    ):
        logger.warning(
            "CUDA device detected with num_workers > 0. This can cause multiprocessing errors "
            "when using models in DataLoader workers. Setting num_workers=0 for safety."
        )
        logger.warning(
            "To use multiprocessing with CUDA, consider using --enable-cache to pre-compute embeddings."
        )
        logger.warning(
            "To force multiprocessing (at your own risk), set: export REIGN_FORCE_MULTIPROCESSING=true"
        )
        effective_num_workers = 0
    elif force_multiprocessing and args.device == "cuda" and args.data_loader_num_workers > 0:
        logger.warning(
            "REIGN_FORCE_MULTIPROCESSING=true detected. Using multiprocessing with CUDA. "
            "This may cause CUDA context errors!"
        )

    # Create data loaders using the dataset utility function
    logger.info(f"Creating data loaders for {args.dataset}")
    logger.info(f"Train qrels split: {args.train_split}, Eval qrels split: {args.eval_split}")
    logger.info(f"Using cached embeddings: {args.enable_cache}")
    logger.info(f"Effective num_workers: {effective_num_workers}")
    if args.max_samples:
        logger.info(f"Using only {args.max_samples} samples for debugging")

    train_loader, eval_loader = create_data_loaders(
        dataset_name=args.dataset,
        train_split=args.train_split,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=effective_num_workers,
        max_samples=args.max_samples,
        collate_fn="default",
        use_cached_dataset=args.enable_cache,
        feature_extractor=feature_extractor if args.enable_cache else None,
        n_distractors_per_sample=args.n_distractors_per_sample,
    )
    logger.info(
        f"Data loaders created. Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}"
    )

    # Handle metric monitoring logic
    lower_is_better = args.lower_is_better
    if args.metric_to_monitor == "loss":
        lower_is_better = True
        logger.info("Metric to monitor is 'loss', automatically setting lower_is_better=True")

    # Set up evaluator and checkpoint handler
    logger.info("Setting up evaluator and checkpoint handler")
    evaluator = Evaluator(
        batch_size=args.eval_batch_size,
        data_loader=eval_loader,
        top_k=args.top_k,
        use_cached_embeddings=args.enable_cache,
    )
    checkpoint_handler = CheckpointHandler(
        checkpoint_dir=output_dir,
        lower_is_better=lower_is_better,
        resume=args.from_checkpoint is not None,
    )

    # Create model
    logger.info(f"Creating REIGN model with config: {args.model_config}")
    model = ReignLitModel(
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        weight_partial=args.weight_partial,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        negative_batch_size_multiplier=args.negative_batch_size_multiplier,
        checkpoint_handler=checkpoint_handler,
        evaluator=evaluator,
        device=args.device,
        loss_function=args.loss_function,
        gn_model=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        from_checkpoint=args.from_checkpoint,
        use_cached_embeddings=args.enable_cache,
        model_config=args.model_config,
        position_embedding_type=args.position_embedding_type,
        metric_to_monitor=args.metric_to_monitor,
        lower_is_better=lower_is_better,
    )

    # Set up logger
    run_name = Path(output_dir).name if output_dir else "reign"
    train_logger = get_local_logger(
        kind=args.logger,
        save_dir="logs",
        name=run_name,
        hparams=vars(args),
    )
    if train_logger is None:
        logger.info("Metric logging disabled (stdout only)")
    else:
        logger.info("Logging metrics with %s to logs/%s", args.logger, run_name)

    # Create trainer and start training
    logger.info(
        f"Creating Lightning Trainer with max_epochs={args.max_epochs}, precision={args.precision}"
    )
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "log_every_n_steps": args.log_every_n_steps,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "logger": train_logger,
        "precision": args.precision,
    }

    # Add gradient clipping if specified
    if args.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = args.gradient_clip_val
        trainer_kwargs["gradient_clip_algorithm"] = args.gradient_clip_algorithm
        logger.info(
            f"Gradient clipping configured: {args.gradient_clip_algorithm} with value {args.gradient_clip_val}"
        )

    trainer = L.Trainer(**trainer_kwargs)

    logger.info("Starting training")
    trainer.fit(model, train_loader, eval_loader)
    logger.info("Training completed")

    # Print final profiling summary
    if args.enable_profiling:
        logger.info("Final Performance Profiling Summary:")
        profiler.print_stats(min_percentage=0.1)


if __name__ == "__main__":
    main()

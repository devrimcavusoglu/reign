# coding=utf-8
"""
Utility functions for the REIGN training pipeline.
"""

import functools
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict

import torch

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Simple performance profiler to identify bottlenecks in training.
    """

    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.lock = threading.Lock()
        self.enabled = True

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def clear(self):
        """Clear all profiling data."""
        with self.lock:
            self.timings.clear()
            self.call_counts.clear()

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks."""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time

            with self.lock:
                self.timings[name].append(elapsed)
                self.call_counts[name] += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        with self.lock:
            stats = {}
            for name, times in self.timings.items():
                stats[name] = {
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "call_count": self.call_counts[name],
                    "percentage": 0.0,  # Will be calculated later
                }

            # Calculate percentages
            total_time = sum(stat["total_time"] for stat in stats.values())
            if total_time > 0:
                for stat in stats.values():
                    stat["percentage"] = (stat["total_time"] / total_time) * 100

            return stats

    def print_stats(self, min_percentage: float = 1.0):
        """Print profiling statistics."""
        stats = self.get_stats()

        if not stats:
            logger.info("No profiling data available")
            return

        logger.info("Performance Profiling Results:")
        logger.info("-" * 80)
        logger.info(f"{'Function':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'%':<6}")
        logger.info("-" * 80)

        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total_time"], reverse=True)

        for name, stat in sorted_stats:
            if stat["percentage"] >= min_percentage:
                logger.info(
                    f"{name:<30} {stat['call_count']:<8} {stat['total_time']:<10.4f} "
                    f"{stat['avg_time']:<10.4f} {stat['percentage']:<6.1f}"
                )


# Global profiler instance
profiler = PerformanceProfiler()


def profile_function(name: str = None):
    """Decorator to profile function execution time."""

    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler.enabled:
                return func(*args, **kwargs)

            with profiler.profile(func_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def profile_block(name: str):
    """Context manager for profiling code blocks."""
    with profiler.profile(name):
        yield


import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from lightning.pytorch.loggers import CSVLogger, Logger


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fd_in:
        return json.load(fd_in)


def get_local_logger(
    kind: str = "csv",
    save_dir: str = "logs",
    name: Optional[str] = None,
    version: Optional[str] = None,
    hparams: Optional[Dict[str, Any]] = None,
) -> Optional[Logger]:
    """Build a Lightning logger that writes metrics to the local filesystem.

    No external experiment-tracking service is involved: everything stays under
    ``save_dir``.

    Args:
        kind: ``"csv"`` (default) writes ``<save_dir>/<name>/<version>/metrics.csv``;
            ``"tensorboard"`` writes TensorBoard event files to the same tree
            (requires the optional ``tensorboard`` package); ``"none"`` disables
            metric logging and returns ``None``.
        save_dir: Root directory for the run tree.
        name: Run-family name (defaults to ``"reign"``).
        version: Sub-directory for this run (Lightning auto-increments if unset).
        hparams: Optional hyperparameter snapshot recorded alongside the metrics.
    """
    kind = (kind or "none").lower()
    if kind == "none":
        return None

    name = name or "reign"
    if kind == "csv":
        train_logger: Logger = CSVLogger(save_dir=save_dir, name=name, version=version)
    elif kind == "tensorboard":
        try:
            from lightning.pytorch.loggers import TensorBoardLogger

            train_logger = TensorBoardLogger(save_dir=save_dir, name=name, version=version)
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "TensorBoard logging requires the optional 'tensorboard' package "
                "(pip install 'reign[tensorboard]'); use --logger csv otherwise."
            ) from exc
    else:
        raise ValueError(f"Unknown logger kind '{kind}' (expected: csv | tensorboard | none)")

    if hparams:
        train_logger.log_hyperparams(dict(hparams))
    return train_logger


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        for i in range(10):
            try:
                torch.save(*args, **kwargs)
                break
            except:
                print("Saving model failed for {} times, will retry".format(i + 1))
                time.sleep(30.0)


class CheckpointHandler:
    """
    Checkpoint manager for saving and loading from a checkpoint of trained models.

    TODO: resume from checkpoint.

    Args:
        checkpoint_dir (str) : directory to save checkpoints to.
        lower_is_better (bool) : whether lower metric is better or not.
    """

    def __init__(self, checkpoint_dir: str, lower_is_better: bool = True, resume: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_checkpoint_dir = self.checkpoint_dir / "best"
        self.best_checkpoint_dir.mkdir(exist_ok=resume)
        self.last_checkpoint_dir = self.checkpoint_dir / "last"
        self.last_checkpoint_dir.mkdir(exist_ok=resume)
        self.lower_is_better = lower_is_better
        self._metric_value = float("inf") if lower_is_better else -float("inf")
        self._best_epoch = None
        self.resume = resume

    @property
    def metric_value(self):
        return self._metric_value

    def update_checkpoint_dir(self, path: str) -> None:
        self.checkpoint_dir = Path(path)

    def is_better(self, metric):
        if self.lower_is_better:
            return metric < self._metric_value
        return metric > self._metric_value

    def save_model(self, model, model_dict: Dict[str, Any], output_dir: str) -> None:
        output_dir = Path(output_dir)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        elif isinstance(model, torch.nn.Module):
            # save the model
            save_on_master(model.state_dict(), output_dir / f"model_checkpoint.pt")
        else:
            raise "Cannot save the give model object."

        # save the model_dict (optimizer, lr_scheduler, epoch, hparams_initial, hparams)
        save_on_master(model_dict, output_dir / f"training_artifacts.pt")

    def save(
        self, model, model_dict: Dict[str, Any], stats: Dict[str, Any], metric: float, epoch: int
    ) -> None:
        # save the last checkpoint
        self.save_model(model, model_dict, self.last_checkpoint_dir)

        is_best = self.is_better(metric)
        if is_best:
            self._metric_value = metric
            self._best_epoch = epoch
            self.save_model(model, model_dict, self.best_checkpoint_dir)

        stats["best_epoch"] = self._best_epoch if epoch > 0 else None
        stats["best_metric"] = self._metric_value
        if is_main_process():
            with (self.checkpoint_dir / "training_logs.jsonl").open("a") as f:
                f.write(json.dumps(stats) + "\n")

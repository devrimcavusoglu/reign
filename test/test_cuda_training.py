"""CUDA training tests for ``reign.train.ReignLitModel``.

Covers the GPU path end to end: device placement of the encoder and the
Guidance Network, a real training step (finite loss, gradients reaching the
REIGN parameters), allocation/cleanup across steps, the ``16-mixed`` precision
path, the cached-embedding batch path, and a full ``Trainer.fit``.

Offline and cheap: instead of pulling a GN from the Hub, every test builds a
randomly-initialised BERT encoder on disk (``tiny_gn``) and pairs it with
``tiny-l1``, the smallest entry of the ``train.py`` config registry. The whole
file needs a few tens of MB of VRAM and a couple of seconds.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizerFast

from reign.dataset import collate_cached_data, collate_data
from reign.eval import Evaluator
from reign.feature_extractor import ReignFeatureExtractor
from reign.train import ReignLitModel
from reign.utils import CheckpointHandler

SEED = 17
BATCH_SIZE = 4
# stride < chunk_size, i.e. the sliding-window chunking the paper uses; the
# feature extractor rejects stride > chunk_size outright.
CHUNK_SIZE = 64
STRIDE = 48
# 192d / 1 layer / gn_projection_dim taken from the GN — the smallest registry entry.
MODEL_CONFIG = "tiny-l1"
REIGN_HIDDEN_SIZE = 192

# Stepping the module outside a Trainer is the point of the standalone tests;
# Lightning's "self.log() has no trainer" notice is expected there, not a defect.
NO_TRAINER_LOG = "ignore:You are trying to `self.log\\(\\)`"
# Likewise for the fit-based tests: single-process loading is deliberate (workers
# would fork the CUDA context), and the precision test has nothing to validate.
LIGHTNING_SETUP_NOTES = (
    "ignore:The '.*_dataloader' does not have many workers",
    "ignore:You defined a `validation_step` but have no `val_dataloader`",
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cuda_device():
    """CUDA device, or skip: every test in this file is GPU-only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def tiny_gn(tmp_path_factory) -> str:
    """Path to a randomly-initialised, on-disk BERT Guidance Network.

    ``ReignFeatureExtractor`` needs a *fast* tokenizer (chunking relies on
    ``return_overflowing_tokens``) and an encoder whose ``hidden_size`` becomes
    REIGN's ``gn_projection_dim``. Building both locally keeps the tests
    hermetic — no network, no Hub cache — and their footprint negligible.
    """
    root = tmp_path_factory.mktemp("tiny_gn")
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = "the a an and of with for article text chunk rewritten unrelated about".split()
    vocab = specials + words + [f"w{i}" for i in range(64)]
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    gn_dir = root / "gn"
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True, model_max_length=128)
    encoder = BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
        )
    )
    tokenizer.save_pretrained(gn_dir)
    encoder.save_pretrained(gn_dir)
    return str(gn_dir)


def _doc(tag: str, idx: int, repeat: int) -> str:
    """Deterministic filler; ``repeat`` controls how many GN chunks it spans."""
    return f"the {tag} article w{idx} with text about a chunk " * repeat


def _sample_rows(n: int = 8) -> List[Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]]:
    """``ReignDataset``-shaped rows: (original, pair, [distractors])."""
    return [
        (
            {"text": _doc("original", i, 12 + i)},
            {"text": _doc("rewritten", i, 10 + i)},
            [{"text": _doc("unrelated", i, 9 + i)}],
        )
        for i in range(n)
    ]


def _mock_hooks() -> Tuple[MagicMock, MagicMock]:
    """Stand-ins for the Evaluator / CheckpointHandler that ``train.main`` wires in."""
    evaluator = MagicMock(spec=Evaluator)
    checkpoint_handler = MagicMock(spec=CheckpointHandler)
    checkpoint_handler.metric_value = 0.0
    return evaluator, checkpoint_handler


def _lit_model(
    gn_path: str,
    *,
    use_cached_embeddings: bool = False,
    lr: float = 1e-3,
    model_config: str = MODEL_CONFIG,
):
    """Build ``ReignLitModel`` the way ``train.main`` does, at tiny scale.

    Seeds first so two calls with the same arguments give the same weights.
    """
    L.seed_everything(SEED, workers=True)
    evaluator, checkpoint_handler = _mock_hooks()
    model = ReignLitModel(
        batch_size=BATCH_SIZE,
        lr=lr,
        max_epochs=1,
        weight_decay=0.01,
        temperature=0.1,
        weight_partial=0.5,
        negative_batch_size_multiplier=2,
        device="cuda",
        gn_device="cuda",
        gn_batch_size=BATCH_SIZE,
        gn_model=gn_path,
        chunk_size=CHUNK_SIZE,
        stride=STRIDE,
        loss_function="cosine",
        model_config=model_config,
        use_cached_embeddings=use_cached_embeddings,
        checkpoint_handler=checkpoint_handler,
        evaluator=evaluator,
    )
    return model, evaluator, checkpoint_handler


def _grad_stats(model: ReignLitModel) -> Tuple[int, bool]:
    """(number of REIGN params carrying a finite grad, whether any is non-zero)."""
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "gradients must stay finite"
    return len(grads), any(g.abs().sum() > 0 for g in grads)


class _PairDataset(Dataset):
    """In-memory stand-in for ``ReignDataset``; pairs with ``collate_data``."""

    def __init__(self, n: int = 8):
        self.rows = _sample_rows(n)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


class TestCUDATraining:
    """Test suite for CUDA training functionality."""

    @pytest.mark.cuda
    def test_cuda_availability(self):
        """CUDA is visible and basic device operations work."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this system")

        assert torch.cuda.device_count() > 0
        properties = torch.cuda.get_device_properties(0)
        assert properties.total_memory > 0
        major, _ = torch.cuda.get_device_capability(0)
        assert major >= 3  # anything older cannot run the training kernels

        tensor = torch.randn(8, 8, device="cuda")
        assert tensor.is_cuda
        assert torch.isfinite(tensor @ tensor.T).all()

    @pytest.mark.cuda
    def test_model_initialization_cuda(self, tiny_gn, cuda_device):
        """``ReignLitModel`` lands on CUDA, is wired from the GN, and encodes text."""
        model, evaluator, checkpoint_handler = _lit_model(tiny_gn)

        assert all(p.is_cuda for p in model.parameters())
        assert model.feature_extractor.device.type == "cuda"
        assert model.evaluator is evaluator
        assert model.checkpoint_handler is checkpoint_handler

        # The registry entry supplies the shape; chunking parameters and the GN
        # width are injected by ReignLitModel and persisted on the config.
        config = model.model.config
        assert config.hidden_size == REIGN_HIDDEN_SIZE
        assert config.num_hidden_layers == 1
        assert config.gn_projection_dim == model.feature_extractor.model.config.hidden_size
        assert (config.gn_chunk_size, config.gn_stride) == (CHUNK_SIZE, STRIDE)

        texts = [_doc("original", i, 12 + i) for i in range(BATCH_SIZE)]
        with torch.no_grad():
            outputs = model.forward(texts)
        pooled = outputs.pooler_output
        assert pooled.is_cuda
        assert pooled.shape == (BATCH_SIZE, REIGN_HIDDEN_SIZE)
        assert torch.isfinite(pooled).all()

        # Sizes outside the registry are rejected up front, with the valid names.
        with pytest.raises(ValueError, match=r"Invalid model_config"):
            _lit_model(tiny_gn, model_config="no-such-size")

    @pytest.mark.cuda
    @pytest.mark.filterwarnings(NO_TRAINER_LOG)
    def test_training_step_cuda(self, tiny_gn, cuda_device):
        """A training step computes a finite CUDA loss and gradients reach REIGN."""
        model, _, _ = _lit_model(tiny_gn)
        batch = collate_data(_sample_rows(BATCH_SIZE))

        model.train()
        loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.is_cuda
        assert loss.requires_grad
        assert torch.isfinite(loss)

        loss.backward()
        n_grads, any_nonzero = _grad_stats(model)
        assert n_grads == sum(1 for p in model.model.parameters() if p.requires_grad)
        assert any_nonzero, "no gradient signal reached the REIGN encoder"

        # Same seed, same batch, same loss: the step is reproducible.
        replay, _, _ = _lit_model(tiny_gn)
        replay.train()
        assert torch.allclose(replay.training_step(batch, batch_idx=0), loss, atol=1e-6)

    @pytest.mark.cuda
    @pytest.mark.slow
    @pytest.mark.filterwarnings(NO_TRAINER_LOG)
    def test_memory_management_cuda(self, tiny_gn, cuda_device):
        """Repeated steps do not leak, and dropping the model returns its VRAM."""
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        model, _, _ = _lit_model(tiny_gn)
        after_model = torch.cuda.memory_allocated()
        assert after_model > baseline, "model weights should occupy device memory"

        model.train()
        per_step = []
        for batch_idx, rows in enumerate(
            [_sample_rows(BATCH_SIZE), _sample_rows(BATCH_SIZE), _sample_rows(BATCH_SIZE)]
        ):
            loss = model.training_step(collate_data(rows), batch_idx=batch_idx)
            assert torch.isfinite(loss)
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            per_step.append(torch.cuda.memory_allocated())

        # Activations are released with each loss, so the resident set stays flat.
        drift_mb = (per_step[-1] - per_step[0]) / 1024 ** 2
        assert drift_mb < 32, f"memory grew {drift_mb:.2f} MB across steps"

        del model
        gc.collect()
        torch.cuda.empty_cache()
        residual_mb = (torch.cuda.memory_allocated() - baseline) / 1024 ** 2
        assert residual_mb < 1, f"{residual_mb:.2f} MB still resident after dropping the model"

    @pytest.mark.cuda
    def test_cuda_error_handling(self, cuda_device):
        """CUDA-specific failures raise, and the context survives them."""
        # An allocation larger than the card can ever serve must fail cleanly.
        too_many_elements = torch.cuda.get_device_properties(0).total_memory  # 4 bytes each
        with pytest.raises(torch.cuda.OutOfMemoryError):
            torch.empty(too_many_elements, dtype=torch.float32, device=cuda_device)
        torch.cuda.empty_cache()

        # The context is still usable — an OOM must not poison the device.
        assert torch.randn(8, 8, device=cuda_device).is_cuda

        # Mixing devices is an error rather than an implicit transfer.
        with pytest.raises(RuntimeError, match="device"):
            torch.randn(4, 4, device=cuda_device) + torch.randn(4, 4, device="cpu")

    @pytest.mark.cuda
    @pytest.mark.filterwarnings(*LIGHTNING_SETUP_NOTES)
    def test_mixed_precision_training(self, tiny_gn, cuda_device):
        """The ``16-mixed`` path trains: scaled fp16 forward, finite loss, real update."""
        model, _, _ = _lit_model(tiny_gn, lr=1e-2)
        loader = DataLoader(_PairDataset(BATCH_SIZE), batch_size=BATCH_SIZE, collate_fn=collate_data)
        before = [p.detach().cpu().clone() for p in model.model.parameters()]

        trainer = L.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            limit_train_batches=1,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, loader)

        assert trainer.precision == "16-mixed"
        # fp16 needs loss scaling; bf16/32 plugins carry no scaler.
        assert trainer.precision_plugin.scaler is not None
        assert trainer.state.finished

        loss = trainer.callback_metrics["train/loss"]
        assert torch.isfinite(loss), "fp16 overflowed instead of being rescaled"

        after = [p.detach().cpu().clone() for p in model.model.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after)), "scaler skipped every step"

    @pytest.mark.cuda
    @pytest.mark.filterwarnings(NO_TRAINER_LOG)
    def test_cached_embeddings_training(self, tiny_gn, cuda_device, tmp_path):
        """Precompute GN embeddings, collate them, and train from the cache."""
        extractor = ReignFeatureExtractor(
            batch_size=BATCH_SIZE,
            model_name_or_path=tiny_gn,
            chunk_size=CHUNK_SIZE,
            stride=STRIDE,
            device=cuda_device,
            cache_root=tmp_path / "cache",
            enable_cache=True,
        )

        originals = [_doc("original", i, 12 + 6 * i) for i in range(BATCH_SIZE)]
        rewritten = [_doc("rewritten", i, 10 + 5 * i) for i in range(BATCH_SIZE)]
        unrelated = [_doc("unrelated", i, 9 + 4 * i) for i in range(BATCH_SIZE)]

        cached = {
            name: extractor.compute_and_cache_dataset_embeddings(texts, name)
            for name, texts in (
                ("originals", originals),
                ("rewritten", rewritten),
                ("unrelated", unrelated),
            )
        }
        for name, entries in cached.items():
            assert len(entries) == BATCH_SIZE, f"{name} cache is incomplete"
            for embeddings, mask in entries:
                # Cached tensors stay on CPU by design: __getitem__ runs in
                # DataLoader workers, which cannot touch CUDA.
                assert embeddings.device.type == "cpu" and mask.device.type == "cpu"
                assert embeddings.dim() == 2 and mask.dim() == 1
                assert embeddings.shape[0] == mask.shape[0]
        # Longer documents produce more chunks, so padding is actually exercised.
        assert len({e.shape[0] for e, _ in cached["originals"]}) > 1

        subset = extractor.get_cached_embeddings("originals", indices=[0, 1])
        assert subset is not None and len(subset) == 2

        # One distractor per sample: get_combined_batch requires the flattened
        # distractor count to be a whole multiple of the batch size.
        samples = [
            (
                (*cached["originals"][i], {"article_id": f"O{i}"}),
                (*cached["rewritten"][i], {"article_id": f"S{i}"}),
                [(*cached["unrelated"][i], {"article_id": f"D{i}"})],
            )
            for i in range(BATCH_SIZE)
        ]
        collated = collate_cached_data(samples)
        assert len(collated) == 7  # 3 (embeddings, mask) pairs + metadata
        orig_emb, orig_mask, synth_emb, synth_mask, dist_emb, dist_mask, metadata = collated
        assert orig_emb.shape[0] == synth_emb.shape[0] == dist_emb.shape[0] == BATCH_SIZE
        assert orig_emb.shape[-1] == extractor.model.config.hidden_size
        # Each stream is padded to its own longest document, with a mask to match.
        for emb, mask in ((orig_emb, orig_mask), (synth_emb, synth_mask), (dist_emb, dist_mask)):
            assert emb.shape[:2] == mask.shape
            assert mask.dtype == torch.int64
            assert mask.sum() > 0
        assert [m["article_id"] for m in metadata["original_metadata"]] == [
            f"O{i}" for i in range(BATCH_SIZE)
        ]

        model, _, _ = _lit_model(tiny_gn, use_cached_embeddings=True)
        assert model.use_cached_embeddings
        # Lightning normally does this transfer; the standalone step must not.
        batch = tuple(t.to(cuda_device) if torch.is_tensor(t) else t for t in collated)

        model.train()
        loss = model.training_step(batch, batch_idx=0)
        assert loss.is_cuda and torch.isfinite(loss)

        loss.backward()
        _, any_nonzero = _grad_stats(model)
        assert any_nonzero, "cached-embedding step produced no gradient signal"

        # The cache survives the extractor that wrote it, keyed by stride.
        reopened = ReignFeatureExtractor(
            batch_size=BATCH_SIZE,
            model_name_or_path=tiny_gn,
            chunk_size=CHUNK_SIZE,
            stride=STRIDE,
            device=cuda_device,
            cache_root=tmp_path / "cache",
            enable_cache=True,
        )
        assert reopened.cache.has_cache(tiny_gn, CHUNK_SIZE, "originals", stride=STRIDE)
        assert not reopened.cache.has_cache(tiny_gn, CHUNK_SIZE, "originals", stride=CHUNK_SIZE)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.filterwarnings(*LIGHTNING_SETUP_NOTES)
def test_cuda_training_integration(tiny_gn, cuda_device):
    """Full ``Trainer.fit`` on GPU: both loops run, weights move, hooks fire."""
    model, evaluator, checkpoint_handler = _lit_model(tiny_gn, lr=1e-2)
    train_loader = DataLoader(_PairDataset(8), batch_size=BATCH_SIZE, collate_fn=collate_data)
    val_loader = DataLoader(_PairDataset(4), batch_size=BATCH_SIZE, collate_fn=collate_data)
    before = [p.detach().cpu().clone() for p in model.model.parameters()]

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    assert trainer.state.finished
    assert trainer.global_step == len(train_loader)
    for key in ("train/loss", "val/loss"):
        assert torch.isfinite(trainer.callback_metrics[key]), f"{key} is not finite"

    after = [p.detach().cpu().clone() for p in model.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "fit left the weights untouched"

    # The validation hook drives evaluation and checkpointing once per epoch.
    evaluator.evaluate_with_integrated_dataset.assert_called_once()
    checkpoint_handler.save.assert_called_once()

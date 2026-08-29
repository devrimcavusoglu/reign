"""Standard-IR contrastive fine-tuning entry point (query / positive / negative).

The methodologically-correct replacement for the GoodWiki-Long-Synthetic
``original/synthetic/distractor`` shim when fine-tuning REIGN on standard
``(query, corpus, qrels)`` IR datasets (e.g. DAPFAM, GoodWiki-IR). See
``reign/ir_dataset.py`` for the data semantics.

* Standard InfoNCE: anchor = query, positive = a relevant doc; negatives are the
  dataset's **own provided negatives** (``score==0``) and/or in-batch
  other-query positives — with **false-negative masking** so a doc relevant to
  the anchor (positive, or a soft-positive partial) is never its negative.
* Graded relevance via ``--partial-policy`` (no stored-score refactor):
  ``score==1`` → ``soft_positive`` (loss ``partial_weight`` α) | ``negative`` |
  ``ignore``. ``score==1`` semantics are dataset-dependent (GoodWiki-IR=topical
  distractor=negative; DAPFAM has none; graded-IR corpora typically=partial).
* Frozen GN, only the REIGN encoder trains. The in-training val metric is the
  existing ``Evaluator`` in-batch proxy (checkpoint selection only); the
  authoritative number comes from ``scripts/evaluate_reign.py``.

DAPFAM example::

    python -m reign.train_ir \\
        --dataset data/dapfam_ir_fulltext --train-split train --eval-split val \\
        --gn-model thenlper/gte-base --model-config base-l3 \\
        --output-dir reign-base-l3_gn-gte-base_dapfam-ft-c512s512 \\
        --partial-policy ignore --n-negatives-per-sample 20 --no-in-batch-negatives \\
        --chunk-size 512 --gn-stride 512 --max-epochs 15 \\
        --temperature 0.07 --enable-cache --precision 16-mixed
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from reign import MODEL_DIR
from reign.dataset import collate_cached_data
from reign.eval import Evaluator
from reign.feature_extractor import ReignFeatureExtractor
from reign.ir_dataset import IRContrastiveCachedDataset, collate_cached_ir_data
from reign.train import ReignLitModel
from reign.utils import CheckpointHandler, get_local_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ir")


class ReignIRLitModel(ReignLitModel):
    """ReignLitModel with a standard-IR ``compute_loss``.

    Overrides only the loss path: REIGN-forwards query/positive/partial/negative
    role tensors, builds the InfoNCE ``(input1, input2, target)`` rows + a
    ``(B, B_neg)`` false-negative mask (by corpus-id membership, never roll
    arithmetic), and calls the (graded-capable, mask-aware) ``InfoNCELoss``.
    Validation is delegated to the inherited ``Evaluator`` path (see
    ``on_validation_epoch_end``); ``validation_step`` is a no-op.
    """

    def __init__(self, *args, in_batch_negatives: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_batch_negatives = in_batch_negatives

    def validation_step(self, batch, batch_idx):
        # Metric comes from the Evaluator over the eval loader in
        # on_validation_epoch_end; no per-batch val loss needed.
        return None

    def compute_loss(self, batch, mode="train"):
        (q_emb, q_mask, p_emb, p_mask, par_emb, par_mask, neg_emb, neg_mask, meta) = batch

        q = self.model(inputs_embeds=q_emb, attention_mask=q_mask).pooler_output  # (B, D)
        p = self.model(inputs_embeds=p_emb, attention_mask=p_mask).pooler_output  # (B, D)
        B, D = q.shape
        device = q.device

        # Negative pool (column order must match neg_cids): provided negatives
        # first, then optional in-batch other-query positives.
        neg_blocks, neg_cids = [], []
        if neg_emb.dim() == 3 and neg_emb.shape[0] > 0:
            neg_blocks.append(self.model(inputs_embeds=neg_emb, attention_mask=neg_mask).pooler_output)
            neg_cids += [str(m.get("article_id", "")) for m in meta["negative_metadata"]]
        if self.in_batch_negatives:
            neg_blocks.append(p)  # other-query positives are negatives (mask removes own/relevant)
            neg_cids += [str(m.get("article_id", "")) for m in meta["synthetic_metadata"]]

        if not neg_blocks:
            return torch.zeros((), device=device, requires_grad=True)
        negatives = torch.cat(neg_blocks, dim=0)  # (N_neg, D)
        n_neg = negatives.shape[0]

        input1 = [q]
        input2 = [p]
        target = [torch.ones(B, device=device)]

        use_partials = (
            getattr(self.loss, "partial_weight", 0.0) > 0
            and par_emb.dim() == 3
            and par_emb.shape[0] > 0
        )
        if use_partials:
            K = par_emb.shape[0] // B
            par = self.model(inputs_embeds=par_emb, attention_mask=par_mask).pooler_output
            input1.append(q.repeat_interleave(K, dim=0))  # (B·K, D) — loss (B,K) contract
            input2.append(par)
            target.append(torch.zeros(B * K, device=device))

        input1.append(torch.zeros(n_neg, D, device=device, dtype=q.dtype))  # unused by loss
        input2.append(negatives)
        target.append(torch.full((n_neg,), -1.0, device=device))

        input1 = torch.cat(input1, dim=0)
        input2 = torch.cat(input2, dim=0)
        target = torch.cat(target, dim=0)

        # False-negative mask (B, n_neg): True where a negative column is in
        # fact relevant to that anchor (its own positive, or any score≥1 doc).
        rel_sets = meta["anchor_relevant_idsets"]  # length B, sets of str cids
        mask = torch.zeros(B, n_neg, dtype=torch.bool)
        for j, cid in enumerate(neg_cids):
            if not cid:
                continue
            for i in range(B):
                if cid in rel_sets[i]:
                    mask[i, j] = True

        loss = self.loss(input1, input2, target, false_neg_mask=mask.to(device))
        self.log(f"{mode}/loss", loss, batch_size=B)
        return loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    # Dataset
    p.add_argument(
        "--dataset",
        required=True,
        help=(
            "IR dataset in BEIR/MTEB layout: either a Hugging Face Hub id or a local "
            "directory. It must expose three configs — 'corpus' (split 'corpus', "
            "columns _id/title/text), 'queries' (split 'queries', columns _id/text) and "
            "'default' (splits train/val/test of qrels, columns query-id/corpus-id/score, "
            "score 2=relevant, 1=partially relevant, 0=provided negative). Local "
            "directories may store each config as a DatasetDict save_to_disk tree."
        ),
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="val")
    p.add_argument("--max-samples", type=int, default=None)
    # IR semantics
    p.add_argument(
        "--partial-policy",
        choices=["soft_positive", "negative", "ignore"],
        default="soft_positive",
        help="How to treat score==1 docs (dataset-dependent; DAPFAM=ignore, "
        "GoodWiki-IR=negative, graded-relevance corpora=soft_positive).",
    )
    p.add_argument("--n-partials-per-sample", type=int, default=0)
    p.add_argument("--n-negatives-per-sample", type=int, default=0)
    p.add_argument(
        "--in-batch-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also use other-query positives in the batch as negatives "
        "(false-neg masked). Off for DAPFAM (provided negatives); on for "
        "GoodWiki-IR.",
    )
    # Model
    p.add_argument("--gn-model", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--from-checkpoint",
        default=None,
        help="Lightning RESUME: restore weights + optimizer + epoch from a "
        "Lightning .ckpt and continue that run.",
    )
    p.add_argument(
        "--warm-start-from",
        default=None,
        help="Domain-adaptation WARM START: load weights from an HF "
        "ReignModel save_pretrained dir (e.g. a GoodWiki-trained backbone) but "
        "start a FRESH run on this dataset — new optimizer, LR schedule, epoch "
        "0, and a fresh best-metric tracker writing to --output-dir. Mutually "
        "exclusive with --from-checkpoint.",
    )
    # Training
    p.add_argument("--batch-size", type=int, default=18)
    p.add_argument("--eval-batch-size", type=int, default=18)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument(
        "--weight-partial",
        type=float,
        default=0.0,
        help="α: soft-positive weight in InfoNCE numerator (only with "
        "--partial-policy soft_positive).",
    )
    p.add_argument("--gradient-clip-val", type=float, default=None)
    p.add_argument("--gradient-clip-algorithm", default="norm")
    # Chunking / cache
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--gn-stride", "--stride", dest="stride", type=int, default=512)
    p.add_argument("--cache-root", default=os.path.expanduser("~/.reign_cache"))
    p.add_argument("--enable-cache", action="store_true", default=True)
    # Lightning
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-loader-num-workers", type=int, default=4)
    p.add_argument("--check-val-every-n-epoch", type=int, default=3)
    p.add_argument("--num-sanity-val-steps", type=int, default=0)
    p.add_argument("--log-every-n-steps", type=int, default=5)
    # Eval / monitor
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--metric-to-monitor", default="ndcg@10")
    p.add_argument(
        "--logger",
        default="csv",
        choices=["csv", "tensorboard", "none"],
        help="Local Lightning logger for train/val metrics (default: csv, written to logs/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting standard-IR InfoNCE training with: %s", vars(args))

    if args.from_checkpoint is not None and args.warm_start_from is not None:
        raise ValueError(
            "--from-checkpoint (Lightning resume) and --warm-start-from "
            "(fresh run, warm weights) are mutually exclusive."
        )
    # Both load weights into the model; only --from-checkpoint also resumes the
    # Lightning run (optimizer/epoch). Warm-start → fresh optimizer, ckpt_path None.
    weight_init = args.from_checkpoint or args.warm_start_from

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(args.seed)

    output_dir = MODEL_DIR / args.output_dir
    # Refuse to clobber a *real* prior run (a committed checkpoint), but tolerate
    # an empty/partial leftover dir — those are routinely pre-created by a
    # crashed run or the checkpoint handler and must not block a fresh start.
    has_ckpt = (output_dir / "best" / "config.json").exists() or (
        output_dir / "last" / "config.json"
    ).exists()
    if has_ckpt and args.from_checkpoint is None:
        raise FileExistsError(
            f"{output_dir} already holds a checkpoint; pass --from-checkpoint to "
            f"resume, or remove the directory to retrain from scratch."
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Output dir: %s", output_dir)

    feature_extractor = ReignFeatureExtractor(
        batch_size=args.batch_size,
        device=args.device,
        model_name_or_path=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        cache_root=args.cache_root,
        enable_cache=args.enable_cache,
    )

    logger.info("Building train (IR) / eval (proxy) dataloaders")
    train_ds = IRContrastiveCachedDataset(
        dataset_name=args.dataset,
        qrels_split=args.train_split,
        feature_extractor=feature_extractor,
        partial_policy=args.partial_policy,
        n_partials_per_sample=args.n_partials_per_sample,
        n_negatives_per_sample=args.n_negatives_per_sample,
        mode="train",
        max_samples=args.max_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_loader_num_workers,
        collate_fn=collate_cached_ir_data,
    )
    # Eval loader: positives(+partials)-only, no negatives → the existing
    # Evaluator in-batch proxy works unchanged (Refinement-1).
    eval_ds = IRContrastiveCachedDataset(
        dataset_name=args.dataset,
        qrels_split=args.eval_split,
        feature_extractor=feature_extractor,
        partial_policy=args.partial_policy,
        n_partials_per_sample=args.n_partials_per_sample,
        n_negatives_per_sample=0,
        mode="eval",
        max_samples=args.max_samples,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.data_loader_num_workers,
        collate_fn=collate_cached_data,
    )
    logger.info("Train batches: %d | Eval batches: %d", len(train_loader), len(eval_loader))

    evaluator = Evaluator(
        batch_size=args.eval_batch_size,
        data_loader=eval_loader,
        top_k=args.top_k,
        use_cached_embeddings=True,
    )
    checkpoint_handler = CheckpointHandler(
        checkpoint_dir=output_dir,
        lower_is_better=False,
        resume=args.from_checkpoint is not None,
    )

    model = ReignIRLitModel(
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        weight_partial=args.weight_partial,
        temperature=args.temperature,
        max_epochs=args.max_epochs,
        negative_batch_size_multiplier=0,  # IR path forms its own negatives.
        checkpoint_handler=checkpoint_handler,
        evaluator=evaluator,
        device=args.device,
        loss_function="infonce",
        gn_model=args.gn_model,
        chunk_size=args.chunk_size,
        stride=args.stride,
        from_checkpoint=weight_init,
        use_cached_embeddings=True,
        model_config=args.model_config,
        metric_to_monitor=args.metric_to_monitor,
        lower_is_better=False,
        in_batch_negatives=args.in_batch_negatives,
    )

    train_logger = get_local_logger(
        kind=args.logger,
        save_dir="logs",
        name=output_dir.name,
        hparams=vars(args),
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
        logger=train_logger,
        enable_checkpointing=False,
    )

    logger.info("Starting training (Lightning Trainer.fit)")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
        ckpt_path=args.from_checkpoint,
    )
    logger.info("Training complete. Best metric: %s", checkpoint_handler.metric_value)


if __name__ == "__main__":
    main()

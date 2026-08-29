from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeWayCosineEmbeddingLoss(nn.Module):
    """
    Three-way cosine embedding loss that considers positive, negative, and partial matches.

    This loss function extends the standard CosineEmbeddingLoss by adding a third state
    for partial matches (s=0), which uses a weighted combination of the positive and
    negative loss terms.

    The loss is defined as:
    L(x, y, s) =
        1 - cos(x, y)                                              if s = 1 (positive pair)
        λ * (1 - cos(x, y)) + (1 - λ) * max(0, cos(x, y))          if s = 0 (partial match)
        max(0, cos(x, y))                                          if s = -1 (negative pair)

    Note:
    - λ is the weight parameter for partial matches.
    - s is the target value, which can be 1 (positive pair), 0 (partial match), or -1 (negative pair).
    - The loss is equivalent to the standard CosineEmbeddingLoss when s = 1 or -1 or when
        λ is either 0 (partial matches are treated as negative pairs) or 1 (partial matches
        are treated as positive pairs).

    Args:
        weight_partial (float, optional): Weight parameter for partial matches. Defaults to 0.5.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """

    def __init__(self, weight_partial: float = 0.5, reduction: str = "mean"):
        """
        Initialize the ThreeWayCosineEmbeddingLoss.

        Args:
            weight_partial (float, optional): Weight parameter for partial matches. Defaults to 0.5.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(ThreeWayCosineEmbeddingLoss, self).__init__()
        self.weight_partial = weight_partial
        self.reduction = reduction

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the three-way cosine embedding loss.

        Args:
            input1 (torch.Tensor): First input tensor of shape (N, D) where N is the batch size
                                  and D is the embedding dimension.
            input2 (torch.Tensor): Second input tensor of shape (M, D).
            target (torch.Tensor): Target tensor of shape (N,) with values 1, 0, or -1:
                                  1 for positive pairs, 0 for partial matches, -1 for negative pairs.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(input1, input2, dim=1)

        # Compute loss for each sample based on target value
        positive_mask = target == 1
        partial_mask = target == 0
        negative_mask = target == -1

        # Initialize loss tensor
        loss = torch.zeros_like(cos_sim)

        # Positive pairs: 1 - cos(x, y)
        if positive_mask.any():
            loss[positive_mask] = 1 - cos_sim[positive_mask]

        # Partial matches: λ * (1 - cos(x, y)) + (1 - λ) * max(0, cos(x, y))
        if partial_mask.any():
            partial_loss = self.weight_partial * (1 - cos_sim[partial_mask]) + (
                1 - self.weight_partial
            ) * torch.clamp(cos_sim[partial_mask], min=0)
            loss[partial_mask] = partial_loss

        # Negative pairs: max(0, cos(x, y))
        if negative_mask.any():
            loss[negative_mask] = torch.clamp(cos_sim[negative_mask], min=0)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning, with optional graded soft positives.

    Standard formulation (``partial_weight=0``, default):

        L(x, y) = -log[ exp(sim(x, y+)/τ) / (exp(sim(x, y+)/τ) + Σ exp(sim(x, y-)/τ)) ]

    With ``partial_weight > 0``, ``target == 0`` rows are treated as graded
    soft positives instead of being dropped:

        L = -log[ (exp(sim(x, y+)/τ) + α · Σ_k exp(sim(x, y_p^k)/τ))
                  /
                  (exp(sim(x, y+)/τ) + α · Σ_k exp(sim(x, y_p^k)/τ) + Σ exp(sim(x, y-)/τ)) ]

    where ``α = partial_weight``. This is the principled way to incorporate
    graded relevance labels: score=2 docs pull at full weight, score=1
    partials pull at weight ``α``, and in-batch other-query docs are negatives.
    With ``α = 0`` the loss recovers standard InfoNCE; with ``α = 1`` partials
    are treated identically to positives.

    Anchor-to-partial matching uses positional row-correspondence: ``input1[i]``
    is the anchor for ``input2[i]`` regardless of target. So when the trainer
    emits B positive rows, B·K partial rows (K partials per anchor, anchors
    repeated), and B·nbsm negative rows, the partials are interpreted as
    "K partials per query, in the order they appear after the positive block".
    The trainer's ``get_combined_batch`` does this layout deterministically
    via ``repeat_interleave``.

    Args:
        temperature (float, optional): Softmax temperature. Defaults to 0.01.
        partial_weight (float, optional): Weight of soft positives in the
            numerator. ``0`` recovers standard InfoNCE (default). Typical
            graded-IR range: 0.3–0.7.
        reduction (str, optional): 'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """

    def __init__(
        self,
        temperature: float = 0.01,
        partial_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.partial_weight = partial_weight
        self.reduction = reduction

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        false_neg_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for the InfoNCE loss.

        Args:
            input1 (torch.Tensor): Anchor embeddings of shape (N, D)
            input2 (torch.Tensor): Paired embeddings of shape (N, D)
            target (torch.Tensor): Target tensor of shape (N,) with values 1, 0, or -1.
            false_neg_mask (torch.Tensor, optional): Boolean tensor of shape
                ``(B, B_neg)`` aligned to the ``(anchor, negative-column)`` grid of
                ``neg_sim`` (B = #positive/anchor rows, B_neg = #negative rows).
                ``True`` marks an (anchor, negative) cell where that negative is
                actually relevant to the anchor (a false negative — e.g. a shared
                in-batch column that is another query's positive but also relevant
                here); such cells are removed from the denominator. ``None``
                (default) recovers standard InfoNCE bit-for-bit.

        Returns:
            torch.Tensor: The computed loss value.
        """
        device = input1.device

        pos_mask = target == 1
        partial_mask = target == 0
        neg_mask = target == -1

        if not pos_mask.any() or not neg_mask.any():
            return torch.zeros((), device=device, requires_grad=True)

        # Normalised per-row embeddings ------------------------------------
        anchors = F.normalize(input1[pos_mask], dim=1)  # (B, D)
        positives = F.normalize(input2[pos_mask], dim=1)  # (B, D)
        negatives = F.normalize(input2[neg_mask], dim=1)  # (B_neg, D)

        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature  # (B,)
        neg_sim = torch.matmul(anchors, negatives.T) / self.temperature  # (B, B_neg)

        # False-negative masking: drop (anchor, negative) cells where the
        # negative is in fact relevant to the anchor (standard in-batch InfoNCE
        # shares negative columns across anchors, so another query's positive —
        # or a provided negative that is a partial here — can be a false neg).
        # exp(-inf) == 0 ⇒ the cell vanishes from the denominator sum below.
        if false_neg_mask is not None:
            if false_neg_mask.shape != neg_sim.shape:
                raise ValueError(
                    f"false_neg_mask shape {tuple(false_neg_mask.shape)} != "
                    f"neg_sim shape {tuple(neg_sim.shape)} (B, B_neg)."
                )
            neg_sim = neg_sim.masked_fill(false_neg_mask.to(neg_sim.device), float("-inf"))

        # Numerator = exp(pos_sim) [+ α · mean_k exp(partial_sim)] ----------
        # ``mean`` (rather than ``sum``) decouples ``K`` from ``α``: the total
        # partial mass in the numerator is ``α`` regardless of how many
        # partials are sampled per query. This matches the graded relevance
        # interpretation — α=0.5 means "partial-relevance category contributes
        # half as much as fully-relevant", not "each partial doc contributes
        # half" (which under K>1 would dominate the positive signal and led
        # to the rapid val degradation observed under ``α=0.5, K=4`` with sum).
        numerator = torch.exp(pos_sim)  # (B,)
        if self.partial_weight > 0 and partial_mask.any():
            partial_anchors = F.normalize(input1[partial_mask], dim=1)  # (B·K, D)
            partials = F.normalize(input2[partial_mask], dim=1)  # (B·K, D)
            # Row-wise sim, reshape to (B, K) — assumes the trainer emits
            # partials as ``[anchor_0_partial_0, ..., anchor_0_partial_{K-1},
            # anchor_1_partial_0, ...]`` which is what ``repeat_interleave``
            # in ``get_combined_batch`` produces.
            partial_sim_flat = torch.sum(partial_anchors * partials, dim=1) / self.temperature
            B = anchors.shape[0]
            if partial_sim_flat.numel() % B != 0:
                raise ValueError(
                    f"Partial count {partial_sim_flat.numel()} is not a multiple of "
                    f"batch size {B}; cannot reshape to (B, K)."
                )
            K = partial_sim_flat.numel() // B
            partial_sim = partial_sim_flat.view(B, K)  # (B, K)
            numerator = numerator + self.partial_weight * torch.exp(partial_sim).mean(dim=1)

        # Denominator = numerator + Σ exp(neg_sim) -------------------------
        denominator = numerator + torch.exp(neg_sim).sum(dim=1)

        loss = -torch.log(numerator / (denominator + 1e-12))  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

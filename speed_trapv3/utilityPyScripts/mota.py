from __future__ import annotations

from typing import Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from sparrow_datums import AugmentedBoxTracking, BoxTracking, pairwise_iou


class MOTA:
    """A summable metric class to track the components of MOTA."""

    def __init__(
        self,
        false_negatives: int = 0,
        false_positives: int = 0,
        id_switches: int = 0,
        n_truth: int = 0,
    ) -> None:
        self.false_negatives = false_negatives
        self.false_positives = false_positives
        self.id_switches = id_switches
        self.n_truth = n_truth

    def __add__(self, other: Union[int, "MOTA"]) -> "MOTA":
        """Add two MODA objects."""
        if isinstance(other, int):
            return self
        return MOTA(
            false_negatives=self.false_negatives + other.false_negatives,
            false_positives=self.false_positives + other.false_positives,
            id_switches=self.id_switches + other.id_switches,
            n_truth=self.n_truth + other.n_truth,
        )

    def __radd__(self, other: Union[int, "MOTA"]) -> "MOTA":
        """Add two MOTA objects."""
        return self + other

    def __repr__(self) -> str:
        """Create a string representation."""
        return (
            f"MOTA(false_negatives={self.false_negatives}, "
            f"false_positives={self.false_positives}, "
            f"id_switches={self.id_switches}, "
            f"n_truth={self.n_truth})"
        )

    @property
    def value(self) -> float:
        """Compute the MODA metric."""
        n_errors = (
            abs(self.false_negatives)
            + abs(self.false_positives)
            + abs(self.id_switches)
        )
        if self.n_truth == 0:
            if n_errors == 0:
                return 1.0
            else:
                return 0.0
        return 1 - n_errors / self.n_truth

    def to_dict(self) -> dict[str, float]:
        """Return a dict representation of the MOTA object."""
        return dict(
            false_negatives=self.false_negatives,
            false_positives=self.false_positives,
            id_switches=self.id_switches,
            n_truth=self.n_truth,
            value=self.value,
        )


def compute_mota(
    predicted_tracking: Union[AugmentedBoxTracking, BoxTracking],
    ground_truth_tracking: Union[AugmentedBoxTracking, BoxTracking],
    iou_threshold: float = 0.5,
) -> MOTA:
    """
    Compute MOTA for a predicted box tracking chunk.

    Parameters
    ----------
    predicted_tracking : BoxTracking
        Predicted tracking
    ground_truth_tracking : BoxTracking
        Ground truth tracking
    iou_threshold : float
        The overlap threshold below which boxes are not considered the same
    """
    n_false_positives = 0
    n_false_negatives = 0
    n_id_switches = 0
    n_ground_truth = 0
    matches: dict[int, int] = {}
    for pred_frame, gt_frame in zip(predicted_tracking, ground_truth_tracking):
        pred_finite_mask = np.isfinite(pred_frame.x)
        gt_finite_mask = np.isfinite(gt_frame.x)
        finite_pred_frame = pred_frame[pred_finite_mask]
        finite_gt_frame = gt_frame[gt_finite_mask]
        n_ground_truth += len(finite_gt_frame)
        if len(finite_pred_frame) == 0:
            n_false_negatives += len(finite_gt_frame)
            continue
        elif len(finite_gt_frame) == 0:
            n_false_positives += len(finite_pred_frame)
            continue
        iou = pairwise_iou(pred_frame, gt_frame)
        cost = 1 - iou
        cost = np.nan_to_num(cost, nan=np.finfo(cost.dtype).max)
        pred_indices, gt_indices = linear_sum_assignment(cost)
        iou = np.nan_to_num(iou, nan=np.finfo(iou.dtype).min)
        valid = iou[pred_indices, gt_indices] > iou_threshold
        new_matches: dict[int, int] = {}
        for pred, gt in zip(pred_indices[valid], gt_indices[valid]):
            new_matches[pred] = gt
            if pred in matches and matches[pred] != gt:
                n_id_switches += 1
        matches = new_matches

        all_pred_indices = set(np.arange(len(pred_frame))[pred_finite_mask])
        all_gt_indices = set(np.arange(len(gt_frame))[gt_finite_mask])

        false_positives = all_pred_indices - set(pred_indices[valid])
        false_negatives = all_gt_indices - set(gt_indices[valid])

        n_false_positives += len(false_positives)
        n_false_negatives += len(false_negatives)

    return MOTA(
        false_negatives=n_false_negatives,
        false_positives=n_false_positives,
        id_switches=n_id_switches,
        n_truth=n_ground_truth,
    )

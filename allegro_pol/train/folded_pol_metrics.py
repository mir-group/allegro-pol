# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
import torch
from torchmetrics import Metric
from nequip.data import AtomicDataDict
from nequip.data._keys import POLARIZATION_KEY
from typing import Callable


class _FoldedPolarizationBase(Metric):
    """Base class for folded polarization metrics.

    Handles the polarization folding logic using minimum image convention,
    then applies a modifier function to the folded differences.

    Note: This metric requires access to batch and cell information, so it needs
    to be used with field=None in MetricsManager to get full AtomicDataDict objects.
    """

    def __init__(self, modifier: Callable = torch.nn.Identity(), **kwargs):
        super().__init__(**kwargs)
        self.modifier = modifier
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: AtomicDataDict.Type, target: AtomicDataDict.Type) -> None:
        """Update metric state with batch of predictions and targets.

        Args:
            preds: Model predictions as AtomicDataDict
            target: Target values as AtomicDataDict
        """
        # Check if we have cell information for folding
        if AtomicDataDict.CELL_KEY not in preds:
            raise ValueError("Cell information required for polarization folding")

        # Get variables
        cell = preds[AtomicDataDict.CELL_KEY].to(
            preds[POLARIZATION_KEY].dtype
        )  # (Nbatch, 3, 3)

        # Difference between prediction and label
        pol_diff = preds[POLARIZATION_KEY] - target[POLARIZATION_KEY]  # (Nbatch, 3)

        # Map pol_diff to fractional coordinates
        frac_pol_diff = torch.einsum(
            "bi, bij -> bj", pol_diff, torch.linalg.inv(cell)
        )  # (Nbatch, 3)

        # Fold difference into "unit cell" in fractional coordinates
        frac_pol_diff = torch.remainder(frac_pol_diff, 1.0)

        # Apply minimum image convention
        frac_pol_diff = torch.where(
            frac_pol_diff > 0.5, frac_pol_diff - 1.0, frac_pol_diff
        )
        frac_pol_diff = torch.where(
            frac_pol_diff < -0.5, frac_pol_diff + 1.0, frac_pol_diff
        )

        # Map back from fractional to Cartesian
        pol_diff_folded = torch.einsum(
            "bi, bij -> bj", frac_pol_diff, cell
        )  # (Nbatch, 3)

        # Normalize by number of atoms for per-atom polarization (consistent with PerAtomModifier)
        num_atoms_reciprocal = (
            preds[AtomicDataDict.NUM_NODES_KEY].reciprocal().reshape(-1)
        )  # (Nbatch,)
        pol_diff_per_atom = torch.einsum(
            "n..., n -> n...", pol_diff_folded, num_atoms_reciprocal
        )

        # Apply modifier and update using _MeanX pattern
        sample_count = pol_diff_per_atom.numel()
        if sample_count > 0:
            modified_data = self.modifier(pol_diff_per_atom)
            current_mean = (
                self.sum.div(self.count) if torch.is_nonzero(self.count) else 0
            )
            sample_mean = modified_data.mean()
            delta = sample_mean - current_mean
            new_count = self.count + sample_count
            new_mean = current_mean + delta * sample_count / new_count
            self.count = new_count
            self.sum = new_mean * self.count

    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return self.sum.div(self.count)


class FoldedPerAtomPolarizationMSE(_FoldedPolarizationBase):
    """Folded per-atom polarization mean squared error."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.square, **kwargs)

    def __str__(self) -> str:
        return "folded_per_atom_pol_mse"


class FoldedPerAtomPolarizationMAE(_FoldedPolarizationBase):
    """Folded per-atom polarization mean absolute error."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.abs, **kwargs)

    def __str__(self) -> str:
        return "folded_per_atom_pol_mae"


class FoldedPerAtomPolarizationRMSE(FoldedPerAtomPolarizationMSE):
    """Folded per-atom polarization root mean squared error."""

    def compute(self) -> torch.Tensor:
        """Compute final metric value."""
        return torch.sqrt(self.sum.div(self.count))

    def __str__(self) -> str:
        return "folded_per_atom_pol_rmse"

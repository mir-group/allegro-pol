import pytest
import torch
from nequip.data import AtomicDataDict
from nequip.data._keys import POLARIZATION_KEY

from allegro_pol.train.folded_pol_metrics import (
    FoldedPerAtomPolarizationMSE,
    FoldedPerAtomPolarizationMAE,
    FoldedPerAtomPolarizationRMSE,
)

# tolerance constant for float64 precision
_TOL = 1e-10


def create_test_data(pred_pol, target_pol, cell, num_nodes):
    """Helper to create test data dictionaries."""
    dtype = torch.get_default_dtype()
    preds = {
        POLARIZATION_KEY: torch.tensor(pred_pol, dtype=dtype),
        AtomicDataDict.CELL_KEY: torch.tensor(cell, dtype=dtype),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor(num_nodes),
    }
    targets = {
        POLARIZATION_KEY: torch.tensor(target_pol, dtype=dtype),
    }
    return preds, targets


def manual_folding_calculation(pred_pol, target_pol, cell, num_nodes):
    """Manual calculation of folded polarization for validation."""
    dtype = torch.get_default_dtype()
    pred_pol = torch.tensor(pred_pol, dtype=dtype)
    target_pol = torch.tensor(target_pol, dtype=dtype)
    cell = torch.tensor(cell, dtype=dtype)
    num_nodes = torch.tensor(num_nodes)

    # calculate difference
    pol_diff = pred_pol - target_pol

    # convert to fractional coordinates
    frac_pol_diff = torch.einsum("bi, bij -> bj", pol_diff, torch.linalg.inv(cell))

    # apply remainder and minimum image convention
    frac_pol_diff = torch.remainder(frac_pol_diff, 1.0)
    frac_pol_diff = torch.where(frac_pol_diff > 0.5, frac_pol_diff - 1.0, frac_pol_diff)
    frac_pol_diff = torch.where(
        frac_pol_diff < -0.5, frac_pol_diff + 1.0, frac_pol_diff
    )

    # convert back to Cartesian
    pol_diff_folded = torch.einsum("bi, bij -> bj", frac_pol_diff, cell)

    # normalize by number of atoms
    num_atoms_reciprocal = num_nodes.reciprocal().reshape(-1)
    pol_diff_per_atom = torch.einsum(
        "n..., n -> n...", pol_diff_folded, num_atoms_reciprocal
    )

    return pol_diff_per_atom


class TestFoldedPolarizationMetrics:
    """Test suite for folded polarization metrics."""

    @pytest.fixture(
        params=[
            FoldedPerAtomPolarizationMSE,
            FoldedPerAtomPolarizationMAE,
            FoldedPerAtomPolarizationRMSE,
        ]
    )
    def metric_class(self, request):
        """Parametrize over all metric types."""
        return request.param

    @pytest.fixture
    def identity_cell(self):
        """3x3 identity cell matrix."""
        return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]

    @pytest.fixture
    def cubic_cell(self):
        """3x3x3 cubic cell."""
        return [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]]

    @pytest.fixture
    def orthorhombic_cell(self):
        """2x4x6 orthorhombic cell."""
        return [[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]]

    def test_zero_difference(self, metric_class, identity_cell):
        """Test that identical predictions and targets give zero error."""
        metric = metric_class()
        pol = [[1.0, 2.0, 3.0]]
        preds, targets = create_test_data(pol, pol, identity_cell, [1])

        metric.update(preds, targets)
        result = metric.compute()

        assert torch.allclose(result, torch.tensor(0.0), atol=_TOL)

    @pytest.mark.parametrize(
        "pred_pol,target_pol,cell,num_nodes,expected_folded",
        [
            # No folding needed - small differences
            (
                [[0.1, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                [1],
                [[0.1, 0.0, 0.0]],
            ),
            # Positive folding - cubic cell
            (
                [[2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
                [1],
                [[-1.0, 0.0, 0.0]],
            ),
            # Negative folding - cubic cell
            (
                [[-2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
                [1],
                [[1.0, 0.0, 0.0]],
            ),
            # Boundary case - exactly at 0.5 (should stay as 0.5 since > 0.5 is false)
            (
                [[1.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
                [1],
                [[1.5, 0.0, 0.0]],
            ),
            # Multiple components
            (
                [[2.0, 4.0, 6.0]],
                [[0.0, 0.0, 0.0]],
                [[[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]],
                [1],
                [[0.0, 0.0, 0.0]],
            ),
            # Large displacement
            (
                [[10.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
                [1],
                [[1.0, 0.0, 0.0]],
            ),
        ],
    )
    def test_folding_calculations(
        self, pred_pol, target_pol, cell, num_nodes, expected_folded
    ):
        """Test that folding calculations match expected values."""
        folded = manual_folding_calculation(pred_pol, target_pol, cell, num_nodes)
        expected = torch.tensor(expected_folded, dtype=torch.get_default_dtype())
        assert torch.allclose(folded, expected, atol=_TOL)

    @pytest.mark.parametrize(
        "modifier,expected_result",
        [
            (torch.square, 0.0),  # MSE: diff [1,0,0] folds to [0,0,0] in identity cell
            (torch.abs, 0.0),  # MAE: diff [1,0,0] folds to [0,0,0] in identity cell
        ],
    )
    def test_metric_modifiers(self, modifier, expected_result, identity_cell):
        """Test different metric modifiers (square, abs)."""
        from allegro_pol.train.folded_pol_metrics import _FoldedPolarizationBase

        metric = _FoldedPolarizationBase(modifier=modifier)
        pred_pol = [[1.0, 0.0, 0.0]]  # this will fold to [0,0,0] in identity cell
        target_pol = [[0.0, 0.0, 0.0]]

        preds, targets = create_test_data(pred_pol, target_pol, identity_cell, [1])
        metric.update(preds, targets)
        result = metric.compute()

        assert torch.allclose(result, torch.tensor(expected_result), atol=_TOL)

    def test_rmse_vs_mse(self, identity_cell):
        """Test that RMSE is sqrt of MSE."""
        mse_metric = FoldedPerAtomPolarizationMSE()
        rmse_metric = FoldedPerAtomPolarizationRMSE()

        pred_pol = [[0.5, 0.0, 0.0]]  # use non-integer to avoid folding to zero
        target_pol = [[0.0, 0.0, 0.0]]

        preds, targets = create_test_data(pred_pol, target_pol, identity_cell, [1])

        mse_metric.update(preds, targets)
        rmse_metric.update(preds, targets)

        mse_result = mse_metric.compute()
        rmse_result = rmse_metric.compute()

        assert torch.allclose(rmse_result, torch.sqrt(mse_result), atol=_TOL)

    def test_per_atom_normalization(self, identity_cell):
        """Test that results are properly normalized by number of atoms."""
        metric = FoldedPerAtomPolarizationMSE()

        # use non-integer polarization difference that won't fold to zero
        pred_pol = [[0.5, 0.0, 0.0]]  # this won't fold in identity cell
        target_pol = [[0.0, 0.0, 0.0]]

        # test with different atom counts
        for num_atoms in [1, 2, 4, 8]:
            preds, targets = create_test_data(
                pred_pol, target_pol, identity_cell, [num_atoms]
            )
            metric.reset()
            metric.update(preds, targets)
            result = metric.compute()

            # per-atom error should scale as 1/num_atoms^2
            # diff = [0.5, 0, 0], per_atom = [0.5/num_atoms, 0, 0], squared = [0.25/num_atoms^2, 0, 0], mean = 0.25/(3*num_atoms^2)
            expected = 0.25 / (3 * num_atoms**2)
            assert torch.allclose(result, torch.tensor(expected), atol=_TOL)

    def test_batch_accumulation(self, identity_cell):
        """Test that multiple batches accumulate correctly."""
        metric = FoldedPerAtomPolarizationMSE()

        # create multiple batches with known errors (avoid integer differences that fold to 0)
        batches = [
            ([[0.5, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [1]),  # error = 0.25/3
            ([[0.0, 0.5, 0.0]], [[0.0, 0.0, 0.0]], [2]),  # error = 0.25/(3*4) = 0.25/12
            ([[0.0, 0.0, 0.5]], [[0.0, 0.0, 0.0]], [3]),  # error = 0.25/(3*9) = 0.25/27
        ]

        for pred_pol, target_pol, num_nodes in batches:
            preds, targets = create_test_data(
                pred_pol, target_pol, identity_cell, num_nodes
            )
            metric.update(preds, targets)

        result = metric.compute()
        # expected: mean of [0.25/3, 0.25/12, 0.25/27] weighted by sample count (3 samples each)
        expected = (0.25 / 3 + 0.25 / 12 + 0.25 / 27) / 3
        assert torch.allclose(result, torch.tensor(expected), atol=_TOL)

    @pytest.mark.parametrize(
        "cell_matrix",
        [
            # Triclinic cell
            [[[2.0, 1.0, 0.5], [0.0, 2.0, 0.3], [0.0, 0.0, 2.0]]],
            # Hexagonal-like cell
            [[[3.0, 1.5, 0.0], [0.0, 2.598, 0.0], [0.0, 0.0, 4.0]]],
        ],
    )
    def test_non_orthogonal_cells(self, metric_class, cell_matrix):
        """Test that non-orthogonal cells work without errors."""
        metric = metric_class()

        pred_pol = [[1.0, 1.0, 1.0]]
        target_pol = [[0.0, 0.0, 0.0]]

        preds, targets = create_test_data(pred_pol, target_pol, cell_matrix, [1])
        metric.update(preds, targets)
        result = metric.compute()

        # just check that computation completes without error and result is reasonable
        assert torch.isfinite(result)
        assert result >= 0  # all metrics should be non-negative

    def test_missing_cell_raises_error(self, metric_class):
        """Test that missing cell information raises ValueError."""
        metric = metric_class()

        # create data without cell
        preds = {
            POLARIZATION_KEY: torch.tensor([[1.0, 2.0, 3.0]]),
            AtomicDataDict.NUM_NODES_KEY: torch.tensor([1]),
        }
        targets = {
            POLARIZATION_KEY: torch.tensor([[0.0, 0.0, 0.0]]),
        }

        with pytest.raises(ValueError, match="Cell information required"):
            metric.update(preds, targets)

    def test_metric_reset(self, metric_class, identity_cell):
        """Test that metrics can be reset and reused."""
        metric = metric_class()

        pred_pol = [[1.0, 0.0, 0.0]]
        target_pol = [[0.0, 0.0, 0.0]]
        preds, targets = create_test_data(pred_pol, target_pol, identity_cell, [1])

        # first computation
        metric.update(preds, targets)
        result1 = metric.compute()

        # reset and compute again
        metric.reset()
        metric.update(preds, targets)
        result2 = metric.compute()

        assert torch.allclose(result1, result2)

    @pytest.mark.parametrize(
        "metric_class,expected_name",
        [
            (FoldedPerAtomPolarizationMSE, "folded_per_atom_pol_mse"),
            (FoldedPerAtomPolarizationMAE, "folded_per_atom_pol_mae"),
            (FoldedPerAtomPolarizationRMSE, "folded_per_atom_pol_rmse"),
        ],
    )
    def test_string_representations(self, metric_class, expected_name):
        """Test string representations of metrics."""
        metric = metric_class()
        assert str(metric) == expected_name

    def test_random_data_stress(self, metric_class):
        """Stress test with random data to ensure robustness."""
        metric = metric_class()
        torch.manual_seed(42)  # for reproducibility

        for _ in range(10):
            # random data
            batch_size = torch.randint(1, 5, (1,)).item()
            pred_pol = torch.randn(batch_size, 3) * 10  # large range
            target_pol = torch.randn(batch_size, 3) * 10

            # random cell (ensure invertible)
            cell = torch.randn(batch_size, 3, 3)
            cell = (
                cell @ cell.transpose(-1, -2) + torch.eye(3) * 0.1
            )  # make positive definite

            num_nodes = torch.randint(1, 20, (batch_size,))

            preds = {
                POLARIZATION_KEY: pred_pol,
                AtomicDataDict.CELL_KEY: cell,
                AtomicDataDict.NUM_NODES_KEY: num_nodes,
            }
            targets = {
                POLARIZATION_KEY: target_pol,
            }

            metric.reset()
            metric.update(preds, targets)
            result = metric.compute()

            # basic sanity checks
            assert torch.isfinite(result)
            assert result >= 0
            assert not torch.isnan(result)

    def test_numerical_precision_edge_cases(self, metric_class, identity_cell):
        """Test numerical precision at boundaries."""
        metric = metric_class()

        # test very small differences
        pred_pol = [[1e-10, 0.0, 0.0]]
        target_pol = [[0.0, 0.0, 0.0]]

        preds, targets = create_test_data(pred_pol, target_pol, identity_cell, [1])
        metric.update(preds, targets)
        result = metric.compute()

        assert torch.isfinite(result)
        assert result >= 0

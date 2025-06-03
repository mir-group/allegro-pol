# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
from nequip.data import AtomicDataDict, PerAtomModifier
from nequip.train.metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
)
from nequip.train.metrics_manager import MetricsManager

from typing import Dict, Final, List

from .._keys import POLARIZABILITY_KEY
from .folded_pol_metrics import (
    FoldedPerAtomPolarizationMSE,
    FoldedPerAtomPolarizationMAE,
    FoldedPerAtomPolarizationRMSE,
)


_EFP_METRICS_COEFFS_KEYS: Final[List[str]] = [
    "total_energy_rmse",
    "per_atom_energy_rmse",
    "forces_rmse",
    "per_atom_polarization_rmse",
    "total_energy_mae",
    "per_atom_energy_mae",
    "forces_mae",
    "per_atom_polarization_mae",
    "stress_rmse",
    "stress_mae",
    "born_charge_rmse",
    "born_charge_mae",
    "polarizability_rmse",
    "polarizability_mae",
]


def EnergyForecPolarizationLoss(
    coeffs: Dict[str, float] = {
        AtomicDataDict.TOTAL_ENERGY_KEY: 1.0,
        AtomicDataDict.FORCE_KEY: 1.0,
        AtomicDataDict.POLARIZATION_KEY: 1.0,
    },
    per_atom_energy: bool = True,
    do_stress: bool = False,
    do_born_charge: bool = False,
    do_polarizability: bool = False,
    type_names=None,
):
    """Simplified :class:`MetricsManager` wrapper for polarization training loss.

    Base: energy + forces + polarization (with folding)
    Optional: stress, born charges, polarizability

    Args:
        coeffs: Relative weights for each loss component
        per_atom_energy: Whether to normalize total energy by number of atoms
        do_stress: Include stress in loss
        do_born_charge: Include Born charges in loss
        do_polarizability: Include polarizability in loss
        type_names: Atom type names for per-type metrics
    """

    metrics = [
        {
            "name": "per_atom_energy_mse" if per_atom_energy else "total_energy_mse",
            "field": (
                PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY)
                if per_atom_energy
                else AtomicDataDict.TOTAL_ENERGY_KEY
            ),
            "coeff": coeffs[AtomicDataDict.TOTAL_ENERGY_KEY],
            "metric": MeanSquaredError(),
        },
        {
            "name": "forces_mse",
            "field": AtomicDataDict.FORCE_KEY,
            "coeff": coeffs[AtomicDataDict.FORCE_KEY],
            "metric": MeanSquaredError(),
        },
        {
            "name": "per_atom_polarization_mse",
            "field": None,  # Special case - metric takes full AtomicDataDict objects
            "coeff": coeffs[AtomicDataDict.POLARIZATION_KEY],
            "metric": FoldedPerAtomPolarizationMSE(),
        },
    ]

    if do_stress:
        metrics.append(
            {
                "name": "stress_mse",
                "field": AtomicDataDict.STRESS_KEY,
                "coeff": coeffs[AtomicDataDict.STRESS_KEY],
                "metric": MeanSquaredError(),
            }
        )

    if do_born_charge:
        metrics.append(
            {
                "name": "born_charge_mse",
                "field": AtomicDataDict.BORN_CHARGE_KEY,
                "coeff": coeffs[AtomicDataDict.BORN_CHARGE_KEY],
                "metric": MeanSquaredError(),
            }
        )

    if do_polarizability:
        metrics.append(
            {
                "name": "polarizability_mse",
                "field": POLARIZABILITY_KEY,
                "coeff": coeffs[POLARIZABILITY_KEY],
                "metric": MeanSquaredError(),
            }
        )

    return MetricsManager(metrics, type_names=type_names)


def EnergyForecPolarizationMetrics(
    coeffs: Dict[str, float] = {
        "total_energy_rmse": 1.0,
        "per_atom_energy_rmse": None,
        "forces_rmse": 1.0,
        "polarization_rmse": 1.0,
        "total_energy_mae": None,
        "per_atom_energy_mae": None,
        "forces_mae": None,
        "polarization_mae": None,
    },
    do_stress: bool = False,
    do_born_charge: bool = False,
    do_polarizability: bool = False,
    type_names=None,
):
    """Simplified :class:`MetricsManager` wrapper for polarization validation metrics.

    Base: energy + forces + polarization (MAE/RMSE)
    Optional: stress, born charges, polarizability

    Args:
        coeffs: Relative weights for weighted_sum metric
        do_stress: Include stress metrics
        do_born_charge: Include Born charge metrics
        do_polarizability: Include polarizability metrics
        type_names: Atom type names for per-type metrics
    """
    assert all(
        [k in _EFP_METRICS_COEFFS_KEYS for k in coeffs.keys()]
    ), f"Unrecognized key found in `coeffs`, only the following are recognized: {_EFP_METRICS_COEFFS_KEYS}"

    metrics = [
        # Energy metrics
        {
            "name": "total_energy_rmse",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("total_energy_rmse", None),
        },
        {
            "name": "total_energy_mae",
            "field": AtomicDataDict.TOTAL_ENERGY_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("total_energy_mae", None),
        },
        {
            "name": "per_atom_energy_rmse",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("per_atom_energy_rmse", None),
        },
        {
            "name": "per_atom_energy_mae",
            "field": PerAtomModifier(AtomicDataDict.TOTAL_ENERGY_KEY),
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("per_atom_energy_mae", None),
        },
        # Force metrics
        {
            "name": "forces_rmse",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": RootMeanSquaredError(),
            "coeff": coeffs.get("forces_rmse", None),
        },
        {
            "name": "forces_mae",
            "field": AtomicDataDict.FORCE_KEY,
            "metric": MeanAbsoluteError(),
            "coeff": coeffs.get("forces_mae", None),
        },
        # Polarization metrics (with folding)
        {
            "name": "per_atom_polarization_rmse",
            "field": None,  # Special case - metric takes full AtomicDataDict objects
            "metric": FoldedPerAtomPolarizationRMSE(),
            "coeff": coeffs.get("polarization_rmse", None),
        },
        {
            "name": "per_atom_polarization_mae",
            "field": None,  # Special case - metric takes full AtomicDataDict objects
            "metric": FoldedPerAtomPolarizationMAE(),
            "coeff": coeffs.get("polarization_mae", None),
        },
    ]

    if do_stress:
        metrics.extend(
            [
                {
                    "name": "stress_rmse",
                    "field": AtomicDataDict.STRESS_KEY,
                    "metric": RootMeanSquaredError(),
                    "coeff": coeffs.get("stress_rmse", None),
                },
                {
                    "name": "stress_mae",
                    "field": AtomicDataDict.STRESS_KEY,
                    "metric": MeanAbsoluteError(),
                    "coeff": coeffs.get("stress_mae", None),
                },
            ]
        )

    if do_born_charge:
        metrics.extend(
            [
                {
                    "name": "born_charge_rmse",
                    "field": AtomicDataDict.BORN_CHARGE_KEY,
                    "metric": RootMeanSquaredError(),
                    "coeff": coeffs.get("born_charge_rmse", None),
                },
                {
                    "name": "born_charge_mae",
                    "field": AtomicDataDict.BORN_CHARGE_KEY,
                    "metric": MeanAbsoluteError(),
                    "coeff": coeffs.get("born_charge_mae", None),
                },
            ]
        )

    if do_polarizability:
        metrics.extend(
            [
                {
                    "name": "polarizability_rmse",
                    "field": POLARIZABILITY_KEY,
                    "metric": RootMeanSquaredError(),
                    "coeff": coeffs.get("polarizability_rmse", None),
                },
                {
                    "name": "polarizability_mae",
                    "field": POLARIZABILITY_KEY,
                    "metric": MeanAbsoluteError(),
                    "coeff": coeffs.get("polarizability_mae", None),
                },
            ]
        )

    return MetricsManager(metrics, type_names=type_names)

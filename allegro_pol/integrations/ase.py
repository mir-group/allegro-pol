# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
from typing import Dict

from nequip.integrations.ase import NequIPCalculator
from nequip.data import AtomicDataDict
from nequip.scripts._compile_utils import COMPILE_TARGET_DICT

from .._compile import AOTI_ASE_POL_BC_TARGET
from .._keys import POLARIZABILITY_KEY


class NequIPPolCalculator(NequIPCalculator):
    implemented_properties = [
        *NequIPCalculator.implemented_properties,
        AtomicDataDict.POLARIZATION_KEY,
        AtomicDataDict.BORN_CHARGE_KEY,
        POLARIZABILITY_KEY,
    ]

    @classmethod
    def _get_aoti_compile_target(cls) -> Dict:
        return COMPILE_TARGET_DICT[AOTI_ASE_POL_BC_TARGET]

    def save_extra_outputs(self, out: AtomicDataDict.Type):
        if AtomicDataDict.POLARIZATION_KEY in out:
            polarization = out[AtomicDataDict.POLARIZATION_KEY].detach().cpu().numpy()
            if polarization.ndim == 2 and polarization.shape[0] == 1:
                polarization = polarization[0]
            self.results[AtomicDataDict.POLARIZATION_KEY] = polarization

        if AtomicDataDict.BORN_CHARGE_KEY in out:
            self.results[AtomicDataDict.BORN_CHARGE_KEY] = (
                out[AtomicDataDict.BORN_CHARGE_KEY].detach().cpu().numpy()
            )

        if POLARIZABILITY_KEY in out:
            polarizability = out[POLARIZABILITY_KEY].detach().cpu().numpy()
            if polarizability.ndim == 3 and polarizability.shape[0] == 1:
                polarizability = polarizability[0]
            self.results[POLARIZABILITY_KEY] = polarizability

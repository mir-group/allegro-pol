# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
from typing import Dict

import torch

from nequip.data import AtomicDataDict
from nequip.integrations.torchsim import NequIPTorchSimCalc
from nequip.scripts._compile_utils import COMPILE_TARGET_DICT

from .._compile import AOTI_BATCH_POL_BC_TARGET
from .._keys import POLARIZABILITY_KEY


class NequIPPolTorchSimCalc(NequIPTorchSimCalc):
    @classmethod
    def _get_aoti_compile_target(cls) -> Dict:
        return COMPILE_TARGET_DICT[AOTI_BATCH_POL_BC_TARGET]

    def save_extra_outputs(
        self, out: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> None:
        if AtomicDataDict.POLARIZATION_KEY in out:
            results[AtomicDataDict.POLARIZATION_KEY] = out[
                AtomicDataDict.POLARIZATION_KEY
            ].detach()

        if AtomicDataDict.BORN_CHARGE_KEY in out:
            results[AtomicDataDict.BORN_CHARGE_KEY] = out[
                AtomicDataDict.BORN_CHARGE_KEY
            ].detach()

        if POLARIZABILITY_KEY in out:
            results[POLARIZABILITY_KEY] = out[POLARIZABILITY_KEY].detach()

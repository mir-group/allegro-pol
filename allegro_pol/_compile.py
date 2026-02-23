# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
from nequip.data import AtomicDataDict
from nequip.scripts._compile_utils import (
    ASE_OUTPUTS,
    BATCH_INPUTS,
    LMP_OUTPUTS,
    PAIR_NEQUIP_INPUTS,
    batched_data_settings,
    single_frame_data_settings,
    single_frame_batch_map_settings,
    register_compile_targets,
)

from allegro._compile import PAIR_ALLEGRO_INPUTS, allegro_data_settings

from ._keys import POLARIZABILITY_KEY

AOTI_PAIR_ALLEGRO_POL_TARGET = "pair_allegro_pol"
AOTI_PAIR_ALLEGRO_POL_BC_TARGET = "pair_allegro_pol_bc"
AOTI_ASE_POL_BC_TARGET = "ase_pol_bc"
AOTI_BATCH_POL_BC_TARGET = "batch_pol_bc"


PAIR_ALLEGRO_POL_OUTPUTS = [*LMP_OUTPUTS, AtomicDataDict.POLARIZATION_KEY]
PAIR_ALLEGRO_POL_BC_OUTPUTS = [
    *PAIR_ALLEGRO_POL_OUTPUTS,
    AtomicDataDict.BORN_CHARGE_KEY,
    POLARIZABILITY_KEY,
]

PAIR_ALLEGRO_POL_TARGET = {
    "input": PAIR_ALLEGRO_INPUTS,
    "output": PAIR_ALLEGRO_POL_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": allegro_data_settings,
}

PAIR_ALLEGRO_POL_BC_TARGET = {
    "input": PAIR_ALLEGRO_INPUTS,
    "output": PAIR_ALLEGRO_POL_BC_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": allegro_data_settings,
}

ASE_POL_BC_OUTPUTS = [
    *ASE_OUTPUTS,
    AtomicDataDict.POLARIZATION_KEY,
    AtomicDataDict.BORN_CHARGE_KEY,
    POLARIZABILITY_KEY,
]

ASE_POL_BC_TARGET = {
    "input": PAIR_NEQUIP_INPUTS,
    "output": ASE_POL_BC_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}

BATCH_POL_BC_TARGET = {
    "input": BATCH_INPUTS,
    "output": ASE_POL_BC_OUTPUTS,
    "batch_map_settings": lambda batch_map: batch_map,  # no static shapes
    "data_settings": batched_data_settings,
}

register_compile_targets(
    {
        AOTI_PAIR_ALLEGRO_POL_TARGET: PAIR_ALLEGRO_POL_TARGET,
        AOTI_PAIR_ALLEGRO_POL_BC_TARGET: PAIR_ALLEGRO_POL_BC_TARGET,
        AOTI_ASE_POL_BC_TARGET: ASE_POL_BC_TARGET,
        AOTI_BATCH_POL_BC_TARGET: BATCH_POL_BC_TARGET,
    }
)

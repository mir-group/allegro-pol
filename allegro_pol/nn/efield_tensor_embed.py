# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, ScalarMLPFunction, with_edge_vectors_

from allegro.nn._strided import MakeWeightedChannels

from typing import Union

from .. import _keys


@compile_mode("script")
class TwoBodySphericalHarmonicElectricFieldTensorEmbed(
    GraphModuleMixin, torch.nn.Module
):
    """Construct two-body tensor embedding as weighted spherical harmonics with electric field.

    Constructs tensor basis as spherical harmonic projections of edge vectors combined with
    electric field spherical harmonics, and tensor embedding as weighted tensor basis
    (weights learnt from scalar embedding).

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        irreps_elec_field_sh (int, str, or o3.Irreps): irreps for electric field spherical harmonics (default same as edge SH)
        num_tensor_features (int): number of tensor feature channels
        electric_field_normalization (float): normalization factor for electric field (default 1.0)
    """

    num_tensor_features: int
    electric_field_normalization: float

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, Irreps],
        num_tensor_features: int,
        irreps_elec_field_sh: Union[int, str, Irreps] = None,
        electric_field_normalization: float = 1.0,
        forward_weight_init: bool = True,
        # bookkeeping args
        scalar_embedding_in_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=None,
        # optional hyperparameters
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        weight_individual_irreps: bool = True,
    ):
        super().__init__()

        self.scalar_embedding_in_field = scalar_embedding_in_field
        self.tensor_basis_out_field = tensor_basis_out_field
        self.tensor_embedding_out_field = tensor_embedding_out_field
        self.electric_field_normalization = electric_field_normalization

        # Set up edge spherical harmonics
        if isinstance(irreps_edge_sh, int):
            irreps_edge_sh = Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            irreps_edge_sh = Irreps(irreps_edge_sh)

        # Set up electric field spherical harmonics (default to same as edge)
        if irreps_elec_field_sh is None:
            irreps_elec_field_sh = irreps_edge_sh
        elif isinstance(irreps_elec_field_sh, int):
            irreps_elec_field_sh = Irreps.spherical_harmonics(irreps_elec_field_sh)
        else:
            irreps_elec_field_sh = Irreps(irreps_elec_field_sh)

        # Combined irreps for both edge and electric field
        irreps_combined = irreps_edge_sh + irreps_elec_field_sh

        self.irreps_edge_sh = irreps_edge_sh
        self.irreps_elec_field_sh = irreps_elec_field_sh

        # Spherical harmonics modules
        self.sh_edge = SphericalHarmonics(
            irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
        self.sh_elec_field = SphericalHarmonics(
            irreps_elec_field_sh,
            False,
            edge_sh_normalization,  # Don't normalize electric field SH
        )

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.scalar_embedding_in_field],
            irreps_out={
                self.tensor_basis_out_field: irreps_combined,
                self.tensor_embedding_out_field: irreps_combined,
            },
        )

        # use learned weights from two-body scalar track to weight
        # initial spherical harmonics embedding
        self._edge_weighter = MakeWeightedChannels(
            irreps_in=irreps_combined,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )

        # hardcode a linear projection
        self.env_embed_linear = ScalarMLPFunction(
            input_dim=self.irreps_in[self.scalar_embedding_in_field].num_irreps,
            output_dim=self._edge_weighter.weight_numel,
            forward_weight_init=forward_weight_init,
        )
        assert not self.env_embed_linear.is_nonlinear

        self._output_dtype = torch.get_default_dtype()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # sph embedding
        data = with_edge_vectors_(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        edge_sh = self.sh_edge(edge_vec).to(self._output_dtype)  # (Nedge, lm_edge)

        # process electric field
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
            num_batch = AtomicDataDict.num_frames(data)
        else:
            num_batch = 1

        # get electric field and normalize
        if _keys.EXTERNAL_ELECTRIC_FIELD_KEY not in data:
            data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY] = torch.zeros(
                num_batch,
                3,
                dtype=edge_vec.dtype,
                device=edge_vec.device,
            )

        elec_field = data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY].div(
            self.electric_field_normalization
        )  # (Nbatch, 3)
        elec_field_sh_embed = self.sh_elec_field(elec_field).to(
            self._output_dtype
        )  # (Nbatch, lm_field)

        # map electric field sh embedding: (Nbatch, lm) -> (Nedge, lm)
        edge_batch = torch.index_select(
            batch, 0, data[AtomicDataDict.EDGE_INDEX_KEY][0]
        )  # (Nedge,)
        per_edge_field_sh_embed = torch.index_select(
            elec_field_sh_embed, 0, edge_batch
        )  # (Nedge, lm_field)

        # combine edge and electric field spherical harmonics
        combined_sh = torch.cat(
            (edge_sh, per_edge_field_sh_embed), dim=-1
        )  # (Nedge, lm_edge + lm_field)

        # compute weights from scalar embedding
        edge_invariants = data[self.scalar_embedding_in_field]
        weights = self.env_embed_linear(edge_invariants)

        # store unweighted spherical harmonics embedding as two-body tensor basis
        data[self.tensor_basis_out_field] = combined_sh

        # store two-body tensor features (weighted spherical harmonics embedding)
        data[self.tensor_embedding_out_field] = self._edge_weighter(
            combined_sh, weights
        )

        return data

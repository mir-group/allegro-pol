import torch

from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


@compile_mode("script")
class ForceStressPolarizationOutput(GraphModuleMixin, torch.nn.Module):
    """`ForceStressOutput` from NequIP, extended to predict polarization related quantities through autogad."""

    do_born_charge: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        do_born_charge: bool = True,
    ):
        super().__init__()

        self.do_born_charge = do_born_charge
        self.func = func

        # check and init irreps
        irreps_in = self.func.irreps_in.copy()
        irreps_in.pop(_keys.EXTERNAL_ELECTRIC_FIELD_KEY)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=self.func.irreps_out.copy(),
        )
        self.irreps_out[AtomicDataDict.FORCE_KEY] = "1o"
        self.irreps_out[AtomicDataDict.STRESS_KEY] = "1o"
        self.irreps_out[AtomicDataDict.VIRIAL_KEY] = "1o"
        self.irreps_out[AtomicDataDict.POLARIZATION_KEY] = "1o"
        if self.do_born_charge:
            self.irreps_out[AtomicDataDict.BORN_CHARGE_KEY] = "1o"
            self.irreps_out[_keys.POLARIZABILITY_KEY] = "1o"

        # for torchscript compat
        self.register_buffer("_empty", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        assert AtomicDataDict.EDGE_VECTORS_KEY not in data

        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
            num_batch: int = AtomicDataDict.num_frames(data)
        else:
            # Special case for efficiency
            batch = self._empty
            num_batch: int = 1

        pos = data[AtomicDataDict.POSITIONS_KEY]

        has_cell: bool = AtomicDataDict.CELL_KEY in data

        if has_cell:
            orig_cell = data[AtomicDataDict.CELL_KEY]
            # Make the cell per-batch
            cell = orig_cell.view(-1, 3, 3).expand(num_batch, 3, 3)
            data[AtomicDataDict.CELL_KEY] = cell
        else:
            # torchscript
            orig_cell = self._empty
            cell = self._empty
        # Add the displacements
        # the GradientOutput will make them require grad
        # See SchNetPack code:
        # https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/atomistic/model.py#L45
        # SchNetPack issue:
        # https://github.com/atomistic-machine-learning/schnetpack/issues/165
        # Paper they worked from:
        # Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
        # https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content
        if num_batch > 1:
            displacement = torch.zeros(
                (num_batch, 3, 3),
                dtype=pos.dtype,
                device=pos.device,
            )
        else:
            displacement = torch.zeros(
                (3, 3),
                dtype=pos.dtype,
                device=pos.device,
            )
        displacement.requires_grad_(True)
        data["_displacement"] = displacement
        # in the above paper, the infinitesimal distortion is *symmetric*
        # so we symmetrize the displacement before applying it to
        # the positions/cell
        # This is not strictly necessary (reasoning thanks to Mario):
        # the displacement's asymmetric 1o term corresponds to an
        # infinitesimal rotation, which should not affect the final
        # output (invariance).
        # That said, due to numerical error, this will never be
        # exactly true. So, we symmetrize the deformation to
        # take advantage of this understanding and not rely on
        # the invariance here:
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        did_pos_req_grad: bool = pos.requires_grad
        pos.requires_grad_(True)
        if num_batch > 1:
            # bmm is natom in batch
            # batched [natom, 1, 3] @ [natom, 3, 3] -> [natom, 1, 3] -> [natom, 3]
            data[AtomicDataDict.POSITIONS_KEY] = pos + torch.bmm(
                pos.unsqueeze(-2), torch.index_select(symmetric_displacement, 0, batch)
            ).squeeze(-2)
        else:
            # [natom, 3] @ [3, 3] -> [natom, 3]
            data[AtomicDataDict.POSITIONS_KEY] = torch.addmm(
                pos, pos, symmetric_displacement
            )
        # assert torch.equal(pos, data[AtomicDataDict.POSITIONS_KEY])
        # we only displace the cell if we have one:
        if has_cell:
            # bmm is num_batch in batch
            # here we apply the distortion to the cell as well
            # this is critical also for the correctness
            # if we didn't symmetrize the distortion, since without this
            # there would then be an infinitesimal rotation of the positions
            # but not cell, and it thus wouldn't be global and have
            # no effect due to equivariance/invariance.
            if num_batch > 1:
                # [n_batch, 3, 3] @ [n_batch, 3, 3]
                data[AtomicDataDict.CELL_KEY] = cell + torch.bmm(
                    cell, symmetric_displacement
                )
            else:
                # [3, 3] @ [3, 3] --- enforced to these shapes
                tmpcell = cell.squeeze(0)
                data[AtomicDataDict.CELL_KEY] = torch.addmm(
                    tmpcell, tmpcell, symmetric_displacement
                ).unsqueeze(0)
            # assert torch.equal(cell, data[AtomicDataDict.CELL_KEY])

        # For polarization:
        if _keys.EXTERNAL_ELECTRIC_FIELD_KEY not in data:
            data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY] = torch.zeros(
                num_batch,
                3,
                dtype=pos.dtype,
                device=pos.device,
            )
        assert data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY].shape == (num_batch, 3)

        compute_custom = bool(
            data.get("compute_custom_output", torch.tensor(True)).item()
        )
        # two booleans to guide behavior
        # - compute_custom: determined during MD
        # - do_born_charge: determined during training and deployment
        # if compute_custom, compute polarization at least
        # subsequently compute born charges depending on flag

        # at least polarization is computed via autograd if compute_custom is True
        data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY].requires_grad_(compute_custom)

        # Call model and get gradients
        data = self.func(data)

        grad_vars = [pos, data["_displacement"]]
        if compute_custom:
            grad_vars.append(data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY])

        grads = torch.autograd.grad(
            [data[AtomicDataDict.TOTAL_ENERGY_KEY].sum()],
            grad_vars,
            create_graph=self.training or (compute_custom and self.do_born_charge),
            # ^ needed to allow gradients of this output during training or when taking another gradient for the Born charge
        )

        # Put negative sign on forces
        forces = grads[0]
        if forces is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute forces autograd"
        forces = torch.neg(forces)
        data[AtomicDataDict.FORCE_KEY] = forces

        # Store virial
        virial = grads[1]
        if virial is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute virial autograd"
        virial = virial.view(num_batch, 3, 3)

        # we only compute the stress (1/V * virial) if we have a cell whose volume we can compute
        if has_cell:
            # ^ can only scale by cell volume if we have one...:
            # Rescale stress tensor
            # See https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/atomistic/output_modules.py#L180
            # See also https://en.wikipedia.org/wiki/Triple_product
            # See also https://gitlab.com/ase/ase/-/blob/master/ase/cell.py,
            #          which uses np.abs(np.linalg.det(cell))
            # First dim is batch, second is vec, third is xyz
            # Note the .abs(), since volume should always be positive
            # det is equal to a dot (b cross c)
            volume = torch.linalg.det(cell).abs().unsqueeze(-1)
            stress = virial / volume.view(num_batch, 1, 1)
            data[AtomicDataDict.CELL_KEY] = orig_cell
        else:
            stress = self._empty  # torchscript
        data[AtomicDataDict.STRESS_KEY] = stress

        # see discussion in https://github.com/libAtoms/QUIP/issues/227 about sign convention
        # they say the standard convention is virial = -stress x volume
        # looking above this means that we need to pick up another negative sign for the virial
        # to fit this equation with the stress computed above
        virial = torch.neg(virial)
        data[AtomicDataDict.VIRIAL_KEY] = virial

        # polarization
        if compute_custom:
            polarization = grads[2]  # dE / d electric field, [n_graph, 3]
            if polarization is None:
                # condition needed to unwrap optional for torchscript
                assert False, "failed to compute polarization autograd"
            polarization = torch.neg(polarization)  # P = - dE/d field
        else:
            polarization = torch.zeros((num_batch, 3), device=pos.device)
            born_charges = torch.zeros((pos.shape[0], 3, 3), device=pos.device)

        if compute_custom and self.do_born_charge:
            # use batched: https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad
            dPdr = []
            dPdE = []
            for i in range(3):
                gradPi = torch.autograd.grad(
                    [polarization.sum(dim=0)[i]],
                    [
                        pos,
                        data[_keys.EXTERNAL_ELECTRIC_FIELD_KEY],
                    ],  # for both born charges and polarizabilities
                    # [pos],  # previously for born charges only (no polarizability)
                    create_graph=self.training,  # needed to allow gradients of this output during training
                    retain_graph=(i != 2 or self.training),
                )
                # collect Born charges
                dPidr = gradPi[0]
                assert dPidr is not None
                dPdr.append(dPidr)
                # collect polarizabilities
                dPidE = gradPi[1]  # [n_graph, 3]
                assert dPidE is not None
                dPdE.append(dPidE)

            born_charges = torch.cat(dPdr).reshape(
                (3,) + pos.shape
            )  # [3 (pol), N_atom, 3 (xyz pos)]
            born_charges = born_charges.transpose(
                1, 0
            )  # [N_atom, 3 (pol), 3 (xyz pos)]
            polarizability = torch.cat(dPdE, dim=1).reshape(
                (num_batch, 3, 3)
            )  # [n_graph, 3, 3]
        else:
            born_charges = torch.zeros((pos.shape[0], 3, 3), device=pos.device)
            polarizability = torch.zeros((num_batch, 3, 3), device=pos.device)

        assert polarization is not None
        assert born_charges is not None
        assert polarizability is not None

        # Remove helper
        del data["_displacement"]
        if not did_pos_req_grad:
            # don't give later modules one that does
            pos.requires_grad_(False)

        data[AtomicDataDict.POLARIZATION_KEY] = polarization
        data[AtomicDataDict.BORN_CHARGE_KEY] = born_charges
        data[_keys.POLARIZABILITY_KEY] = polarizability

        return data

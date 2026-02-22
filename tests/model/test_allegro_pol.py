# This file is a part of the `allegro-pol` package. Please see LICENSE and README at the root for information on using it.
import pytest
from nequip.utils.unittests.model_tests_compilation import CompilationTestsMixin
from nequip.utils.versions import _TORCH_GE_2_6

_CUEQ_INSTALLED = False

if _TORCH_GE_2_6:
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401

        _CUEQ_INSTALLED = True
    except Exception:
        pass

BESSEL_CONFIG = {
    "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
    "num_bessels": 8,
}

COMMON_CONFIG = {
    "_target_": "allegro_pol.model.AllegroPolarizationModel",
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 20.0,
    "radial_chemical_embed": BESSEL_CONFIG,
    "radial_chemical_embed_dim": 16,
    "scalar_embed_mlp_hidden_layers_depth": 1,
    "scalar_embed_mlp_hidden_layers_width": 32,
    "num_layers": 2,
    "l_max": 2,
    "parity": True,
    "num_scalar_features": 32,
    "num_tensor_features": 4,
    "allegro_mlp_hidden_layers_depth": 2,
    "allegro_mlp_hidden_layers_width": 32,
    "readout_mlp_hidden_layers_depth": 1,
    "readout_mlp_hidden_layers_width": 8,
    "do_born_charge": True,
}

minimal_config0 = dict(
    **COMMON_CONFIG,
)

minimal_config1 = dict(
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)


class TestAllegroPol(CompilationTestsMixin):
    """Test suite for Allegro Polarization models"""

    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.fixture(scope="class")
    def nequip_compile_tol(self, model_dtype):
        return {"float32": 5e-5, "float64": 1e-10}[model_dtype]

    @pytest.fixture(scope="class")
    def ase_calculator_cls(self):
        from allegro_pol.integrations.ase import NequIPPolCalculator

        return NequIPPolCalculator

    @pytest.fixture(scope="class")
    def ase_aoti_compile_target(self):
        from allegro_pol._compile import AOTI_ASE_POL_BC_TARGET

        return AOTI_ASE_POL_BC_TARGET

    @pytest.fixture(
        scope="class",
        params=[None]
        + (
            ["enable_CuEquivarianceContracter"]
            if _TORCH_GE_2_6 and _CUEQ_INSTALLED
            else []
        ),
    )
    def nequip_compile_acceleration_modifiers(self, request):
        if request.param is None:
            return None

        def modifier_handler(mode, device, model_dtype):
            if request.param == "enable_CuEquivarianceContracter":
                if device == "cpu":
                    pytest.skip("CuEquivarianceContracter tests skipped for CPU")

                if mode == "aotinductor" and model_dtype == "float64":
                    pytest.skip(
                        "CuEquivarianceContracter tests skipped for AOTI and float64 due to known issue"
                    )

                return ["enable_CuEquivarianceContracter"]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        scope="class",
        params=[None]
        + (
            ["enable_CuEquivarianceContracter"]
            if _TORCH_GE_2_6 and _CUEQ_INSTALLED
            else []
        ),
    )
    def train_time_compile_acceleration_modifiers(self, request):
        if request.param is None:
            return None

        def modifier_handler(device):
            if request.param == "enable_CuEquivarianceContracter":
                if device == "cpu":
                    pytest.skip("CuEquivarianceContracter tests skipped for CPU")

                return [{"modifier": "enable_CuEquivarianceContracter"}]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        params=[True, False],
        scope="class",
    )
    def tp_path_channel_coupling(self, request):
        return request.param

    @pytest.fixture(
        params=[
            minimal_config0,
            minimal_config1,
        ],
        scope="class",
    )
    def config(
        self,
        request,
        tp_path_channel_coupling,
    ):
        config = request.param
        config = config.copy()
        config.update({"tp_path_channel_coupling": tp_path_channel_coupling})
        return config

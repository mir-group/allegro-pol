from nequip.data import register_fields, ABBREV

from typing import Final

# Define keys that are not in base nequip
EXTERNAL_ELECTRIC_FIELD_KEY: Final[str] = "external_electric_field"
POLARIZABILITY_KEY: Final[str] = "polarizability"

# Register only the new fields not already in nequip
register_fields(
    graph_fields=[EXTERNAL_ELECTRIC_FIELD_KEY, POLARIZABILITY_KEY],
    cartesian_tensor_fields={
        POLARIZABILITY_KEY: "ij=ji",
    },
)

# Update ABBREV for the new keys only
ABBREV.update({EXTERNAL_ELECTRIC_FIELD_KEY: "E_ext", POLARIZABILITY_KEY: "α"})

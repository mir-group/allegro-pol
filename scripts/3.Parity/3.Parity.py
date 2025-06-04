#!/usr/bin/env python3

# ===========
# PARITY PLOT
# ===========

"""
This script produces parity plots comparing DFT and ML predictions for various
physical quantities. The script reads two extended XYZ files:
1. DFT reference data
2. ML model predictions

Considered quantities:
- Energy per atom
- Atomic positions
- Atomic forces
- Polarization
- Born charges
- Polarizability

For each quantity, the script generates a parity plot showing the agreement
between DFT and ML predictions, along with MAE and RMSE metrics.

How to run:
• SiO2:   python3 parity_plot.py SiO2
• BaTiO3: python3 parity_plot.py BaTiO3

Before using this script, make sure your folders are organized as follows:
Nomenclature for input folder: {system} contains {system}.xyz and {system}-ML.xyz,
where {system} is either SiO2 or BaTiO3.

Units in the extxyz files:
• Energy in eV
• Forces in eV/A
• Coordinates in A
• Stress in eV/A**2
• Polarization in e*A
• Born charges in e
• Polarizabilities in e*A^2*V^{-1}

Author: S. Falletta
"""

# Libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Material configurations
MATERIAL_CONFIGS = {
    "SiO2": {
        "dft_file": "SiO2/SiO2.xyz",
        "ml_file": "SiO2/SiO2-ML.xyz",
        "output_file": "SiO2/SiO2.pdf",
    },
    "BaTiO3": {
        "dft_file": "BaTiO3/BaTiO3.xyz",
        "ml_file": "BaTiO3/BaTiO3-ML.xyz",
        "output_file": "BaTiO3/BaTiO3.pdf",
    },
}

# Plot settings
plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)


def axis_settings(ax):
    """Configure axis settings for plots."""
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_ticks_position("both")


class extxyz:
    """Class for parsing and handling extended XYZ files."""

    def split(self, string, chr):
        """Split string by character and join with spaces."""
        return " ".join(string.split(chr))

    def __init__(self, filename):
        """Initialize extxyz object and parse file.

        Parameters
        ----------
        filename : str
            Path to the extended XYZ file
        """
        print(f"Parsing file {filename}")

        with open(filename, "r") as f:
            data = f.readlines()
        self.nat = int(data[0].split()[0])
        nframes = int(len(data) / (self.nat + 2))

        # Initialize arrays for physical quantities
        self.E = np.zeros(nframes)
        self.R = np.zeros((nframes, self.nat, 3))
        self.F = np.zeros((nframes, self.nat, 3))
        self.S = np.zeros((nframes, 3, 3))
        self.P = np.zeros((nframes, 3))
        self.q = np.zeros((nframes, self.nat, 3, 3))
        self.α = np.zeros((nframes, 3, 3))
        self.p = np.zeros((nframes, 3, 3))
        self.g = np.zeros((nframes, 3, 3))
        self.at = np.zeros(self.nat)

        for frame in range(nframes):
            # Parse frame header
            L = data[frame * (self.nat + 2) + 1]
            L = self.split(L, "=")
            L = self.split(L, " ")
            L = self.split(L, '"')
            L = L.split()

            # Get property indices
            idx_latt = L.index("Lattice") + 1
            if "energy" in L:
                idx_E = L.index("energy") + 1
            elif "total_energy" in L:
                idx_E = L.index("total_energy") + 1
            idx_S = L.index("stress") + 1
            idx_P = L.index("polarization") + 1
            idx_α = L.index("polarizability") + 1

            # Parse per-frame quantities
            self.p[frame, :, :] = np.array(
                [float(x) for x in L[idx_latt : idx_latt + 9]]
            ).reshape(3, 3)
            self.g[frame, 0, :] = self.p[frame, 0, :] / np.linalg.norm(
                self.p[frame, 0, :]
            )
            self.g[frame, 1, :] = self.p[frame, 1, :] / np.linalg.norm(
                self.p[frame, 1, :]
            )
            self.g[frame, 2, :] = self.p[frame, 2, :] / np.linalg.norm(
                self.p[frame, 2, :]
            )
            self.E[frame] = float(L[idx_E])
            self.S[frame, :, :] = np.array(
                [float(x) for x in L[idx_S : idx_S + 9]]
            ).reshape(3, 3)
            self.P[frame, :] = np.array([float(x) for x in L[idx_P : idx_P + 3]])
            self.α[frame, :, :] = np.array(
                [float(x) for x in L[idx_α : idx_α + 9]]
            ).reshape(3, 3)

            # Apply modulo polarization
            self.P[frame, :] = self.unit_pol(
                self.P[frame, :], self.g[frame, :, :], self.p[frame, :, :]
            )

            # Parse per-atom quantities
            for iat in range(self.nat):
                L = data[frame * (self.nat + 2) + 2 + iat].split()
                self.R[frame, iat, :] = np.array([float(x) for x in L[1:4]])
                self.F[frame, iat, :] = np.array([float(x) for x in L[4:7]])
                self.q[frame, iat, :, :] = np.array(
                    [float(x) for x in L[4 : 4 + 9]]
                ).reshape(3, 3)
                self.F[frame, iat, :] = np.array([float(x) for x in L[13 : 13 + 3]])

    def unit_pol(self, P, g, p):
        """Process polarization vector to handle periodic boundary conditions.

        Parameters
        ----------
        P : np.ndarray
            Polarization vector
        g : np.ndarray
            Metric tensor
        p : np.ndarray
            Lattice vectors

        Returns
        -------
        np.ndarray
            Processed polarization vector
        """
        pol_mod_frac = np.dot(np.linalg.inv(g), p).diagonal()
        P_frac = np.dot(g, P)
        Pnew = P_frac % (np.sign(P_frac) * pol_mod_frac)
        Pnew = np.where(Pnew > 0.5 * pol_mod_frac, Pnew - pol_mod_frac, Pnew)
        Pnew = np.where(Pnew < -0.5 * pol_mod_frac, Pnew + pol_mod_frac, Pnew)
        Pnew = np.dot(np.linalg.inv(g), Pnew)
        return Pnew


def parity_plot(y1, y2, title, type, pdf):
    """Generate parity plot comparing two sets of values.

    Parameters
    ----------
    y1 : np.ndarray
        Reference values (DFT)
    y2 : np.ndarray
        Predicted values (ML)
    title : str
        Plot title
    type : int
        Plot type:
        0: scatter plot
        1: density plot with gaussian filter
        2: smoothed contour plot
    pdf : PdfPages
        PDF file to save the plot
    """
    plt.figure(figsize=(6, 6), dpi=60)
    plt.rcParams.update({"font.size": 24})
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.title(title)
    plt.xlabel("DFT")
    plt.ylabel("ML")

    mae = mean_absolute_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))

    if type == 0:
        plt.scatter(y1, y2, s=10)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "-", alpha=0.75, zorder=0, c="r", lw=2)
        plt.scatter(y1, y2, color="blue", alpha=0.6)

    if type == 1:
        N_bins = 100
        colors = [(1, 1, 1)]
        colors.extend(plt.cm.jet(i) for i in range(plt.cm.jet.N))
        cmap = LinearSegmentedColormap.from_list(
            "custom_jet_white", colors, plt.cm.jet.N
        )
        hist, xedges, yedges = np.histogram2d(y1.ravel(), y2.ravel(), bins=N_bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(
            hist.T,
            extent=extent,
            origin="lower",
            cmap=cmap,
            interpolation="gaussian",
            aspect="auto",
            vmin=0,
            vmax=np.max(hist),
        )
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, ":", alpha=0.30, c="r", lw=2)
        plt.colorbar()

    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    textstr = "\n".join(
        (r"mae  = %.4f" % (round(mae, 4),), r"rmse = %.4f" % (round(rmse, 4),))
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.28, 0.82, textstr, transform=ax.transAxes, bbox=props, fontsize=16)

    pdf.savefig()


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Specify Material"

    system = sys.argv[1]

    if system not in MATERIAL_CONFIGS:
        print("Material not implemented")
        sys.exit(1)

    config = MATERIAL_CONFIGS[system]
    S1 = extxyz(config["dft_file"])
    S2 = extxyz(config["ml_file"])

    with PdfPages(config["output_file"]) as pdf:
        d = {0: "x", 1: "y", 2: "z"}

        # Energy
        parity_plot(S1.E / S1.nat, S2.E / S2.nat, "$U/N$", type=1, pdf=pdf)

        # Positions
        for i in range(3):
            parity_plot(S1.R[:, :, i], S2.R[:, :, i], f"$R_{d[i]}$", type=0, pdf=pdf)

        # Forces
        for i in range(3):
            parity_plot(S1.F[:, :, i], S2.F[:, :, i], f"$F_{d[i]}$", type=1, pdf=pdf)

        # Polarization
        for i in range(3):
            parity_plot(S1.P[:, i], S2.P[:, i], f"$P_{d[i]}$", type=1, pdf=pdf)

        # Born charges
        for i in range(3):
            for j in range(3):
                parity_plot(
                    S1.q[:, :, i, j],
                    S2.q[:, :, i, j],
                    f"$Z_{{{d[i]}{d[j]}}}$",
                    type=1,
                    pdf=pdf,
                )

        # Polarizability
        for i in range(3):
            for j in range(3):
                parity_plot(
                    S1.α[:, i, j],
                    S2.α[:, i, j],
                    f"$α_{{{d[i]}{d[j]}}}$",
                    type=1,
                    pdf=pdf,
                )

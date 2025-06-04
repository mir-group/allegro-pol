#!/usr/bin/env python3

# ===========================
# STATIC DIELECTRIC CONSTANT
# ===========================

"""
This script parses and analyzes dielectric constant calculations from:
1. QuantumEspresso output files of structural relaxations under electric fields
2. LAMMPS output files of structural relaxations using the pair-allegro interface,
   including polarization and polarizability treatment

The script plots the dielectric constant throughout the relaxation process,
comparing DFT and ML results.

How to run:
    python3 dielectric_constant.py SiO2

The input data needs to be specified in the MATERIAL_CONFIGS dictionary at the
bottom of this file.

Units in the input files:
    - QE:
        * Energy in Hartree
        * Volume in Bohr^3
        * Electric field in Ry a.u.
        * Polarization in Ry a.u.
    - LAMMPS:
        * Time in ps
        * Volume in A^3
        * Polarization in e*A
        * Electric field in V/A

Note: Currently only valid for orthorhombic cells.

Author: S. Falletta
"""

# Libraries
import os
import subprocess
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

matplotlib.use("Agg")

# Plot settings
plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)

# Colors for plots
COLORS = {"dft": "#d40000", "ml": "#0055d4"}

# Constants and unit conversions
CONSTANTS = {
    "bohr2A": 0.529177249,
    "hartree2eV": 27.211396641308,
    "eps0const": 5.5263499562e-3,  # in [e * Volt^{-1} * Ansgrom^{-1}]
    "Punit": None,  # Will be set after bohr2A is defined
    "Efieldunit": None,  # Will be set after other constants are defined
    "Vunit": None,  # Will be set after bohr2A is defined
}

# Initialize derived constants
CONSTANTS["Punit"] = CONSTANTS["bohr2A"] / np.sqrt(2.0)
CONSTANTS["Efieldunit"] = CONSTANTS["hartree2eV"] / (CONSTANTS["bohr2A"] * np.sqrt(2))
CONSTANTS["Vunit"] = CONSTANTS["bohr2A"] ** 3

# Material configurations
MATERIAL_CONFIGS = {
    "SiO2": {
        "data_DFT_E0": "SiO2/DFT/SiO2-E0.out",
        "data_DFT_Ez": "SiO2/DFT/SiO2-Ez-1e-3.out",
        "data_ML": "SiO2/ML/SiO2-sc222_1e-3.dat",
        "efield_direction": 2,  # 0:x, 1:y, 2:z
        "ml_truncate_index": 21,  # index at which to truncate the ML results
    }
}


def axis_settings(ax):
    """Configure axis settings for consistent plotting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to configure
    """
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_ticks_position("both")


def plot_epsilon(P, P0, E, V, xlabel, title, col, pdf):
    """Plot dielectric constant values throughout the relaxation process.

    Parameters
    ----------
    P : list of numpy.ndarray
        Polarization along z in the presence of the field
    P0 : list of numpy.ndarray
        Polarization along z in the absence of the field
    E : list of float
        Electric field along z
    V : list of float
        Volume
    xlabel : str
        Label for x-axis
    title : str
        Plot title
    col : str
        Color for the plot
    pdf : matplotlib.backends.backend_pdf.PdfPages
        PdfPages object to save the figure
    """
    N = len(P)
    plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(xlabel)
    plt.ylabel("ε")
    plt.title(title)

    for i in range(N):
        t = range(len(P[i]))
        eps = np.array(
            [
                1 + (P[i][n] - P0[i][0]) / (E[i] * CONSTANTS["eps0const"] * V[i])
                for n in t
            ]
        )
        plt.scatter(t, eps, lw=5, c=col)
        plt.plot(t, eps, lw=2.5, c=col)
        plt.scatter(t[0], eps[0], c="C2", s=125, zorder=100)
        plt.scatter(t[-1], eps[-1], c="C2", s=125, zorder=100)

        eps_inf = round(eps[0], 2)
        eps_0 = round(eps[-1], 2)
        ax.text(
            t[-1] * 0.2,
            eps[0] * 1.0,
            f"ε$_∞$ = {eps_inf}",
            ha="center",
            va="center",
            color="C2",
            fontsize=20,
        )
        ax.text(
            t[-1] * 0.8,
            eps[-1] * 0.95,
            f"ε$_0$ = {eps_0}",
            ha="center",
            va="center",
            color="C2",
            fontsize=20,
        )
    pdf.savefig()


class LAMMPS_outfile:
    """Parse polarization values during the LAMMPS relaxation.

    Units in the LAMMPS file:
    * time in ps
    * volume in A**3
    * polarization (in e*A)
    Valid only for orthorombic cells

    Parameters
    ----------
    infile : str
        Path to the LAMMPS output file
    zerofield : bool, optional
        Whether to set electric field to zero, by default False
    """

    def __init__(self, infile, zerofield=False):
        if not os.path.exists(infile):
            raise FileNotFoundError(f"The file '{infile}' does not exist.")

        # general info
        self.system = infile.split(".dat")[0]
        print("• " + self.system)

        # Get cell dimensions
        cmd = "grep -A 1 'Replication is creating a 1x1x1' " + infile
        cmd += " | tail -1"
        data = subprocess.check_output(cmd, shell=True, text=True).split(" ")
        self.A = float(data[9].split("(")[1])
        self.B = float(data[10])
        self.C = float(data[11].split(")")[0])

        # Get electric field
        cmd = "grep 'fix born all addbornforce' " + infile + " | tail -1"
        data = subprocess.check_output(cmd, shell=True, text=True).split()[-3:]
        self.Efield = np.array([float(x) for x in data])
        if zerofield:
            self.Efield = np.zeros(3)

        self.volume = self.A * self.B * self.C

        # Get number of atoms
        cmd = "grep 'atoms' " + infile + " | head -2 | tail -1"
        self.nat = int(
            subprocess.check_output(cmd, shell=True, text=True)
            .splitlines()[0]
            .split()[0]
        )

        # Polarization
        cmd = "grep -A 100000000 'Step          Time' " + infile
        data = subprocess.check_output(cmd, shell=True, text=True).splitlines()
        data_vals = []
        for line in data:
            if "Step" in line:
                continue
            if "Loop time" in line:
                break
            data_vals.append(line)

        self.nframes = len(data_vals)
        self.time = np.zeros(self.nframes)
        self.P = np.zeros((self.nframes, 3))

        for i in range(self.nframes):
            self.time[i] = float(data_vals[i].split()[1])
            self.P[i, :] = np.array([float(x) for x in data_vals[i].split()[6:9]])
            # add contributions due to polarizability if the field is non zero
            if np.abs(np.sum(self.Efield)) > 0:
                alpha = np.array(
                    [float(x) for x in data_vals[i].split()[12 : 12 + 9]]
                ).reshape(3, 3)
                self.P[i, :] += np.dot(self.Efield, alpha)


class QE_outfile:
    """Parse polarization values during the DFT relaxation.

    Parameters
    ----------
    infile : str
        Path to the QuantumEspresso output file
    """

    def __init__(self, infile):
        # general info
        self.system = infile.split(".out")[0]
        print("• " + self.system)

        with open(infile, "r") as file:
            data = file.read()
        self.nframes = data.count("End of self-consistent calculation")

        # Get volume
        cmd = "grep 'unit-cell volume' " + infile
        self.volume = (
            float(subprocess.check_output(cmd, shell=True, text=True).split()[3])
            * CONSTANTS["Vunit"]
        )

        # Get electric field
        string = "In a.u.(Ry)  cartesian system of reference"
        cmd = "grep -A 3 '" + string + "' " + infile
        Efield = subprocess.check_output(cmd, shell=True, text=True).splitlines()[1:]
        self.Efield = np.array([float(x) for x in Efield]) * CONSTANTS["Efieldunit"]

        # Polarization
        self.P = np.zeros((self.nframes, 3))
        string = "End of self-consistent calculation"
        cmd = "grep -B 10 '" + string + "' " + infile
        data = subprocess.check_output(cmd, shell=True, text=True).splitlines()

        for n in range(self.nframes):
            P_el = (
                np.array([float(data[j + n * 12].split()[1]) for j in range(3)])
                * CONSTANTS["Punit"]
            )
            P_ion = (
                np.array([float(data[j + n * 12].split()[1]) for j in range(4, 7)])
                * CONSTANTS["Punit"]
            )
            self.P[n, :] = P_ion + P_el


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 dielectric_constant.py <material>")
        sys.exit(1)

    system = sys.argv[1]
    if system not in MATERIAL_CONFIGS:
        print(f"Material '{system}' not implemented")
        sys.exit(1)

    config = MATERIAL_CONFIGS[system]

    # Parse files
    S0_DFT = QE_outfile(config["data_DFT_E0"])
    Sz_DFT = QE_outfile(config["data_DFT_Ez"])
    S0_ML = LAMMPS_outfile(config["data_ML"], zerofield=True)
    Sz_ML = LAMMPS_outfile(config["data_ML"])

    # Create output directory if it doesn't exist
    os.makedirs(system, exist_ok=True)

    with PdfPages(f"{system}/{system}.pdf") as pdf:
        # Dielectric constant from QE trajectory
        plot_epsilon(
            [Sz_DFT.P[:, config["efield_direction"]]],
            [S0_DFT.P[:, config["efield_direction"]]],
            [Sz_DFT.Efield[config["efield_direction"]]],
            [Sz_DFT.volume],
            "relax steps",
            "DFT",
            COLORS["dft"],
            pdf,
        )

        # Dielectric constant from LAMMPS trajectory
        ml_idx = config["ml_truncate_index"]
        ml_dir = config["efield_direction"]
        plot_epsilon(
            [Sz_ML.P[:ml_idx, ml_dir]],
            [S0_ML.P[:ml_idx, ml_dir]],
            [Sz_ML.Efield[ml_dir]],
            [Sz_ML.volume],
            "relax steps",
            "LAMMPS",
            COLORS["ml"],
            pdf,
        )

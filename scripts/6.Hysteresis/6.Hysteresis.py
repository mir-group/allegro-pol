# =================
# HYSTERESIS LAMMPS
# =================

"""
This script parses the LAMMPS output file from an MLMD simulation at fixed volume
(orthorhombic cells only) and temperature under a sinusoidal electric field.
The simulation is performed using the pair-allegro interface, which includes
the treatment of polarization, Born charges, and polarizability. When provided
with multiple files, as in the case of finite-temperature MD, the polarization
is averaged over all trajectories.

The script plots:
• Polarization hysteresis vs electric field
• Polarization vs time
• Electric field vs time

How to run: python3 hysteresis.py BaTiO3-8640-T300

The input data must be specified at the bottom of this file.

Before using this script, ensure your folders are organized as in the provided
example.

TODO: Generalize to non-orthorhombic cells.

Author: S. Falletta
"""

# Libraries
import os
import sys
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

# Plot settings
plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 14, "legend.handlelength": 0.5}
)

# Colors
c_ml = ["#0055d4", "#d40000"]

# Constants
bohr2A = 0.529177249
A2bohr = 1.8897259886
hartree2eV = 27.211396641308
e2C = 1.602176634e-19
A2m = 1e-10
m2cm = 1e2
C2μC = 1e6
V2MV = 1e-6

# Material configurations
MATERIAL_CONFIG = {
    "BaTiO3-8640-T0": {"infiles": ["BaTiO3-8640-T0.dat"], "plot_time_ev": False},
    "BaTiO3-8640-T300": {
        "infiles": [f"BaTiO3-{i}.dat" for i in range(1, 11)],
        "plot_time_ev": True,
    },
}


def axis_settings(ax):
    """Configure axis settings for plots."""
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_ticks_position("both")


def plot_init(label_x, label_y, title):
    """Initialize plot with given labels and title."""
    plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.19, bottom=0.19, top=0.79, right=0.99)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


def get_box_dimensions(infile):
    """Extract box dimensions from LAMMPS output file."""
    cmd = "grep 'orthogonal box ' " + infile + " | tail -1"
    data = subprocess.check_output(cmd, shell=True, text=True).split(" ")

    cmd = "grep -A 1 'Replication is creating ' " + infile + " | tail -1"
    replica_line = subprocess.check_output(cmd, shell=True, text=True)

    if replica_line != "":
        cmd = "grep -A 1 'Replication is creating ' " + infile + " | tail -1"
        data = subprocess.check_output(cmd, shell=True, text=True).split(" ")

    A = float(data[9].split("(")[1])
    B = float(data[10])
    C = float(data[11].split(")")[0])
    volume = A * B * C

    return A, B, C, volume


def get_time_series_data(infile):
    """Extract time series data from LAMMPS output file."""
    cmd = "grep -A 100000000 'v_efield         v_Pz' " + infile
    data = subprocess.check_output(cmd, shell=True, text=True).splitlines()

    data_vals = []
    for line in data:
        if "Step" in line:
            continue
        if "Loop time" in line:
            break
        data_vals.append(line)

    nframes = len(data_vals)
    time = np.zeros(nframes)
    E = np.zeros(nframes)
    P = np.zeros(nframes)

    for i in range(nframes):
        E[i] = float(data_vals[i].split()[6])
        P[i] = float(data_vals[i].split()[7])
        time[i] = float(data_vals[i].split()[1])

    return time, E, P, nframes


def get_unit_conversions(units):
    """Get unit conversion factors based on specified units."""
    if units == "eV":
        Pconv = 1.0  # e * A
        Econv = 1.0  # V / A
        Ωconv = 1.0  # A^3
        Eunit = "V$\cdot$A$^{-1}$"
        Punit = "e$\cdot$A$^{-2}$"
    elif units == "atomic":
        Pconv = np.sqrt(2.0) / bohr2A  # Ry a.u.
        Econv = (bohr2A * np.sqrt(2)) / hartree2eV  # Ry a.u.
        Ωconv = A2bohr**3  # Bohr^3
        Eunit = "a.u."
        Punit = "Ry a.u. / Bohr$^3$"
    else:  # SI units
        Pconv = (e2C * C2μC) * (A2m * m2cm)  # μC * cm
        Econv = V2MV / (A2m * m2cm)  # MV / cm
        Ωconv = (A2m * m2cm) ** 3  # in cm^3
        Eunit = "MV$\cdot$cm$^{-1}$"
        Punit = "μC$\cdot$cm$^{-2}$"

    return Pconv, Econv, Ωconv, Eunit, Punit


def process_hysteresis_data(infile, units="SI"):
    """
    Process LAMMPS output file for hysteresis analysis.

    Args:
        infile (str): Path to LAMMPS output file
        units (str): Units system to use ("SI", "eV", or "atomic")

    Returns:
        dict: Dictionary containing processed data
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"The file '{infile}' does not exist.")

    system = infile.split(".dat")[0]
    print(f"• {system}")

    # Get box dimensions and time series data
    A, B, C, volume = get_box_dimensions(infile)
    time, E, P, nframes = get_time_series_data(infile)

    # Get unit conversions
    Pconv, Econv, Ωconv, Eunit, Punit = get_unit_conversions(units)

    # Apply unit conversions
    P *= Pconv / Ωconv
    E *= Econv

    return {
        "system": system,
        "A": A,
        "B": B,
        "C": C,
        "volume": volume,
        "time": time,
        "E": E,
        "P": P,
        "nframes": nframes,
        "Eunit": Eunit,
        "Punit": Punit,
    }


def plot_hysteresis_average(data_list, pdf, plot_time_ev):
    """Plot hysteresis curves and time evolution."""
    # Electric field and average polarization
    time = data_list[0]["time"]
    E = data_list[0]["E"]
    P = np.zeros(len(E))
    for data in data_list:
        P += data["P"]
    P /= len(data_list)

    if len(data_list) > 1:
        print(f"Averaging polarization over {len(data_list)} trajectories")

    # Plot average hysteresis
    plot_init(
        f"Efield ({data_list[0]['Eunit']})",
        f"P/$\Omega$ ({data_list[0]['Punit']})",
        "Hysteresis",
    )
    plt.plot(E, P, lw=2.5, color=c_ml[0], zorder=100)
    pdf.savefig()

    if plot_time_ev:
        # Plot Efield
        plot_init("Time (ps)", f"Efield ({data_list[0]['Eunit']})", "Electric field")
        plt.plot(time, E, lw=2.5, color=c_ml[0], zorder=100)
        pdf.savefig()

        # Plot polarization
        plot_init("Time (ps)", f"P/$\Omega$ ({data_list[0]['Punit']})", "Polarization")
        plt.plot(time, P, lw=2.5, color=c_ml[0], zorder=100)
        pdf.savefig()


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Specify Material"
    system = sys.argv[1]

    if system not in MATERIAL_CONFIG:
        print(f"{system} not implemented")
        sys.exit(1)

    config = MATERIAL_CONFIG[system]
    infiles = config["infiles"]
    plot_time_ev = config["plot_time_ev"]

    # Process data
    data_list = []
    for infile in infiles:
        data = process_hysteresis_data(system + "/" + infile)
        data_list.append(data)

    # Plot data
    with PdfPages(f"{system}/{system}.pdf") as pdf:
        plot_hysteresis_average(data_list, pdf, plot_time_ev)

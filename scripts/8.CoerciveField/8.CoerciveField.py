# ===============
# COERCIVE EFIELD
# ===============

"""
This script determines the dependency of coercive field with respect to temperature,
given a statistics of hysteresis curves obtained with LAMMPS, carried out with the
pair-allegro interface that includes the treatment of polarization, Born charges,
and polarizability.

The script plots:
• average coercive field from up to dw polarization vs temperature
• average coercive field from dw to up polarization vs temperature
• average coercive field (up-dw, dw-up) vs temperature

Authors: S. Falletta, A. Johansson
"""

# Library
import sys
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)

colors = ["#0088aa", "#5d6c53", "#d40000"]

# Constants
bohr2A = 0.529177249
A2bohr = 1.8897259886
hartree2eV = 27.211396641308
e2C = 1.602176634e-19
A2m = 1e-10
m2cm = 1e2
C2μC = 1e6
V2MV = 1e-6

# Units
Econv = V2MV / (A2m * m2cm)  # eV -> MV / cm

# Material Configuration
MATERIAL_CONFIG = {
    "BaTiO3": {
        "prefix": "log.hyst_BaTiO3-E0_BaTiO3-sc333",  # Nomenclature for files
        "Ti": 280,  # Initial temperature
        "Tf": 350,  # Final temperature
        "niter": 20,  # Size of the statistics at fixed T
    }
}


def plot_init(label_x, label_y, title):
    plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.19, bottom=0.19, top=0.79, right=0.99)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


def axis_settings(ax):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.xaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
    ax.yaxis.set_ticks_position("both")


def find_data(filename="log.lammps", verbose=False):
    """
    Extract data from a LAMMPs log file.

    Parameters
    ----------
    filename: str, optional
        Name of LAMMPs log fil.
        Default: log.lammps.
    verbose: bool, optional
        If True, the header of each chunk will
        be printed. Default: False.

    Returns
    -------
    data: dict
        Dictionary where the keys are the headers
        in the log file, and the values are lists
        of data.
    """
    with open(filename, "r") as infile:
        lines = infile.readlines()
    N = len(lines)
    data = {}

    if verbose:
        print("Reading " + filename)

    for i in range(N):
        line = lines[i]
        if "Per MPI rank memory" in line:
            i = i + 1
            line = lines[i]
            if verbose:
                print(line)
            headers = line.split()
            for word in headers:
                if word not in data.keys():
                    data[word] = []
            i += 1
            while i < N and "Loop time" not in lines[i]:
                word = lines[i].split()
                if len(word) != len(headers):
                    return data
                for j in range(len(word)):
                    data[headers[j]].append(float(word[j]))
                i += 1
    return data


def analyze(system, prefix, T, i):
    data = find_data(f"{system}/{prefix}_{T}_2_{i}")
    Pz = np.array(data["v_Pz"])
    E = data["v_efield"]
    Nhalf = len(Pz) // 2
    switchdown = np.argmax(Pz < 0)
    switchup = Nhalf + np.argmax(Pz[Nhalf:] > 0)
    return [E[switchup], E[switchdown]]


if __name__ == "__main__":

    assert len(sys.argv) > 1, "Specify Material"
    system = sys.argv[1]

    if system not in MATERIAL_CONFIG:
        print(f"{system} not implemented")
        exit()

    config = MATERIAL_CONFIG[system]
    prefix = config["prefix"]
    Ti = config["Ti"]
    Tf = config["Tf"]
    niter = config["niter"]

    Ts = np.arange(Ti, Tf + 1, 10)
    nTs = len(Ts)
    futures = []

    with Pool() as pool:
        for T in Ts:
            for i in range(niter):
                futures.append(pool.apply_async(analyze, (system, prefix, T, i)))

        results = [fut.get() for fut in futures]

    upfields = np.reshape([result[0] for result in results], (nTs, niter)) * Econv
    downfields = np.reshape([result[1] for result in results], (nTs, niter)) * Econv

    mean_upfield = np.abs(np.mean(upfields, axis=1))
    mean_downfield = np.abs(np.mean(downfields, axis=1))

    std_upfield = np.std(upfields, axis=1) / np.sqrt(niter)
    std_downfield = np.std(downfields, axis=1) / np.sqrt(niter)

    with PdfPages(system + "/" + system + ".pdf") as pdf:
        plot_init("T (K)", "$E_c$ (MV$\cdot$cm$^{-1}$)", "")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        # Plot up to down transition
        plt.plot(Ts, mean_upfield, label="up→dw", c=colors[0], lw=2.5)
        plt.fill_between(
            Ts,
            mean_upfield + std_upfield,
            mean_upfield - std_upfield,
            alpha=0.5,
            color=colors[0],
        )

        # Plot down to up transition
        plt.plot(Ts, mean_downfield, label="dw→up", c=colors[1], lw=2.5)
        plt.fill_between(
            Ts,
            mean_downfield + std_downfield,
            mean_downfield - std_downfield,
            alpha=0.5,
            color=colors[1],
        )

        # Plot mean
        mean_field = (mean_downfield + mean_upfield) / 2
        plt.plot(Ts, mean_field, label="mean", c=colors[2], lw=2.5)
        plt.scatter(Ts, mean_field, c=colors[2], s=100, zorder=100)

        plt.legend(frameon=False)
        pdf.savefig()

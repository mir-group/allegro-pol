# =============
# WALL DYNAMICS
# =============

"""
This script parses a extxyz file with the dipoles per cell, which is 
generated by hysteresis.py, and determines the mobility of domain walls 
through a majority rule.

How to run:
    python3 9.Mobility.py BaTiO3

The input data needs to be specified in the MATERIAL_CONFIG dictionary 
at the bottom of this file.

The script works by:
1. Parsing the extxyz file containing dipoles per unit cell
2. Folding atomic positions into a consistent supercell across all frames
3. Building a 2D map of the Cartesian space with bins corresponding to unit cells
4. Calculating the average dipole orientation in each bin
5. Determining domain wall positions and their mobility

Note: Currently only valid for orthorhombic cells.

Units in the input file:
    - Coordinates in A
    - Dipoles in e*A

Author: S. Falletta
"""

# Libraries
import sys
import numpy as np
import matplotlib.colors as mcolors
import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    'font.size': 24,
    'legend.fontsize': 20,
    'legend.handlelength': 0.5
})

MATERIAL_CONFIG = {
    "BaTiO3": {
        "file_extxyz": "BaTiO3/BaTiO3-dipoles.xyz",
        "p_unit": [
            [3.966084, 0, 0],
            [0, 3.966084, 0], 
            [0, 0, 4.304025]
        ]
    }
}

def axis_settings(ax):
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(
        which='major', width=3.0, length=12, direction="in"
    )
    ax.xaxis.set_tick_params(
        which='minor', width=3.0, length=6, direction="in"
    )
    ax.yaxis.set_tick_params(
        which='major', width=3.0, length=12, direction="in"
    )
    ax.yaxis.set_tick_params(
        which='minor', width=3.0, length=6, direction="in"
    )
    ax.yaxis.set_ticks_position('both')


def plot_init(label_x, label_y, title):
    fig = plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.19, bottom=0.19, top=0.79, right=0.99)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return fig, ax


class extxyz_dipoles:

    def split(self, string, chr):
        return ' '.join(string.split(chr))

    def __init__(self, system, filename, p_unit):
        print("Parsing file {:s}".format(filename))

        with open(filename, "r") as f:
            data = f.readlines()
        self.system = system
        self.nat = int(data[0].split()[0])
        self.nframes = int(len(data) / (self.nat + 2))
        self.p_unit = np.array(p_unit)
        self.filename = filename

        # atomic positions and dipoles
        self.p = np.zeros((self.nframes, 3, 3))
        self.R = np.zeros((self.nframes, self.nat, 3))
        self.θ = np.zeros((self.nframes, self.nat))
        self.dipoles = np.zeros((self.nframes, self.nat, 3))
        self.kinds = []

        for frame in tqdm.trange(self.nframes):
            # split 2nd line
            L = data[frame * (self.nat + 2) + 1]
            L = self.split(L, '=')
            L = self.split(L, ' ')
            L = self.split(L, '"')
            L = L.split()

            # get indexes
            idx_latt = L.index('Lattice') + 1
            self.p[frame, :, :] = np.array(
                [float(x) for x in L[idx_latt:idx_latt + 9]]
            ).reshape(3, 3)

            # per atom quantities
            kinds = []
            for iat in range(self.nat):
                L = data[frame * (self.nat + 2) + 2 + iat].split()
                assert len(L) == 8, f"missing info in line: {L}"
                kinds.append(L[0])
                self.R[frame, iat, :] = np.array([float(x) for x in L[1:4]])
                self.dipoles[frame, iat, :] = np.array([float(x) for x in L[4:7]])
                self.θ[frame, iat] = float(L[7])
            self.kinds.append(kinds)

    def fold_images(self, δ=0.5):
        """
        fold center of dipoles within a consistent supercell across all frames.
        tuning of δ might be required
        """
        # Iteratively shift coordinates
        for frame in range(self.nframes):
            for ia in range(self.nat):
                r = np.array(self.R[frame, ia, :])
                for _ in range(3):
                    if r[0] > self.p[frame, 0, 0] - δ:
                        r -= self.p[frame, 0, :]
                    elif r[0] < 0 - δ:
                        r += self.p[frame, 0, :]
                    if r[1] > self.p[frame, 1, 1] - δ:
                        r -= self.p[frame, 1, :]
                    elif r[1] < 0 - δ:
                        r += self.p[frame, 1, :]
                    if r[2] > self.p[frame, 2, 2] - δ:
                        r -= self.p[frame, 2, :]
                    elif r[2] < 0 - δ:
                        r += self.p[frame, 2, :]

                # Check if coordinates are shifted beyond expected bounds
                if (r[0] < -δ or r[1] < -δ or r[2] < -δ or
                    r[0] > self.p[frame, 0, 0] + δ or
                    r[1] > self.p[frame, 1, 1] + δ or
                    r[2] > self.p[frame, 2, 2] + δ):
                    print("Add more iterations of shift")

                # Update coordinates
                self.R[frame, ia, :] = r

                # in addition, shift them by 0.5 unit cell along each direction
                self.R[frame, ia, :] += 0.5 * (
                    self.p_unit[0, :] + self.p_unit[1, :] + self.p_unit[2, :]
                )


    def map_structure(self, save_pdf=True):
        """
        Construct a 2D map of the Cartesian space x,y composed of bins,
        each of which corresponds to the unit cell of BaTiO3. Place
        the center of the bin at the center of the unit cell.
        Additionally, sum the dipoles for each bin, calculate their 
        average, and color the bins based on the angle between 
        the dipole and the z-axis. Ensure that there are n_z dipoles
        per bin, and raise an error if not.
        Option to skip plotting to speed up calculations.
        """
        print("Building the map structure")

        self.map_data = {}

        # Loop over frames
        for frame in tqdm.trange(self.nframes):
            if save_pdf:
                pdf = PdfPages(f"{system}/{system}-{frame}.pdf")

            # Assume all lattice parameters are fixed
            p = self.p[frame, :, :]
            p_unit = self.p_unit
            dipoles = self.dipoles[frame, :, :]
            R = self.R[frame, :, :]

            # Calculate replication index for each direction
            r_x = int(p[0, 0] / p_unit[0, 0])
            r_y = int(p[1, 1] / p_unit[1, 1])
            r_z = int(p[2, 2] / p_unit[2, 2])

            n_x, n_y, n_z = int(r_x), int(r_y), int(r_z)

            # Initialize color map
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            cmap = plt.cm.coolwarm

            if save_pdf:
                fig, ax = plot_init("x", "y", f"dipoles at step {frame}")

            # Data structure for storing the bin information
            frame_data = {}

            # Loop over each bin (i, j) in the xy-plane
            for i in range(n_x):
                for j in range(n_y):
                    # Find the dipoles that fall inside this bin
                    x_min, x_max = i * p_unit[0, 0], (i + 1) * p_unit[0, 0]
                    y_min, y_max = j * p_unit[1, 1], (j + 1) * p_unit[1, 1]

                    # Get all dipoles in the current bin
                    dipoles_in_bin = []
                    for k in range(len(R)):
                        x, y = R[k, 0], R[k, 1]
                        if x_min <= x < x_max and y_min <= y < y_max:
                            dipoles_in_bin.append(dipoles[k, :])

                    # Raise an error if the number of dipoles does not match n_z
                    if len(dipoles_in_bin) != n_z:
                        raise ValueError(
                            f"Expected {n_z} dipoles in bin ({i}, {j}), "
                            f"found {len(dipoles_in_bin)}"
                        )

                    # Calculate the average dipole in the bin
                    average_dipole = np.mean(dipoles_in_bin, axis=0)

                    # Calculate the cosine of the angle between the average
                    # dipole and the z-axis
                    z_axis = np.array([0, 0, 1])
                    cosine_angle = np.dot(average_dipole, z_axis) / (
                        np.linalg.norm(average_dipole) + 1e-10
                    )

                    # Store the data for later post-processing
                    frame_data[(i, j)] = {
                        'average_dipole': average_dipole,
                        'cosine_angle': cosine_angle
                    }

                    if save_pdf:
                        # Color based on the cosine of the angle
                        color = cmap(norm(cosine_angle))

                        # Draw the unit cell as a rectangle with the chosen color
                        rect = plt.Rectangle(
                            (i, j), 1, 1, facecolor=color, edgecolor='black'
                        )
                        ax.add_patch(rect)

                        # Add a dot at the center of each unit cell
                        center_x = i + 0.5
                        center_y = j + 0.5
                        ax.plot(
                            center_x, center_y, 'o', color='black'
                        )  # Black dot at the center

            # Save the map data for the current frame
            self.map_data[frame] = frame_data

            if save_pdf:
                # Add horizontal and vertical grid lines
                for i in range(n_x + 1):
                    ax.axvline(
                        x=i, color='black', linestyle='--', linewidth=0.5
                    )
                for j in range(n_y + 1):
                    ax.axhline(
                        y=j, color='black', linestyle='--', linewidth=0.5
                    )

                # Set limits and labels
                ax.set_xlim(0, n_x)
                ax.set_ylim(0, n_y)
                ax.set_aspect('equal')

                # Add the color bar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Cosine of angle with z-axis')

                # Save the plot to the PDF
                pdf.savefig()

                # Close the PDF
                pdf.close()


if __name__ == "__main__":

    assert len(sys.argv) > 1, "Specify Material"
    system = sys.argv[1]

    if system not in MATERIAL_CONFIG:
        print(f"{system} not implemented")
        exit()

    config = MATERIAL_CONFIG[system]
    file_extxyz = config["file_extxyz"]
    p_unit = config["p_unit"]

    S = extxyz_dipoles(system, file_extxyz, p_unit)
    S.fold_images()
    S.map_structure()

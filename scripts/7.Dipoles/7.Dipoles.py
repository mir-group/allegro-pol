#!/usr/bin/env python3

# =====================
# FERROELECTRIC DIPOLES
# =====================

"""
This script parses an extxyz structural file containing Born charges (orthorombic
perovskite structure only), and determines the ferroelectric dipoles. The script
plots:
• 2D or 3D plots of the dipoles
• angle distribution of dipoles
• extxyz file with dipoles per unit cell.

The extxyz outfile containing the dipoles can be used to visualize the dipoles
with commonly-used softwares, such as Ovito.

How to run: python3 dipoles.py BaTiO3

Before using this script, make sure your folders are organized as follows:
Nomenclature for input folder: {system}/{infile}
where:
• system: name of the material (e.g. "BaTiO3")
• infile: name of the extxyz file containing Born charges

The input data needs to be specified in the MATERIAL_CONFIG dictionary at the
bottom of this file.

TODO: if your structure is not that of a perovskite, you need to edit this script.
See the neighbor_list subroutine.

Author: S. Falletta
"""

# Standard library imports
import os
import sys
import time
import math
from multiprocessing import Pool

# Third party imports
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Plot settings
plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)

# Material configurations
MATERIAL_CONFIG = {
    "BaTiO3": {
        "infile": "BaTiO3.xyz",
        "kinds": ["Ba", "Ti", "O"],
        "weights": [0.125, 1.00, 0.50],
        "p_unit": [3.966084, 3.966084, 4.304025],
    }
    # Add more materials here following the same format
}

# Plot settings
PLOT_CONFIG = {"do_plot": True, "pdf": False}


class Arrow3D(FancyArrowPatch):
    """Class for creating 3D arrows in matplotlib plots."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def arrow3dtube(ax, rc, d, s=0.8, w=0.1, h=0.5, hw=2, **kw):
    """
    Create a 3D tube arrow in matplotlib plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the arrow on
    rc : array_like
        Center position of the arrow
    d : array_like
        Dipole vector
    s : float, optional
        Scale factor, default 0.8
    w : float, optional
        Width of the arrow, default 0.1
    h : float, optional
        Head length, default 0.5
    hw : float, optional
        Head width, default 2
    **kw : dict
        Additional keyword arguments passed to plot_surface
    """
    phi = np.pi - math.atan2(d[1], d[0])
    theta = math.acos(d[2] / math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
    a = np.array([[0, 0], [w, 0], [w, (1 - h)], [hw * w, (1 - h)], [0, 1]])
    r, theta2 = np.meshgrid(a[:, 0], np.linspace(0, 2 * np.pi, 30))
    z = np.tile(a[:, 1], r.shape[0]).reshape(r.shape) * s * np.linalg.norm(d)
    x = r * np.sin(theta2) * s * np.linalg.norm(d)
    y = r * np.cos(theta2) * s * np.linalg.norm(d)
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    rot_z = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )
    b1 = np.dot(rot_x, np.c_[x.flatten(), y.flatten(), z.flatten()].T)
    b2 = np.dot(rot_z, b1)
    b2 = b2.T + np.array(rc)
    x = b2[:, 0].reshape(r.shape)
    y = b2[:, 1].reshape(r.shape)
    z = b2[:, 2].reshape(r.shape)
    ax.plot_surface(x, y, z, **kw)


def axis_settings(ax):
    """
    Apply standard axis settings to a matplotlib plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to apply settings to
    """
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_ticks_position("both")


class Trajectory:
    """Class for handling trajectory data and calculating dipoles."""

    def __init__(self, infile, kinds, weights, p_unit):
        """
        Initialize Trajectory class.

        Parameters
        ----------
        infile : str
            Path to input extxyz file
        kinds : list
            List of atom kinds in the system
        weights : list
            Weights for each atom kind
        p_unit : list
            Unit cell parameters
        """
        # parse input data
        self.system = infile[:-4]
        if not os.path.exists(infile):
            raise FileNotFoundError(f"The file '{infile}' does not exist.")
        with open(infile) as f:
            data = f.readlines()
        self.infile = infile
        self.N = int(data[0].split()[0])
        if "Lattice" not in data[1]:
            print("'{:s}' must be formatted in extxyz!".format(infile))
            exit()
        if "c_borncharges" not in data[1]:
            print("'{:s}' must include Born charges!".format(infile))
            exit()

        # from lattice vectors to lattice parameters
        self.weights = weights
        self.kinds = kinds
        self.p = np.array([float(x) for x in data[1].split('"')[1].split()]).reshape(
            (3, 3)
        )
        self.A = np.linalg.norm(self.p[0, :])
        self.B = np.linalg.norm(self.p[1, :])
        self.C = np.linalg.norm(self.p[2, :])
        self.alpha = (
            np.arccos(np.dot(self.p[1, :], self.p[2, :]) / (self.B * self.C))
            * 180
            / np.pi
        )
        self.beta = (
            np.arccos(np.dot(self.p[0, :], self.p[2, :]) / (self.A * self.C))
            * 180
            / np.pi
        )
        self.gamma = (
            np.arccos(np.dot(self.p[0, :], self.p[1, :]) / (self.A * self.B))
            * 180
            / np.pi
        )
        self.latt = [self.A, self.B, self.C, self.alpha, self.beta, self.gamma]
        self.latt_line = data[1]
        self.N_steps = int(len(data) / (self.N + 2))
        self.frames = {
            i: Structure(data[i * (self.N + 2) : (i + 1) * (self.N + 2)], self.latt)
            for i in range(self.N_steps)
        }

        # unit cell parameters
        self.p_unit = p_unit
        self.Nx = int(self.A / self.p_unit[0])
        self.Ny = int(self.B / self.p_unit[1])
        self.Nz = int(self.C / self.p_unit[2])

        # calculate neighbor list for first frame and keep that fixed
        self.neighbor_list(self.frames[0].atoms)

        # calculate dipoles
        self.dipoles()

        # print dipoles in extxyz
        self.print_extxyz()

    def dist(self, x1, x2, ix):
        """
        Calculate minimal image convention distance.

        Parameters
        ----------
        x1 : float
            Center position (Ti atom)
        x2 : float
            Position to calculate distance from
        ix : int
            Lattice direction index

        Returns
        -------
        float
            Minimal image distance
        """
        idx = np.argmin(
            [abs(x2 - x1), abs(x2 + self.latt[ix] - x1), abs(x2 - self.latt[ix] - x1)]
        )
        if idx == 0:
            return x2 - x1
        if idx == 1:
            return x2 + self.latt[ix] - x1
        if idx == 2:
            return x2 - self.latt[ix] - x1

    def neighbor_list(self, atoms, cutoff=[3.0, 3.0, 3.0]):
        """
        Construct neighbor list from first frame.

        Parameters
        ----------
        atoms : list
            List of Atom objects
        cutoff : list, optional
            Cutoff distances for each direction, default [3.0, 3.0, 3.0]
            Example:
            * [0.5,0.5,3.0] selects only the O above and below the Ti atom
            * [3.0,3.0,3.0] selects all atoms in unit cell
        """
        # get indexes of Ti atoms
        idxs = [i for i, A in enumerate(atoms) if A.kind == self.kinds[1]]

        # construct dictionary of all neighbors of Ti atoms
        self.neighbors = {
            i: [
                j
                for j in range(len(atoms))
                if j not in idxs
                and all(
                    abs(self.dist(x1, x2, ix)) < ci
                    for x2, x1, ci, ix in zip(atoms[j].r, atoms[i].r, cutoff, [0, 1, 2])
                )
            ]
            for i in idxs
        }

        # check that you have 6 surrounding O atoms and 8 surrounding Ba atoms
        if any(
            (
                (sum(atoms[j].kind == self.kinds[0] for j in self.neighbors[i]) != 8)
                or (sum(atoms[j].kind == self.kinds[2] for j in self.neighbors[i]) != 6)
                or (
                    (
                        sum(
                            atoms[j].kind == self.kinds[0]
                            or atoms[j].kind == self.kinds[2]
                            for j in self.neighbors[i]
                        )
                    )
                    != 14
                )
            )
            for i in idxs
        ):
            print("Error: adjust cutoffs to include all atoms in supercell")
            exit()

    def dipoles(self):
        """Calculate dipoles for all frames."""
        self.dipoles = np.zeros((len(self.frames), len(self.neighbors), 3))
        for n in range(len(self.frames)):
            for k, i in enumerate(self.neighbors.keys()):
                for j in self.neighbors[i]:
                    w = (
                        self.weights[kinds.index(self.frames[n].atoms[j].kind)]
                        if self.frames[n].atoms[j].kind in self.kinds
                        else 1
                    )
                    qB = np.array(
                        [self.frames[n].atoms[j].Z[ix, ix] for ix in range(3)]
                    )
                    dR = np.array(
                        [
                            self.dist(
                                self.frames[n].atoms[i].r[ix],
                                self.frames[n].atoms[j].r[ix],
                                ix,
                            )
                            for ix in range(3)
                        ]
                    )
                    self.dipoles[n, k, :] += w * qB * dR

    def print_extxyz(self):
        """
        Print extxyz file with dipole information on Ti atoms.

        The dipoles are labelled as forces to ease visualization in Ovito.
        """
        nat = len(self.neighbors)
        nframes = len(self.frames)
        with open(self.system + "-dipoles.xyz", "w") as f:
            for n in range(nframes):
                R = np.array([self.frames[n].atoms[k].r for k in self.neighbors.keys()])
                P = self.dipoles[n, :, :]
                cos_theta = P[:, 2] / np.linalg.norm(P[:, :], axis=1)
                f.write("{:d}\n".format(nat))
                f.write(
                    'Lattice="{:s}" Properties=species:S:1:pos:R:3:forces:R:3:theta:R:1 pbc="T T T" \n'.format(
                        " ".join(str(x) for x in self.p.ravel()),
                    )
                )
                for k in range(nat):
                    str_format = "{:5s} " + "{:14.10f} " * 7 + " \n"
                    f.write(
                        str_format.format(
                            "Ti",
                            R[k, 0],
                            R[k, 1],
                            R[k, 2],
                            P[k, 0],
                            P[k, 1],
                            P[k, 2],
                            cos_theta[k],
                        )
                    )
        print("Printed results in file:", self.system + "-dipoles.xyz")

    def plot_dipoles_2D(
        self, n, namefile, Ngrid=25, pdf=False, plane="xz", dpi_png=200
    ):
        """
        Plot dipoles in 2D plane.

        Parameters
        ----------
        n : int
            Frame number
        namefile : str
            Output filename
        Ngrid : int, optional
            Grid size for visualization, default 25
        pdf : bool, optional
            Whether to save as PDF, default False
        plane : str, optional
            Plane to plot in ("xz" or "yz"), default "xz"
        dpi_png : int, optional
            DPI for PNG output, default 200
        """
        # dipoles direction and corresponding positions of Ti atoms
        Pz = self.dipoles[n, :, 2]
        R = np.array([self.frames[n].atoms[k].r for k in self.neighbors.keys()])

        # max values
        Pz_max = np.max(np.abs(self.dipoles[:, :, 2]))
        Pz_min = -Pz_max

        # remap Pz into a grid
        Pz_grid = np.zeros((Ngrid, Ngrid))
        for na in range(len(Pz)):
            ix = int(R[na, 0] / self.A * Ngrid)
            iy = int(R[na, 1] / self.B * Ngrid)
            iz = int(R[na, 2] / self.C * Ngrid)
            if ix < 0:
                ix += Ngrid
            elif ix > Ngrid - 1:
                ix -= Ngrid
            if iy < 0:
                iy += Ngrid
            elif iy > Ngrid - 1:
                iy -= Ngrid
            if iz < 0:
                iz += Ngrid
            elif iz > Ngrid:
                iz -= Ngrid
            if plane == "xz":
                Pz_grid[iz, ix] = Pz[na]
            if plane == "yz":
                Pz_grid[iz, iy] = Pz[na]

        f = plt.figure(figsize=(6, 6), dpi=60)
        plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
        ax = plt.gca()
        plt.title("Step {:d}".format(n), size=20)
        plt.ylabel("$z$ (Å)")
        axis_settings(ax)
        if plane == "xz":
            plt.xlabel("$x$ (Å)")
            plt.imshow(
                Pz_grid,
                extent=[0, self.A, 0, self.C],
                vmax=Pz_max,
                vmin=Pz_min,
                cmap="jet",
                interpolation="lanczos",
            )
        if plane == "yz":
            plt.xlabel("$y$ (Å)")
            plt.imshow(
                Pz_grid,
                extent=[0, self.B, 0, self.C],
                vmax=Pz_max,
                vmin=Pz_min,
                cmap="jet",
                interpolation="lanczos",
            )

        cbar = plt.colorbar()
        cbar.ax.tick_params(axis="y", which="both", width=2.5, length=10)
        cbar.outline.set_linewidth(1.5)

        if not pdf:
            f.savefig(namefile + ".png", dpi=dpi_png)
        else:
            f.savefig(namefile + ".pdf", dpi=dpi_png)
        plt.close()

    def plot_dipoles_3D(
        self, n, namefile, s=1, visualization_mode=2, width=2, pdf=False
    ):
        """
        Plot dipoles in 3D.

        Parameters
        ----------
        n : int
            Frame number
        namefile : str
            Output filename
        s : float, optional
            Scale factor for arrows, default 1
        visualization_mode : int, optional
            Visualization mode (0: quiver, 1: 3D arrows, 2: tube arrows), default 2
        width : float, optional
            Line width, default 2
        pdf : bool, optional
            Whether to save as PDF, default False
        """
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")
        ax.set_xlabel("x (Å)", fontsize="16")
        ax.set_ylabel("y (Å)", fontsize="16")
        ax.set_zlabel("z (Å)", fontsize="16")
        ax.set_xlim((0, self.latt[0]))
        ax.set_ylim((0, self.latt[1]))
        ax.set_zlim((0, self.latt[2]))
        ax.xaxis.line.set_lw(width)
        ax.yaxis.line.set_lw(width)
        ax.zaxis.line.set_lw(width)

        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.set_facecolor("none")
        ax.tick_params(axis="x", which="both", labelsize=14, length=12, width=width)
        ax.tick_params(axis="y", which="both", labelsize=14, length=12, width=width)
        ax.tick_params(axis="z", which="both", labelsize=14, length=12, width=width)
        plt.title("Step {:d}".format(n), size=18)
        plt.subplots_adjust(right=0.85)
        cos_theta = self.dipoles[n, :, 2] / np.linalg.norm(
            self.dipoles[n, :, :], axis=1
        )

        for i, k in enumerate(self.neighbors.keys()):
            # k indexes the Ti atoms
            r, p = self.frames[n].atoms[k].r, self.dipoles[n, :, :]
            color = plt.cm.jet((cos_theta[i] + 1) / 2)

            if visualization_mode == 0:
                ax.quiver(
                    r[0],
                    r[1],
                    r[2],
                    s * p[i, 0],
                    s * p[i, 1],
                    s * p[i, 2],
                    color=color,
                    linewidth=2,
                    arrow_length_ratio=0.5,
                )
            elif visualization_mode == 1:
                arrow_prop_dict = dict(
                    mutation_scale=20,
                    arrowstyle="simple",
                    color=color,
                    shrinkA=3,
                    shrinkB=3,
                )
                a = Arrow3D(
                    [r[0], r[0] + s * p[i, 0]],
                    [r[1], r[1] + s * p[i, 1]],
                    [r[2], r[2] + s * p[i, 2]],
                    linewidth=0.5,
                    **arrow_prop_dict,
                )
                ax.add_artist(a)
            elif visualization_mode == 2:
                arrow3dtube(ax, r, s * p[i, :], color=color)

        # Add color bar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=-1, vmax=1)),
            ax=ax,
            ticks=[-1, 0, 1],
            pad=0.15,
            shrink=0.6,
        )
        cbar.set_label("cos($\\theta$)", fontsize=16)
        cbar.ax.tick_params(axis="both", which="both", length=6, width=width)
        cbar.outline.set_linewidth(1.5)

        ax.view_init(elev=30, azim=45)

        if not pdf:
            f.savefig(namefile + ".png", dpi=300)
        else:
            f.savefig(namefile + ".pdf", dpi=300)
        plt.close()

    def plot_angles(self, n, namefile, pdf=False):
        """
        Plot angle distribution histograms.

        Parameters
        ----------
        n : int
            Frame number
        namefile : str
            Output filename
        pdf : bool, optional
            Whether to save as PDF, default False
        """
        vectors = self.dipoles[n, :, :]

        # Compute polar angles (phi) in radians
        r = np.linalg.norm(vectors, axis=1)
        phi = np.arccos(vectors[:, 2] / r)

        # Define number of bins
        num_bins = 36

        # Compute histogram for phi
        phi_bins = np.linspace(0, np.pi, num_bins + 1)
        phi_hist, phi_bin_edges = np.histogram(phi, bins=phi_bins)

        # Create semi-polar plot for phi
        fig_phi, ax_phi = plt.subplots(subplot_kw={"projection": "polar"})
        ax_phi.set_title("Polar Angle (Phi) Histogram", fontsize=16)
        ax_phi.set_theta_zero_location("N")
        ax_phi.set_theta_direction(-1)

        # Plot the histogram
        theta = np.linspace(0, np.pi, num_bins)
        bars = ax_phi.bar(theta, phi_hist, width=2 * np.pi / num_bins, bottom=0.0)

        # Color the bars based on angle
        for bar, angle in zip(bars, theta):
            bar.set_facecolor(plt.cm.jet(angle / np.pi))
            bar.set_alpha(0.7)

        # set limits
        Nmax = max(phi_hist) * 1.1
        ax_phi.set_ylim(0, Nmax)

        # Restrict the plot to half circle (0 to 180 degrees)
        ax_phi.set_thetamin(0)
        ax_phi.set_thetamax(180)

        # Add color bar for phi
        sm_phi = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(0, np.pi))
        sm_phi.set_array([])
        cbar = fig_phi.colorbar(sm_phi, ax=ax_phi, pad=0.1, label="Angle (radians)")
        cbar.ax.tick_params(labelsize=16)  # Adjust color bar label size
        cbar.set_label("Angle (radians)", fontsize=16)

        # Adjust tick label sizes
        ax_phi.tick_params(axis="both", which="major", labelsize=16)

        if not pdf:
            fig_phi.savefig(namefile + ".png", dpi=300)
        else:
            fig_phi.savefig(namefile + ".pdf")
        plt.close()


class Structure:
    """Class for handling structural data."""

    def split(self, string, chr):
        """Split string by character and join with spaces."""
        return " ".join(string.split(chr))

    def __init__(self, data, latt):
        """
        Initialize Structure class.

        Parameters
        ----------
        data : list
            List of lines from extxyz file
        latt : list
            Lattice parameters
        """
        self.N = int(data[0].split()[0])
        self.latt = latt
        self.atoms = []
        Z = np.zeros((3, 3))
        F = np.zeros(3)
        for line in data[2:]:
            kind = line.split()[0]
            r = np.array([float(x) for x in line.split()[1:4]])
            if len(line.split()) > 15:
                Z = np.array([float(x) for x in line.split()[4:13]]).reshape((3, 3))
                F = np.array([float(x) for x in line.split()[13:16]])
            self.atoms.append(Atom(kind, r, Z, F))


class Atom:
    """Class representing an atom with its properties."""

    def __init__(self, kind, r, Z, F):
        """
        Initialize Atom class.

        Parameters
        ----------
        kind : str
            Atom kind
        r : array_like
            Position vector
        Z : array_like
            Born charge tensor
        F : array_like
            Force vector
        """
        self.kind = kind
        self.r = r
        self.Z = Z
        self.F = F


if __name__ == "__main__":

    assert len(sys.argv) > 1, "Specify Material"
    system = sys.argv[1]

    if system not in MATERIAL_CONFIG:
        print("Material not implemented!")
        exit()

    config = MATERIAL_CONFIG[system]
    infile = config["infile"]
    kinds = config["kinds"]
    weights = config["weights"]
    p_unit = config["p_unit"]

    print("Parsing input file ...")
    start_time = time.time()
    T = Trajectory(f"{system}/{infile}", kinds, weights, p_unit)
    end_time = time.time()
    print("Time to parse file: {:.2f} s".format(end_time - start_time))

    if PLOT_CONFIG["do_plot"]:
        start_time = time.time()
        with Pool() as pool:
            for n in range(T.N_steps):
                pool.apply_async(
                    T.plot_dipoles_2D(
                        n,
                        system + "/" + system + "-" + str(n) + "-2D",
                        pdf=PLOT_CONFIG["pdf"],
                    )
                )
                pool.apply_async(
                    T.plot_dipoles_3D(
                        n,
                        system + "/" + system + "-" + str(n) + "-3D",
                        pdf=PLOT_CONFIG["pdf"],
                    )
                )
                pool.apply_async(
                    T.plot_angles(
                        n,
                        system + "/" + system + "-" + str(n) + "-theta",
                        pdf=PLOT_CONFIG["pdf"],
                    )
                )
        pool.close()
        pool.join()
        end_time = time.time()
        print("Time to process all steps: {:.2f} s".format(end_time - start_time))

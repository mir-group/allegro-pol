# =====================
# FERROELECTRIC DIPOLES
# =====================

"""
Author: S. Falletta

This script parses an extxyz structural file containing Born charges (orthorombic perovskite
structure only), and determines the ferroelectric dipoles. The script plots:
• 2D or 3D plots of the dipoles
• angle distribution of dipoles
• extxyz file with dipoles per unit cell.
The extxyz outfile containing the dipoles can be used to visualize the dipoles with
commonly-used softwares, such as Ovito.

How to run: python3 dipoles.py BaTiO3

TODO: if your structure is not that of a perovskite, you need to edit this script.
See the neighbor_list subroutine.
"""

# Libraries
import sys
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm, trange
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import KDTree
from multiprocessing import Pool
import time
import math

plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)


class Arrow3D(FancyArrowPatch):
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
    rc: center
    d:  dipole
    s:  scale
    w:  width
    h:  head
    hw: head width
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
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    ax.yaxis.set_ticks_position("both")


class Trajectory:

    def __init__(self, infile, kinds, weights):

        # parse input data
        self.system = infile[:-4]
        if not os.path.exists(infile):
            raise FileNotFoundError(f"The file '{infile}' does not exist.")

        # parse infile
        with open(infile, "r") as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            with tqdm(total=total_lines, desc="Reading file", unit="lines") as pbar:
                data = []
                for line in f:
                    data.append(line)
                    pbar.update(1)
        self.N = int(data[0].split()[0])
        if "Lattice" not in data[1]:
            print("'{:s}' must be formatted in extxyz!".format(infile))
            exit()
        if "c_borncharges" not in data[1]:
            print("'{:s}' must include Born charges!".format(infile))
            exit()
        self.weights = weights
        self.kinds = kinds
        self.p = np.array([float(x) for x in data[1].split('"')[1].split()]).reshape(
            (3, 3)
        )
        self.N_steps = int(len(data) / (self.N + 2))

        # construct frames
        # self.frames = {}
        # pool = Pool()
        # for i in range(len(data) // (self.N + 2)):  # Ensure the loop only goes through valid indices
        #    pool.apply_async(
        #        self.compute_structure,
        #        args=(data, i, self.N, self.p),
        #        callback=self.store_result
        #    )
        # pool.close()
        # pool.join()

        # serial subroutine
        self.frames = {}
        for i in tqdm(range(self.N_steps), desc="Processing frames"):
            self.frames[i] = Structure(
                data[i * (self.N + 2) : (i + 1) * (self.N + 2)], self.p
            )

        # calculate neighbor list for first frame and keep that fixed
        self.neighbor_list(self.frames[0].atoms)

        # calculate dipoles
        self.dipoles()

        # print dipoles in extxyz
        self.print_extxyz()

    """ PARALLEL IMPLEMENTATION """

    def compute_structure(data, i, N, p):
        data_slice = data[i * (N + 2) : (i + 1) * (N + 2)]
        return (i, Structure(data_slice, p))

    def store_result(self, result):
        i, structure = result
        self.frames[i] = structure

    """ ------------------------ """

    def neighbor_list(self, atoms, cutoff=[3.0, 3.0, 3.0]):
        """
        Construct neighbor list from first frame.
        Example cutoff:
        * [0.5,0.5,3.0] selects only the O above and below the Ti atom
        * [3.0,3.0,3.0] selects all atoms in unit cell
        """
        # Convert cutoff to a single radius for KDTree query
        cutoff_radius = np.linalg.norm(cutoff)

        # Generate an array of shift vectors
        shifts = np.array(
            [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]
        )

        # Create periodic images
        all_positions = []
        all_kinds = []
        atom_indices = []

        for i, atom in enumerate(atoms):
            img_positions = atom.r + shifts @ self.p.T  # Correct matrix multiplication
            all_positions.append(img_positions)
            all_kinds.extend([atom.kind] * len(shifts))
            atom_indices.extend([i] * len(shifts))

        all_positions = np.concatenate(all_positions)
        all_kinds = np.array(all_kinds)
        atom_indices = np.array(atom_indices)

        # Build KDTree
        tree = KDTree(all_positions)

        # Get indexes of Ti atoms
        ti_idxs = [i for i, atom in enumerate(atoms) if atom.kind == self.kinds[1]]
        ti_positions = np.array([atoms[i].r for i in ti_idxs])

        self.neighbors = {}
        for i, ti_pos in tqdm(
            zip(ti_idxs, ti_positions), desc="Calculating neighbors", unit="atom"
        ):
            nearby_indices = tree.query_ball_point(ti_pos, cutoff_radius)
            neighbors = []

            for idx in nearby_indices:
                if atom_indices[idx] != i and all_kinds[idx] != self.kinds[1]:
                    dist_vector = self.minimum_image_distance(
                        ti_pos, all_positions[idx]
                    )
                    if all(abs(d) < c for d, c in zip(dist_vector, cutoff)):
                        neighbors.append(atom_indices[idx])

            self.neighbors[i] = neighbors

        # Check that you have the correct surrounding atoms
        for i in ti_idxs:
            neighbor_kinds = [atoms[j].kind for j in self.neighbors[i]]
            if (
                sum(kind == self.kinds[0] for kind in neighbor_kinds) != 8
                or sum(kind == self.kinds[2] for kind in neighbor_kinds) != 6
                or len(neighbor_kinds) != 14
            ):
                print("Error: adjust cutoffs to include all atoms in supercell")
                exit()

    def minimum_image_distance(self, x1, x2):
        """
        Minimal image convention: x1 is the center (Ti atom)
        """
        delta = np.dot(
            x2 - x1, np.linalg.inv(self.p)
        )  # Transform into fractional coordinates
        delta -= np.round(delta)  # Apply periodic boundary conditions
        return np.dot(delta, self.p)  # Transform back to Cartesian coordinates

    def dipoles(self):
        self.dipoles = np.zeros((len(self.frames), len(self.neighbors), 3))
        for n in trange(len(self.frames), desc="Calculating dipoles", unit="frame"):
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
                    dR = self.minimum_image_distance(
                        self.frames[n].atoms[i].r, self.frames[n].atoms[j].r
                    )
                    self.dipoles[n, k, :] += w * qB * dR

    def print_extxyz(self):
        """
        print extxyz with the dipole information only sitting on Ti atoms
        The dipoles are labelled as forces, just to ease their visualization in Ovito.
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
                            P[
                                k,
                                0,
                            ],
                            P[
                                k,
                                1,
                            ],
                            P[
                                k,
                                2,
                            ],
                            cos_theta[k],
                        )
                    )
        print("Printed results in file:", self.system + "-dipoles.xyz")

    def plot_dipoles_2D(
        self, n, namefile, Ngrid=25, pdf=False, plane="xz", dpi_png=200
    ):
        """
        Plot dipoles in the x-z plane
        assign a value to each point in space that is essentially the Pz
        Ngrid = 25 seems to be a good compromize to visualize dipoles
        Dipoles are added, and not averaged.

        self.dipoles = np.zeros((nframes,nunitcells,3))
        """

        # dipoles direction and corresponding positions of Ti atoms (center of the dipoles)
        Pz = self.dipoles[n, :, 2]
        R = np.array([self.frames[n].atoms[k].r for k in self.neighbors.keys()])

        # max values
        Pz_max = np.max(np.abs(self.dipoles[:, :, 2]))
        Pz_min = -Pz_max

        # remap Pz into a grid
        Pz_grid = np.zeros((Ngrid, Ngrid))
        for na in range(len(Pz)):
            ix = int(R[na, 0] / self.p[0, 0] * Ngrid)
            iy = int(R[na, 1] / self.p[1, 1] * Ngrid)
            iz = int(R[na, 2] / self.p[2, 2] * Ngrid)
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
                extent=[0, self.p[0, 0], 0, self.p[2, 2]],
                vmax=Pz_max,
                vmin=Pz_min,
                cmap="jet",
                interpolation="lanczos",
            )
        if plane == "yz":
            plt.xlabel("$y$ (Å)")
            plt.imshow(
                Pz_grid,
                extent=[0, self.p[1, 1], 0, self.p[2, 2]],
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
        Plot dipoles in 3D

        visualization_mode:
        * 0: quiver plot
        * 1: 3D arrows
        * 2: tube arrows
        """

        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")
        ax.set_xlabel("x (Å)", fontsize="16")
        ax.set_ylabel("y (Å)", fontsize="16")
        ax.set_zlabel("z (Å)", fontsize="16")
        # plt.tick_params(axis='both', which='major', labelsize=14, width=2)
        ax.set_xlim((0, self.p[0, 0]))
        ax.set_ylim((0, self.p[1, 1]))
        ax.set_zlim((0, self.p[2, 2]))
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
        # plt.rcParams.update({'font.size': 20})
        plt.subplots_adjust(right=0.85)
        # cos angle between the dipole and the z-axis
        cos_theta = self.dipoles[n, :, 2] / np.linalg.norm(
            self.dipoles[n, :, :], axis=1
        )

        for i, k in enumerate(self.neighbors.keys()):

            # k indexes the Ti atoms
            r, p = self.frames[n].atoms[k].r, self.dipoles[n, :, :]
            color = plt.cm.jet(
                (cos_theta[i] + 1) / 2
            )  # Map cosine values to the range [0, 1]

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
        Plot angles histograms
        """

        vectors = self.dipoles[n, :, :]

        # Compute polar angles (phi) in radians
        r = np.linalg.norm(vectors, axis=1)
        phi = np.arccos(vectors[:, 2] / r)

        # Define number of bins
        num_bins = 36  # For example, 36 bins for 5-degree intervals

        # Compute histogram for phi
        phi_bins = np.linspace(0, np.pi, num_bins + 1)
        phi_hist, phi_bin_edges = np.histogram(phi, bins=phi_bins)

        # Compute bin centers
        phi_bin_centers = (phi_bin_edges[:-1] + phi_bin_edges[1:]) / 2

        # Create semi-polar plot for phi
        fig_phi, ax_phi = plt.subplots(subplot_kw={"projection": "polar"})
        bars_phi = ax_phi.bar(
            phi_bin_centers,
            phi_hist,
            width=(np.pi / num_bins),
            edgecolor="k",
            color=cm.jet(1 - phi_bin_centers / np.pi),
        )
        ax_phi.set_title("Polar Angle (Phi) Histogram", fontsize=16)
        ax_phi.set_theta_zero_location("N")  # Set 0 degrees at the top (North)
        ax_phi.set_theta_direction(-1)  # Set clockwise direction

        # set limits
        Nmax = 45
        ax_phi.set_ylim(0, Nmax)  # Adjust radial limits to fit the data

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

    def __init__(self, data, p):
        self.N = int(data[0].strip())
        self.p = p
        self.atoms = []
        for line in data[2:]:
            parts = line.split()
            kind = parts[0]
            r = np.array([float(x) for x in parts[1:4]])
            if len(parts) > 15:
                Z = np.array([float(x) for x in parts[4:13]]).reshape((3, 3))
                F = np.array([float(x) for x in parts[13:16]])
            else:
                Z = np.zeros((3, 3))
                F = np.zeros(3)
            self.atoms.append(Atom(kind, r, Z, F))


class Atom:
    def __init__(self, kind, r, Z, F):
        self.kind = kind
        self.r = r
        self.Z = Z
        self.F = F


# --------------------------------------------------------------------------
# ---------------------------- PRODUCTION ----------------------------------
# --------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]

materials = {
    "BaTiO3-full": (
        "BaTiO3-666-T300-hysteresis.xyz",
        ["Ba", "Ti", "O"],
        [0.125, 1.00, 0.50],
    ),
    "BaTiO3-test": (
        "BaTiO3-666-T300-hysteresis.xyz",
        ["Ba", "Ti", "O"],
        [0.125, 1.00, 0.50],
    ),
    "BaTiO3-54000-200ps": ("BaTiO3.xyz", ["Ba", "Ti", "O"], [0.125, 1.00, 0.50]),
}

if system in materials:
    infile, kinds, weights = materials[system]
else:
    print("Material not implemented!")
    exit()

# Print pdf or png
do_plot = True
pdf = False

# Parallel implementation
if __name__ == "__main__":

    start_time = time.time()
    T = Trajectory(f"{system}/{infile}", kinds, weights)
    end_time = time.time()
    print("Time to parse file: {:.2f} s".format(end_time - start_time))

    if do_plot:
        print("Plotting results ...")
        start_time = time.time()
        with Pool() as pool:
            for n in range(T.N_steps):
                pool.apply_async(
                    T.plot_dipoles_2D(
                        n, system + "/" + system + "-" + str(n) + "-2D", pdf=pdf
                    )
                )
                # pool.apply_async(T.plot_dipoles_3D(n,system+"/"+system+"-"+str(n)+"-3D", pdf=pdf))
                # pool.apply_async(T.plot_angles(n,system+"/"+system+"-"+str(n)+"-theta",  pdf=pdf))
        pool.close()
        pool.join()
        end_time = time.time()
        print("Time to process all steps: {:.2f} s".format(end_time - start_time))


# print("To make a movie, open QuickTime and go to File > Open_Image_Sequence")

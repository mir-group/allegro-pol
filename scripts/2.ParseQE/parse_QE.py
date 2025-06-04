# ==============================
# QE XML PARSER FOR POLARIZATION
# ==============================

"""
Author: S. Falletta

This script parses QE xml files obtained performing calculations under finite
electric fields, and returns an extxyz file with the dataset containing energy,
atomic positions, forces, polarization, polarizability, stress, Born charges
in eV units. In addition, histograms of the input data are plotted into a pdf
file to double check the results.

How to run:
• SiO2:   python3 parse_QE.py SiO2
• BaTiO3: python3 parse_QE.py BaTiO3

Before using this script, make sure your folders are organized as follows:
Nomenclature for input folder: {system}/data/{prefix}-E{axis}-{idf}
where idf is the frame index. Here are a few examples:

For SiO2:
• system = "SiO2"
• prefix = "SiO2-T300", "SiO2-T600"
• axis   = "0" for zero field, "x" for Efield along x, ...
example: SiO2/data/SiO2-E0-T300-0

For BaTiO3:
• system = "BaTiO3"
• prefix = "BaTiO3", "BaTiO3-wall"
• axis   = "0" for zero field, "x" for Efield along x, ...
example: BaTiO3/data/BaTiO3-E0-0

Each {system}/data/{prefix}-E{axis}-{idf} folder must contain the QE xml file,
which we here denote {xmlname}.xml

This convention on the nomenclature is needed as Born charges and polarizability require
combining QE xmlfiles obtained at different runs pertaining to different electric field
conditions. If one wishes changing this convention, this can be done by customizing the
keyword 'filename' in the subroutine 'parse_xml' of the class 'Material'.

Once organized the files, insert the information regarding the folders' nomenclature at
the end of this script, in the section PRODUCTION, where additional details are given:

• nframes:           number of frames per each set of data
• atnum:             dictionary of atomic numbers
• exclude_frames:    set of frames to skip in the writing of the extxyz
• check_dielectrics: double check your inputs by determining the high-frequency dielectric constant
                     and by checking that the acoustic sum rule for Born charges is enforced. This is
                     also useful to identify exclude_frames.

Units in the QE xml file:
• Energy in Hartree
• Forces in Hartree/Bohr
• Coordinates in Bohr
• Stress in Hartree/Bohr**2
• Efield in Ry a.u.
• Polarization in Ry a.u.
• electron charge in sqrt(2)

Units in the extxyz file:
• Energy in eV
• Forces in eV/A
• Coordinates in A
• Stress in eV/A**2
• Efield in V/A
• Polarization in e*A
• Born charges in e
• Polarizabilities in e*A^2*V^{-1}
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update(
    {"font.size": 24, "legend.fontsize": 20, "legend.handlelength": 0.5}
)

# Constants
bohr2A = 0.529177249
A2bohr = 1.8897259886
hartree2eV = 27.211396641308
eps0const = 5.5263499562 * 10**-3  # in [e * Volt^{-1} * Ansgrom^{-1}]
kB = 8.617333262145e-5  # in eV K-1

# unit conversions for QE xml
Eunit = hartree2eV
Funit = hartree2eV / bohr2A
Runit = bohr2A
Sunit = hartree2eV / (bohr2A**3) * (-1)
Efieldunit = hartree2eV / (bohr2A * np.sqrt(2))
Punit = bohr2A / np.sqrt(2.0)
volunit = bohr2A**3
dinv = {0: "x", 1: "y", 2: "z"}
dfull = {"0": 0, "x": 1, "y": 2, "z": 3}

# --------------------------------------------------------------------------
# --------------------------- PLOT FUNCTIONS -------------------------------
# --------------------------------------------------------------------------


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


def plot_init(label_x, label_y, title):
    f = plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


# --------------------------------------------------------------------------
# --------------------- PARSER FOR QE XMLFILE ------------------------------
# --------------------------------------------------------------------------


class Atom:
    def __init__(self, kind, r):
        self.kind = kind
        self.r = np.array(r)


class QE_xmlfile:
    """
    Parser for QE xml file
    """

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = f.readlines()
        self.atoms = []

    def get_lattparam(self):
        self.p = np.zeros((3, 3))
        for i, line in enumerate(self.data):
            if "<a1>" in line:
                self.p[0, :] = (
                    np.array(
                        [
                            float(x)
                            for x in self.data[i]
                            .replace("<a1>", "")
                            .replace("</a1>", "")
                            .split()
                        ]
                    )
                    * Runit
                )
                self.p[1, :] = (
                    np.array(
                        [
                            float(x)
                            for x in self.data[i + 1]
                            .replace("<a2>", "")
                            .replace("</a2>", "")
                            .split()
                        ]
                    )
                    * Runit
                )
                self.p[2, :] = (
                    np.array(
                        [
                            float(x)
                            for x in self.data[i + 2]
                            .replace("<a3>", "")
                            .replace("</a3>", "")
                            .split()
                        ]
                    )
                    * Runit
                )
                self.vol = np.dot(self.p[0, :], np.cross(self.p[1, :], self.p[2, :]))
                break

    def get_metric(self):
        self.g = np.zeros((3, 3))
        self.g[0, :] = self.p[0, :] / np.linalg.norm(self.p[0, :])
        self.g[1, :] = self.p[1, :] / np.linalg.norm(self.p[1, :])
        self.g[2, :] = self.p[2, :] / np.linalg.norm(self.p[2, :])

    def get_nat(self):
        for line in self.data:
            if "nat=" in line:
                self.nat = int(line.split()[1].split('"')[1])
                break

    def get_energy(self):
        for line in self.data:
            if "<etot>" in line:
                self.etot = (
                    float(line.replace("<etot>", "").replace("</etot>", "")) * Eunit
                )

    def get_atoms(self):
        for i, line in enumerate(self.data):
            if "<atomic_positions>" in line:
                for j in range(self.nat):
                    L = self.data[i + 1 + j].split()
                    kind = L[1].split('"')[1]
                    x = float(L[2].split(">")[1]) * Runit
                    y = float(L[3]) * Runit
                    z = float(L[4].split("<")[0]) * Runit
                    self.atoms.append(Atom(kind, [x, y, z]))
                break

    def get_forces(self):
        for i, line in enumerate(self.data):
            if "<forces rank=" in line:
                for j in range(self.nat):
                    L = self.data[i + 1 + j].split()
                    self.atoms[j].forces = np.array([float(x) for x in L]) * Funit
                break

    def get_polarization(self):
        for i, line in enumerate(self.data):
            if "<electronicDipole>" in line:
                L = self.data[i]
                self.Pel = np.array([float(x) for x in self.data[i + 1].split()])
                L = self.data[i + 3]
                Px = float(L.split("e>")[1].split(" ")[0])
                Py = float(L.split()[1])
                Pz = float(L.split("</")[0].split(" ")[-1])
                self.Pion = np.array([Px, Py, Pz])
                self.P = np.array(
                    [float(x) + float(y) for x, y in zip(self.Pel, self.Pion)]
                )
                self.P *= Punit
                self.P = self.unit_pol(self.P)
                break

    def get_stress(self):
        self.S = np.zeros((3, 3))
        for i, line in enumerate(self.data):
            if "<stress rank" in line:
                self.S[0, :] = (
                    np.array([float(x) for x in self.data[i + 1].split()]) * Sunit
                )
                self.S[1, :] = (
                    np.array([float(x) for x in self.data[i + 2].split()]) * Sunit
                )
                self.S[2, :] = (
                    np.array([float(x) for x in self.data[i + 3].split()]) * Sunit
                )
                break

    def get_efield(self):
        for line in self.data:
            if "<electric_field_vector>" in line:
                L = line.split(">")[1].split("</")[0].split(" ")
                self.Efield = (
                    np.array([float(L[0]), float(L[3]), float(L[6])]) * Efieldunit
                )
                break

    def unit_pol(self, P):
        """
        1) Convert polarization to fractional coordinates
        2) Do modulo of Pq, all in fractional coordinates
        3) Convert back to Cartesian coordinates
        """
        pol_mod_frac = np.dot(np.linalg.inv(self.g), self.p).diagonal()
        P_frac = np.dot(self.g, P)
        Pnew = P_frac % (np.sign(P_frac) * pol_mod_frac)
        Pnew = np.where(Pnew > 0.5 * pol_mod_frac, Pnew - pol_mod_frac, Pnew)
        Pnew = np.where(Pnew < -0.5 * pol_mod_frac, Pnew + pol_mod_frac, Pnew)
        Pnew = np.dot(np.linalg.inv(self.g), Pnew)
        return Pnew


# --------------------------------------------------------------------------
# --------------------- DATASET CONSTRUCTION--------------------------------
# --------------------------------------------------------------------------


class Material:

    def __init__(
        self,
        system,
        nframes,
        nat,
        atnum,
        prefix,
        xmlname,
        check_dielectrics=False,
        exclude_frames=[],
    ):
        """
        system:            name of the material
        nframes:           total number of frames
        nat:               total number of atoms
        atnum:             atomic numbers
        check_dielectrics: calculate the high-frequency dielectric constant, to make sure that values of P at a given structure lie within the same branch
        exclude_frames:    insert index of frames to exclude

        nomenclature for files: {system}/data/{prefix}-E{axis}-{idf}
        """
        self.system = system
        self.nframes = nframes
        self.nat = nat
        self.atnum = atnum
        self.prefix = prefix
        self.xmlname = xmlname
        self.check_dielectrics = check_dielectrics
        self.exclude_frames = exclude_frames

        # Physical quantities (here 4 stands for efield 0,x,y,z)
        self.p = np.zeros((self.nframes, 3, 3))
        self.g = np.zeros((self.nframes, 3, 3))
        self.E = np.zeros((self.nframes, 4))
        self.R = np.zeros((self.nframes, self.nat, 3))
        self.F = np.zeros((self.nframes, self.nat, 3, 4))
        self.S = np.zeros((self.nframes, 3, 3, 4))
        self.P = np.zeros((self.nframes, 3, 4))
        self.q = np.zeros((self.nframes, self.nat, 3, 3))
        self.α = np.zeros((self.nframes, 3, 3))
        self.at = np.zeros(self.nat)
        self.εinf = np.zeros((self.nframes, 3))

        # Parse xml
        self.parse_xml()

    def parse_xml(self):
        """
        parse xml files with:
        * 0: zero electric field
        * x: electric field along x
        * y: electric field along y
        * z: electric field along z
        """

        print("Material:", self.system)
        print("Parsing data from QE xml")

        # Loop over frames
        for idf in range(self.nframes):

            # skip exclude_frames
            if idf in self.exclude_frames:
                continue

            # parse calculations at zero and finite electric field for the given frame
            d = {"x": 0, "y": 1, "z": 2}

            self.xml = {}
            for axis in ["0", "x", "y", "z"]:

                filename = "{:s}/data/{:s}-E{:s}-{:s}/{:s}.xml".format(
                    self.system, self.prefix, axis, str(idf), self.xmlname
                )
                self.xml[axis] = QE_xmlfile(filename)
                self.xml[axis].get_nat()
                self.xml[axis].get_lattparam()
                self.xml[axis].get_metric()
                self.xml[axis].get_energy()
                self.xml[axis].get_atoms()
                self.xml[axis].get_forces()
                self.xml[axis].get_polarization()
                self.xml[axis].get_efield()
                self.xml[axis].get_stress()

            # store physical quantities of the zero field calculation
            self.p[idf, 0, :] = self.xml["0"].p[0, :]
            self.p[idf, 1, :] = self.xml["0"].p[1, :]
            self.p[idf, 2, :] = self.xml["0"].p[2, :]
            self.g[idf, 0, :] = self.xml["0"].g[0, :]
            self.g[idf, 1, :] = self.xml["0"].g[1, :]
            self.g[idf, 2, :] = self.xml["0"].g[2, :]
            self.E[idf, 0] = self.xml["0"].etot
            self.S[idf, 0, :, 0] = self.xml["0"].S[0, :]
            self.S[idf, 1, :, 0] = self.xml["0"].S[1, :]
            self.S[idf, 2, :, 0] = self.xml["0"].S[2, :]
            self.P[idf, :, 0] = self.xml["0"].P[:]
            for n in range(self.xml["0"].nat):
                A = self.xml["0"].atoms[n]
                self.at[n] = self.atnum[A.kind]
                self.R[idf, n, :] = A.r
                self.F[idf, n, :, 0] = A.forces
            for axis in ["x", "y", "z"]:
                i = d[axis]
                self.E[idf, i + 1] = self.xml[axis].etot
                self.S[idf, 0, :, i + 1] = self.xml[axis].S[0, :]
                self.S[idf, 1, :, i + 1] = self.xml[axis].S[1, :]
                self.S[idf, 2, :, i + 1] = self.xml[axis].S[2, :]
                self.P[idf, :, i + 1] = self.xml[axis].P[:]
                for n in range(self.xml["0"].nat):
                    A = self.xml[axis].atoms[n]
                    self.F[idf, n, :, i + 1] = A.forces

            # calculate Born charges and polarizabilities
            for axis in ["x", "y", "z"]:
                self.get_born_charges(
                    self.q[idf, :, :, :], self.xml["0"], self.xml[axis], d[axis]
                )
                self.get_polarizabilities(
                    self.α[idf, :, :], self.xml["0"], self.xml[axis], d[axis]
                )

            # calculate high-frequency dielectric constant
            if self.check_dielectrics:
                for i in range(3):
                    self.εinf[idf, i] = 1 + self.α[idf, i, i] / (
                        eps0const * self.xml["0"].vol
                    )
                print(
                    "{:d}, εinf = [{:.8f} {:.8f} {:.8f}]".format(
                        idf, self.εinf[idf, 0], self.εinf[idf, 1], self.εinf[idf, 2]
                    )
                )

        if self.check_dielectrics:

            # Dielectric constants
            εinf_sum = np.sum(self.εinf, axis=0) / (
                self.nframes - len(self.exclude_frames)
            )
            print(
                "ε∞ = 1  + 4pi/V * α = [{:.4f},{:.4f},{:.4f}]".format(
                    round(εinf_sum[0], 4), round(εinf_sum[1], 4), round(εinf_sum[2], 4)
                )
            )

            # Check neutrality condition Born charges
            qsum = np.sum(self.q, axis=(0, 1)) / self.nframes
            print(
                "\nCharge neutrality Zb \n [ {:.2f} {:.2f} {:.2f} \n   {:.2f} {:.2f} {:.2f} \n    {:.2f} {:.2f} {:.2f} ]\n".format(
                    qsum[0, 0],
                    qsum[0, 1],
                    qsum[0, 2],
                    qsum[1, 0],
                    qsum[1, 1],
                    qsum[1, 2],
                    qsum[2, 0],
                    qsum[2, 1],
                    qsum[2, 2],
                )
            )

        # remove exclude_frames
        if len(self.exclude_frames) > 0:
            print("frames removed = ", self.exclude_frames)
            self.exclude_frames = np.array(sorted(self.exclude_frames))
            for frame_del in self.exclude_frames:
                self.E = np.delete(self.E, frame_del, axis=0)
                self.R = np.delete(self.R, frame_del, axis=0)
                self.F = np.delete(self.F, frame_del, axis=0)
                self.S = np.delete(self.S, frame_del, axis=0)
                self.P = np.delete(self.P, frame_del, axis=0)
                self.q = np.delete(self.q, frame_del, axis=0)
                self.α = np.delete(self.α, frame_del, axis=0)
                self.p = np.delete(self.p, frame_del, axis=0)
                self.g = np.delete(self.g, frame_del, axis=0)
                self.εinf = np.delete(self.εinf, frame_del, axis=0)
                self.exclude_frames -= 1
                self.nframes -= 1

    def get_var_P(self, P):
        """
        calculate variance of polarization
        """
        return np.var(P, axis=0)

    def get_born_charges(self, q, xml0, xml, i):
        """
        calculate Born charges
        """
        for n in range(xml0.nat):
            for j in range(3):
                q[n, i, j] = (
                    xml.atoms[n].forces[j] - xml0.atoms[n].forces[j]
                ) / xml.Efield[i]

    def get_polarizabilities(self, α, xml0, xml, j):
        """
        get electronic polarizability (structure is fixed)
        """
        for i in range(3):
            α[i, j] = (xml.P[i] - xml0.P[i]) / xml.Efield[j]


class merge_all:

    def __init__(self, S_vec, iprint):
        """
        S_vec:  vector of Material class
        iprint: 0 (zero Efield), 1 (Efield along x), 2 (Efield along y), 3 (Efield along z)
        """

        self.system = S_vec[0].system
        self.nframes_tot = sum([Sz.nframes for Sz in S_vec])
        self.nat = S_vec[0].nat
        self.S_vec = S_vec
        self.iprint = iprint

        # quantities for npz
        self.p = np.zeros((self.nframes_tot, 3, 3))
        self.E = np.zeros((self.nframes_tot, 4))
        self.R = np.zeros((self.nframes_tot, self.nat, 3))
        self.F = np.zeros((self.nframes_tot, self.nat, 3, 4))
        self.S = np.zeros((self.nframes_tot, 3, 3, 4))
        self.P = np.zeros((self.nframes_tot, 3, 4))
        self.q = np.zeros((self.nframes_tot, self.nat, 3, 3))
        self.α = np.zeros((self.nframes_tot, 3, 3))
        self.at = np.zeros(self.nat)

        # join all quantities into npz arrays to plot histograms
        if all(n == nat[0] for n in nat):
            nprev = 0
            for i in range(len(S_vec)):
                for idf in range(S_vec[i].nframes):
                    self.p[idf + nprev, :, :] = S_vec[i].p[idf, :, :]
                    self.E[idf + nprev, :] = S_vec[i].E[idf, :]
                    self.R[idf + nprev, :, :] = S_vec[i].R[idf, :, :]
                    self.F[idf + nprev, :, :, :] = S_vec[i].F[idf, :, :, :]
                    self.S[idf + nprev, :, :, :] = S_vec[i].S[idf, :, :, :]
                    self.P[idf + nprev, :, :] = S_vec[i].P[idf, :, :]
                    self.q[idf + nprev, :, :, :] = S_vec[i].q[idf, :, :, :]
                    self.α[idf + nprev, :, :] = S_vec[i].α[idf, :, :]
                nprev += S_vec[i].nframes
            self.at = S_vec[0].at
        else:
            print("Number of atoms must be the same across all frames")
            exit()

    def print_extxyz(self):
        namefile = self.system + ".xyz"
        with open(self.system + "/" + namefile, "w") as f:
            nprev = 0
            for M in S_vec:
                for idf in range(M.nframes):
                    f.write("{:d}\n".format(M.xml["0"].nat))
                    f.write(
                        'Lattice="{:s}" Properties=species:S:1:pos:R:3:born_charge:R:9:forces:R:3 polarization="{}" polarizability="{}" original_dataset_index={} total_energy={:s} stress="{}" pbc="T T T" \n'.format(
                            " ".join(str(x) for x in M.p[idf, :, :].ravel()),
                            " ".join(str(x) for x in M.P[idf, :, self.iprint].ravel()),
                            " ".join(str(x) for x in M.α[idf, :, :].ravel()),
                            str(idf + nprev),
                            str(M.E[idf, iprint]),
                            " ".join(
                                str(x) for x in M.S[idf, :, :, self.iprint].ravel()
                            ),
                        )
                    )
                    for n in range(M.nat):
                        str_format = "{:5s} " + "{:14.10f} " * 15 + " \n"
                        f.write(
                            str_format.format(
                                M.xml["0"].atoms[n].kind,
                                M.R[idf, n, 0],
                                M.R[idf, n, 1],
                                M.R[idf, n, 2],
                                M.q[idf, n, 0, 0],
                                M.q[idf, n, 0, 1],
                                M.q[idf, n, 0, 2],
                                M.q[idf, n, 1, 0],
                                M.q[idf, n, 1, 1],
                                M.q[idf, n, 1, 2],
                                M.q[idf, n, 2, 0],
                                M.q[idf, n, 2, 1],
                                M.q[idf, n, 2, 2],
                                M.F[idf, n, 0, self.iprint],
                                M.F[idf, n, 1, self.iprint],
                                M.F[idf, n, 2, self.iprint],
                            )
                        )
                nprev += M.nframes
        print("Printed results in file:", namefile)

    def plot_all_hystogram(self, pdf, iprint=0):

        # Energy
        self.plot_histogram(
            pdf, self.E[:, self.iprint] / self.nat, "Energy per atom (eV)", title="$E$"
        )

        # Forces
        for i in range(3):
            self.plot_histogram(
                pdf,
                self.F[:, :, i, self.iprint].ravel(),
                "Force (eV/A)",
                title="$F_{:s}$".format(str(dinv[i])),
            )

        # Polarization
        for i in range(3):
            self.plot_histogram(
                pdf,
                self.P[:, i, self.iprint].ravel(),
                "Polarization (e$\cdot$A)",
                title="$P_{:s}$".format(str(dinv[i])),
            )

        # born charge
        for i in range(3):
            for j in range(3):
                self.plot_histogram(
                    pdf,
                    self.q[:, :, i, j].ravel(),
                    "Born charges (e)",
                    title="$Z^*_{{{:s}}}$".format(str(i) + str(j)),
                )

        # polarizability
        for i in range(3):
            for j in range(3):
                self.plot_histogram(
                    pdf,
                    1 + self.α[:, i, j].ravel(),
                    "Polarizability (e$\cdot$A$^{2}\cdot$V$^{-1}$)",
                    title="$α_{{{:s}}}$".format(str(i) + str(j)),
                )

    def plot_histogram(self, pdf, data_array, xlabel, title, num_bins=30):
        plot_init(xlabel, "Counts", title)
        plt.hist(
            data_array,
            bins=num_bins,
            edgecolor="black",
            density=False,
            alpha=0.7,
            lw=1.5,
            label="Histogram",
        )
        plt.title(title)
        pdf.savefig()


# --------------------------------------------------------------------------
# ----------------------------- PRODUCTION ---------------------------------
# --------------------------------------------------------------------------

assert len(sys.argv) > 1, "Specify Material"

system = sys.argv[1]
"""
• nframes:           number of frames for each subsystem
• nat:               number of atoms in each subsystem
• atnum:             atomic numbers (fixed across all subsystem)
• prefix:            prefixes subfolder name
• xmlname:           name of xml files (they must be all the same)
• check_dielectrics: check dielectric constant
• exclude_frames:    broken frames for each subsystem
• iprint:            print E and F at zero Efield (0), Efield x (1), Efield y (2), Efield z (3)
"""

if system == "SiO2":
    nframes = [101, 101]
    nat = [72, 72]
    atnum = {"Si": 14, "O": 8}
    prefix = ["SiO2-T300", "SiO2-T600"]
    xmlname = ["SiO2", "SiO2"]
    check_dielectrics = True
    exclude_frames = [[], [5]]
    iprint = 0

elif system == "BaTiO3":
    nframes = [67, 16]
    nat = [135, 135]
    atnum = {"Ba": 56, "Ti": 22, "O": 8}
    prefix = ["BaTiO3", "BaTiO3-wall"]
    xmlname = ["BaTiO3", "BaTiO3"]
    check_dielectrics = True
    exclude_frames = [[18, 35, 38, 44, 46, 49, 58, 60], [13, 14]]
    iprint = 0

# ADD HERE YOUR MATERIAL WITH SPECIFIC FOLDER NOMENCLATURE CONVENTION

else:
    print("System not implemented, insert data or check your input")
    exit()

if not (len(nframes) == len(nat) == len(prefix) == len(exclude_frames)):
    print("Error: check dimensions inputs!")

S_vec = []
with PdfPages(system + "/" + system + ".pdf") as pdf:

    # Parse all inputs
    for i in range(len(nframes)):
        print("\n--------------------------------------------")
        print("Dataset", i)
        S_vec.append(
            Material(
                system=system,
                nframes=nframes[i],
                nat=nat[i],
                atnum=atnum,
                prefix=prefix[i],
                xmlname=xmlname[i],
                check_dielectrics=check_dielectrics,
                exclude_frames=exclude_frames[i],
            )
        )

    # print npz and plot hystograms for the full dataset
    S_merged = merge_all(S_vec, iprint)
    S_merged.print_extxyz()
    S_merged.plot_all_hystogram(pdf)

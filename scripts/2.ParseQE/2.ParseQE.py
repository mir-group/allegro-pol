#!/usr/bin/env python3

# ==============================
# QE XML PARSER FOR POLARIZATION
# ==============================

"""
This script parses QE xml files obtained performing calculations under finite
electric fields, and returns an extxyz file with the dataset containing
energy, atomic positions, forces, polarization, polarizability, stress, Born
charges in eV units. In addition, histograms of the input data are plotted
into a pdf file to double check the results.

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

Each {system}/data/{prefix}-E{axis}-{idf} folder must contain the QE xml
file, which we here denote {xmlname}.xml

This convention on the nomenclature is needed as Born charges and
polarizability require combining QE xmlfiles obtained at different runs
pertaining to different electric field conditions. If one wishes changing
this convention, this can be done by customizing the keyword 'filename' in
the subroutine 'parse_xml' of the class 'Material'.

Once organized the files, insert the information regarding the folders'
nomenclature at the end of this script, in the section PRODUCTION, where
additional details are given:

• nframes:           number of frames per each set of data
• atnum:             dictionary of atomic numbers
• exclude_frames:    set of frames to skip in the writing of the extxyz
• check_dielectrics: double check your inputs by determining the
                     high-frequency dielectric constant and by checking
                     that the acoustic sum rule for Born charges is
                     enforced. This is also useful to identify
                     exclude_frames.

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

Author: S. Falletta
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
eps0const = 5.5263499562e-3  # in [e * Volt^{-1} * Ansgrom^{-1}]
kB = 8.617333262145e-5  # in eV K-1

# Unit conversions for QE xml
Eunit = hartree2eV
Funit = hartree2eV / bohr2A
Runit = bohr2A
Sunit = hartree2eV / (bohr2A**3) * (-1)
Efieldunit = hartree2eV / (bohr2A * np.sqrt(2))
Punit = bohr2A / np.sqrt(2.0)
volunit = bohr2A**3

# Direction mappings
dinv = {0: "x", 1: "y", 2: "z"}
dfull = {"0": 0, "x": 1, "y": 2, "z": 3}

# Material configurations
MATERIAL_CONFIGS = {
    "SiO2": {
        "nframes": [1, 1],
        "nat": [72, 72],
        "atnum": {"Si": 14, "O": 8},
        "prefix": ["SiO2-T300", "SiO2-T600"],
        "xmlname": ["SiO2", "SiO2"],
        "check_dielectrics": True,
        "exclude_frames": [[], []],  # list of indexes to excllude
        "iprint": 0,
    },
    "BaTiO3": {
        "nframes": [1, 1],
        "nat": [135, 135],
        "atnum": {"Ba": 56, "Ti": 22, "O": 8},
        "prefix": ["BaTiO3", "BaTiO3-wall"],
        "xmlname": ["BaTiO3", "BaTiO3"],
        "check_dielectrics": True,
        "exclude_frames": [[], []],  # list of indexes to excllude
        "iprint": 0,
    },
}


def axis_settings(ax):
    """Configure axis settings for plots."""
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(which="major", width=3.0, length=12, direction="in")
        axis.set_tick_params(which="minor", width=3.0, length=6, direction="in")

    ax.yaxis.set_ticks_position("both")


def plot_init(label_x, label_y, title):
    """Initialize plot with standard settings."""
    plt.figure(figsize=(6, 6), dpi=60)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.19, top=0.79, right=0.95)
    ax = plt.gca()
    axis_settings(ax)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


class Atom:
    """Class representing an atom with its properties."""

    def __init__(self, kind, r):
        self.kind = kind
        self.r = np.array(r)


class QE_xmlfile:
    """Parser for QE xml file."""

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = f.readlines()
        self.atoms = []

    def get_lattparam(self):
        """Get lattice parameters."""
        self.p = np.zeros((3, 3))
        for i, line in enumerate(self.data):
            if "<a1>" in line:
                for j in range(3):
                    self.p[j, :] = (
                        np.array(
                            [
                                float(x)
                                for x in self.data[i + j]
                                .replace(f"<a{j+1}>", "")
                                .replace(f"</a{j+1}>", "")
                                .split()
                            ]
                        )
                        * Runit
                    )
                self.vol = np.dot(self.p[0, :], np.cross(self.p[1, :], self.p[2, :]))
                break

    def get_metric(self):
        """Get metric tensor."""
        self.g = np.zeros((3, 3))
        for i in range(3):
            self.g[i, :] = self.p[i, :] / np.linalg.norm(self.p[i, :])

    def get_nat(self):
        """Get number of atoms."""
        for line in self.data:
            if "nat=" in line:
                self.nat = int(line.split()[1].split('"')[1])
                break

    def get_energy(self):
        """Get total energy."""
        for line in self.data:
            if "<etot>" in line:
                self.etot = (
                    float(line.replace("<etot>", "").replace("</etot>", "")) * Eunit
                )

    def get_atoms(self):
        """Get atomic positions."""
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
        """Get atomic forces."""
        for i, line in enumerate(self.data):
            if "<forces rank=" in line:
                for j in range(self.nat):
                    L = self.data[i + 1 + j].split()
                    self.atoms[j].forces = np.array([float(x) for x in L]) * Funit
                break

    def get_polarization(self):
        """Get electronic and ionic polarization."""
        for i, line in enumerate(self.data):
            if "<electronicDipole>" in line:
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
        """Get stress tensor."""
        self.S = np.zeros((3, 3))
        for i, line in enumerate(self.data):
            if "<stress rank" in line:
                for j in range(3):
                    self.S[j, :] = (
                        np.array([float(x) for x in self.data[i + j + 1].split()])
                        * Sunit
                    )
                break

    def get_efield(self):
        """Get electric field vector."""
        for line in self.data:
            if "<electric_field_vector>" in line:
                L = line.split(">")[1].split("</")[0].split(" ")
                self.Efield = (
                    np.array([float(L[0]), float(L[3]), float(L[6])]) * Efieldunit
                )
                break

    def unit_pol(self, P):
        """
        Process polarization vector:
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


class Material:
    """
    Class for handling QE calculation data.

    Parameters
    ----------
    system : str
        Name of the material
    nframes : int
        Total number of frames
    nat : int
        Total number of atoms
    atnum : dict
        Dictionary of atomic numbers
    prefix : str
        Prefix for file naming
    xmlname : str
        Name of XML file
    check_dielectrics : bool, optional
        Whether to check dielectric constants, by default False
    exclude_frames : list, optional
        List of frames to exclude, by default []
    """

    def __init__(
        self,
        system,
        nframes,
        nat,
        atnum,
        prefix,
        xmlname,
        check_dielectrics=False,
        exclude_frames=None,
    ):
        self.system = system
        self.nframes = nframes
        self.nat = nat
        self.atnum = atnum
        self.prefix = prefix
        self.xmlname = xmlname
        self.check_dielectrics = check_dielectrics
        self.exclude_frames = exclude_frames or []

        # Initialize arrays for physical quantities
        # 4 stands for efield 0,x,y,z
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

        # Parse XML files
        self.parse_xml()

    def parse_xml(self):
        """Parse XML files for all frames and field conditions."""
        print(f"Material: {self.system}")
        print("Parsing data from QE xml")

        # Loop over frames
        for idf in range(self.nframes):
            if idf in self.exclude_frames:
                continue

            # Parse calculations at zero and finite electric field
            d = {"x": 0, "y": 1, "z": 2}
            self.xml = {}

            for axis in ["0", "x", "y", "z"]:
                filename = (
                    f"{self.system}/data/{self.prefix}-E{axis}-{idf}/"
                    f"{self.xmlname}.xml"
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

            # Store physical quantities of the zero field calculation
            self.store_zero_field_data(idf)

            # Store data for finite fields
            for axis in ["x", "y", "z"]:
                i = d[axis]
                self.store_finite_field_data(idf, i, axis)

            # Calculate Born charges and polarizabilities
            for axis in ["x", "y", "z"]:
                self.get_born_charges(
                    self.q[idf, :, :, :], self.xml["0"], self.xml[axis], d[axis]
                )
                self.get_polarizabilities(
                    self.α[idf, :, :], self.xml["0"], self.xml[axis], d[axis]
                )

            # Calculate high-frequency dielectric constant
            if self.check_dielectrics:
                self.check_dielectric_constants(idf)

        if self.check_dielectrics:
            self.print_dielectric_summary()

        # Remove excluded frames
        if self.exclude_frames:
            self.remove_excluded_frames()

    def store_zero_field_data(self, idf):
        """Store data from zero field calculation."""
        self.p[idf, :, :] = self.xml["0"].p
        self.g[idf, :, :] = self.xml["0"].g
        self.E[idf, 0] = self.xml["0"].etot
        self.S[idf, :, :, 0] = self.xml["0"].S
        self.P[idf, :, 0] = self.xml["0"].P

        for n in range(self.xml["0"].nat):
            A = self.xml["0"].atoms[n]
            self.at[n] = self.atnum[A.kind]
            self.R[idf, n, :] = A.r
            self.F[idf, n, :, 0] = A.forces

    def store_finite_field_data(self, idf, i, axis):
        """Store data from finite field calculation."""
        self.E[idf, i + 1] = self.xml[axis].etot
        self.S[idf, :, :, i + 1] = self.xml[axis].S
        self.P[idf, :, i + 1] = self.xml[axis].P

        for n in range(self.xml["0"].nat):
            A = self.xml[axis].atoms[n]
            self.F[idf, n, :, i + 1] = A.forces

    def check_dielectric_constants(self, idf):
        """Calculate and print dielectric constants for a frame."""
        for i in range(3):
            self.εinf[idf, i] = 1 + self.α[idf, i, i] / (eps0const * self.xml["0"].vol)
        print(
            f"{idf}, εinf = [{self.εinf[idf, 0]:.8f} "
            f"{self.εinf[idf, 1]:.8f} {self.εinf[idf, 2]:.8f}]"
        )

    def print_dielectric_summary(self):
        """Print summary of dielectric properties."""
        # Dielectric constants
        εinf_sum = np.sum(self.εinf, axis=0) / (self.nframes - len(self.exclude_frames))
        print(
            "ε∞ = 1 + 4pi/V * α = "
            f"[{εinf_sum[0]:.4f}, {εinf_sum[1]:.4f}, {εinf_sum[2]:.4f}]"
        )

        # Check neutrality condition Born charges
        qsum = np.sum(self.q, axis=(0, 1)) / self.nframes
        print("\nCharge neutrality Zb")
        print(f"[ {qsum[0, 0]:.2f} {qsum[0, 1]:.2f} {qsum[0, 2]:.2f}")
        print(f"  {qsum[1, 0]:.2f} {qsum[1, 1]:.2f} {qsum[1, 2]:.2f}")
        print(f"  {qsum[2, 0]:.2f} {qsum[2, 1]:.2f} {qsum[2, 2]:.2f} ]\n")

    def remove_excluded_frames(self):
        """Remove excluded frames from data arrays."""
        print(f"frames removed = {self.exclude_frames}")
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
        """Calculate variance of polarization."""
        return np.var(P, axis=0)

    def get_born_charges(self, q, xml0, xml, i):
        """Calculate Born effective charges."""
        for n in range(xml0.nat):
            for j in range(3):
                q[n, i, j] = (
                    xml.atoms[n].forces[j] - xml0.atoms[n].forces[j]
                ) / xml.Efield[i]

    def get_polarizabilities(self, α, xml0, xml, j):
        """Calculate electronic polarizability (structure is fixed)."""
        for i in range(3):
            α[i, j] = (xml.P[i] - xml0.P[i]) / xml.Efield[j]


def merge_data(S_vec):
    """
    Merge data from multiple Material instances.

    Parameters
    ----------
    S_vec : list
        List of Material instances to merge

    Returns
    -------
    dict
        Dictionary containing merged arrays
    """
    nat = [S.nat for S in S_vec]
    if not all(n == nat[0] for n in nat):
        print("Number of atoms must be the same across all frames")
        exit()

    nframes_tot = sum(S.nframes for S in S_vec)
    nat = S_vec[0].nat

    # Initialize arrays for merged data
    merged = {
        "p": np.zeros((nframes_tot, 3, 3)),
        "E": np.zeros((nframes_tot, 4)),
        "R": np.zeros((nframes_tot, nat, 3)),
        "F": np.zeros((nframes_tot, nat, 3, 4)),
        "S": np.zeros((nframes_tot, 3, 3, 4)),
        "P": np.zeros((nframes_tot, 3, 4)),
        "q": np.zeros((nframes_tot, nat, 3, 3)),
        "α": np.zeros((nframes_tot, 3, 3)),
        "at": np.zeros(nat),
    }

    # Copy data from each Material instance
    nprev = 0
    for S in S_vec:
        for idf in range(S.nframes):
            idx = nprev + idf
            merged["p"][idx] = S.p[idf]
            merged["E"][idx] = S.E[idf]
            merged["R"][idx] = S.R[idf]
            merged["F"][idx] = S.F[idf]
            merged["S"][idx] = S.S[idf]
            merged["P"][idx] = S.P[idf]
            merged["q"][idx] = S.q[idf]
            merged["α"][idx] = S.α[idf]
        nprev += S.nframes

    merged["at"] = S_vec[0].at
    return merged


def write_extxyz(system, merged_data, S_vec, iprint):
    """Write merged data in extended XYZ format."""
    namefile = f"{system}.xyz"
    filepath = f"{system}/{namefile}"

    with open(filepath, "w") as f:
        nprev = 0
        for M in S_vec:
            for idf in range(M.nframes):
                write_frame(f, M, idf, nprev, merged_data, iprint)
            nprev += M.nframes
    print(f"Printed results in file: {namefile}")


def write_frame(f, M, idf, nprev, data, iprint):
    """Write a single frame in extended XYZ format."""
    f.write(f"{M.xml['0'].nat}\n")

    # Write header with lattice and properties
    lattice = " ".join(str(x) for x in M.p[idf, :, :].ravel())
    polarization = " ".join(str(x) for x in M.P[idf, :, iprint].ravel())
    polarizability = " ".join(str(x) for x in M.α[idf, :, :].ravel())
    stress = " ".join(str(x) for x in M.S[idf, :, :, iprint].ravel())

    header = (
        f'Lattice="{lattice}" '
        "Properties=species:S:1:pos:R:3:born_charge:R:9:forces:R:3 "
        f'polarization="{polarization}" '
        f'polarizability="{polarizability}" '
        f"original_dataset_index={idf + nprev} "
        f"total_energy={M.E[idf, iprint]} "
        f'stress="{stress}" '
        'pbc="T T T"\n'
    )
    f.write(header)

    # Write atomic data
    for n in range(M.nat):
        write_atom(f, M, idf, n, iprint)


def write_atom(f, M, idf, n, iprint):
    """Write data for a single atom."""
    atom = M.xml["0"].atoms[n]
    forces = M.F[idf, n, :, iprint]
    born_charges = M.q[idf, n, :, :].ravel()
    pos = M.R[idf, n, :]

    data = [atom.kind] + pos.tolist() + born_charges.tolist() + forces.tolist()
    fmt = "{:5s}" + " {:14.10f}" * 15 + "\n"
    f.write(fmt.format(*data))


def plot_histograms(merged_data, system, iprint):
    """Plot histograms for all physical quantities."""
    with PdfPages(f"{system}/{system}.pdf") as pdf:
        # Energy
        plot_histogram(
            pdf,
            merged_data["E"][:, iprint] / merged_data["at"].size,
            "Energy per atom (eV)",
            title="$E$",
        )

        # Forces
        for i in range(3):
            plot_histogram(
                pdf,
                merged_data["F"][:, :, i, iprint].ravel(),
                "Force (eV/A)",
                title=f"$F_{dinv[i]}$",
            )

        # Polarization
        for i in range(3):
            plot_histogram(
                pdf,
                merged_data["P"][:, i, iprint].ravel(),
                "Polarization (e$\cdot$A)",
                title=f"$P_{dinv[i]}$",
            )

        # Born charges
        for i in range(3):
            for j in range(3):
                plot_histogram(
                    pdf,
                    merged_data["q"][:, :, i, j].ravel(),
                    "Born charges (e)",
                    title=f"$Z^*_{{{i}{j}}}$",
                )

        # Polarizability
        for i in range(3):
            for j in range(3):
                plot_histogram(
                    pdf,
                    1 + merged_data["α"][:, i, j].ravel(),
                    "Polarizability (e$\cdot$A$^{2}\cdot$V$^{-1}$)",
                    title=f"$α_{{{i}{j}}}$",
                )


def plot_histogram(pdf, data_array, xlabel, title, num_bins=30):
    """Plot histogram of data array."""
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


if __name__ == "__main__":

    assert len(sys.argv) > 1, "Specify Material"

    system = sys.argv[1]

    # Get system configuration
    if system not in MATERIAL_CONFIGS:
        print("System not implemented, insert data or check your input")
        exit()

    config = MATERIAL_CONFIGS[system]

    # Validate input dimensions
    dims_match = (
        len(config["nframes"])
        == len(config["nat"])
        == len(config["prefix"])
        == len(config["exclude_frames"])
    )
    if not dims_match:
        print("Error: check dimensions inputs!")
        exit()

    # Process data
    S_vec = []
    for i in range(len(config["nframes"])):
        print("\n" + "-" * 44)
        print(f"Dataset {i}")
        S_vec.append(
            Material(
                system=system,
                nframes=config["nframes"][i],
                nat=config["nat"][i],
                atnum=config["atnum"],
                prefix=config["prefix"][i],
                xmlname=config["xmlname"][i],
                check_dielectrics=config["check_dielectrics"],
                exclude_frames=config["exclude_frames"][i],
            )
        )

    # Merge data and generate outputs
    merged_data = merge_data(S_vec)
    write_extxyz(system, merged_data, S_vec, config["iprint"])
    plot_histograms(merged_data, system, config["iprint"])

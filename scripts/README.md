# Tutorial for using Allegro-pol to predict vibrational, dielectric, and ferroelectric properties of materials

In this tutorial, we provide a step-by-step guide for pre-processing and post-processing tasks to determine:

- Vibrational and dielectric responses of SiO₂.
- Ferroelectric response of BaTiO₃.

### Pre-processing scripts:

1. `1.CalculatorQE`: Contains scripts to generate QE input files for calculations under zero field and finite electric fields along x, y, and z. Folder nomenclature is interfaced with the parsing scripts.
2. `2.ParseQE`: Contains scripts to parse QE XML files to create extxyz file for Allegro-pol, containing energy, forces, polarization, Born charges, and polarizability.

### Post-processing scripts:

3. `3.Parity`: Contains scripts to plot parity comparisons between DFT data and corresponding Allegro predictions obtained through `nequip-evaluate`.
4. `4.Dielectric`: Contains scripts to calculate the dielectric constant during a QuantumEspresso or LAMMPS structural relaxation in the presence of an electric field.
5. `5.Vibrational`: Contains scripts to calculate the IR spectrum and the real and imaginary parts of the dielectric constant from MLMD at fixed volume and temperature, performed with the Allegro-pol model.
6. `6.Hysteresis`: Contains scripts to calculate ferroelectric hysteresis from MLMD at fixed volume and temperature under a sinusoidal electric field, performed with the Allegro-pol model. When provided with multiple datafiles for statistics, the script averages polarization values across all trajectories.
7. `7.Dipoles`: Contains scripts to parse an extxyz structural file containing Born charges and plot ferroelectric dipoles (2D, 3D plots), angle distributions with respect to the z-axis, and extxyz files with dipoles per unit cell. Works only for orthorhombic perovskite structures.
8. `8.CoerciveField`: Contains scripts to plot the coercive field versus temperature, performing statistics at each temperature.
9. `9.Mobility`: Contains scripts for mobility calculations.

### Additional Utilities:

- `combine_LAMMPS_restarts`: Contains scripts to combine LAMMPS restart files.

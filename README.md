# SaturateUnitCell
UnitCellSaturator is a Python tool designed to saturate unit cells for transition from plane-wave to molecular quantum chemistry codes.
It supports parsing and converting VASP (POSCAR) and XYZ file formats, sorting atoms, defining bonds, and visualizing distance histograms. The script can saturate unsaturated atoms in the molecular structure based on user defined parameters (.json).

##Features
- Load and Parse Files: Supports VASP (POSCAR) and XYZ file formats. (currently CONTCAR/POSCAR with selective dynamics can not be recognized)
- Convert VASP to XYZ: Easily convert VASP files to XYZ format. 
- Sort Atoms: Sort atoms by specified element order.
- Define Bonds: Identify and visualize bonds, including distance histograms.
- Saturate Structure: Saturate the molecular structure according to valence rules.

##Requirements
- Python 3.x
- NumPy
- Matplotlib
- JSON configuration files (dic_val_e.json and config.json)

##Usage
python main.py file_name molecule_name [--convert] [--sort] [--define-bond] [--saturate]

##Command Line Arguments
- file_name: The name of the file to load (POSCAR or .xyz).
- molecule_name: The name of the molecule.
- --convert: Convert VASP to XYZ format.
- --sort: Sort atoms by specified element order.
- --define-bond: Define bonds and plot distance histograms.
- --saturate: Saturate the molecule structure.

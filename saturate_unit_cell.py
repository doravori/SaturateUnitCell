import numpy as np
import re
import sys
import argparse
import logging
import matplotlib.pyplot as plt
import math
from config import s_conf
from functools import reduce
import json
import os


# Configure logging
logger = logging.getLogger(__name__)

class Atom:
    def __init__(self, element: str, xyz_position: list[float]):
        self.element = element
        assert len(xyz_position) == 3, "XYZ must be 3 long"
        self.xyz_position = np.array(xyz_position, dtype=np.float64)

    def __repr__(self):
        return (
            f"{self.element:3}"
            + " "
            + " ".join(map(lambda x: f"{x: 20.16f}", self.xyz_position))
            + "\n"
        )

class Molecule:
    def __init__(self, name: str, atoms: list[Atom]):
        self.name = name
        self.atoms = np.array(atoms)

    def __repr__(self):
        return f"Name: {self.name}\nAtoms:\n" + "".join(map(str, self.atoms))

    def get_cart_coord(self, scaling_factor: float, unit_cell: list[float]):
        unit_c = np.array(unit_cell).reshape((3, 3)).astype(np.float64)
        list_of_atoms = []

        for a in range(self.atoms.size):
            list_of_coord = [self.atoms[a].xyz_position * unit_c[u] * scaling_factor for u in range(3)]
            cart_atom = reduce(np.add, list_of_coord)
            list_of_atoms.append(cart_atom)

        cart_mol = np.array(list_of_atoms, dtype=np.float64)
        logger.debug(f"Cartesian coordinates: {cart_mol}")

        for idx in range(len(self.atoms)):
            self.atoms[idx].xyz_position = cart_mol[idx]

    def sort_mol(self, right_order: list[str]):
        present_elements = set(atom.element for atom in self.atoms)
        for element in right_order:
            if element not in present_elements:
                raise ValueError(f"Element {element} not found in the molecule.")

        sorted_molecule = []
        for r in range(len(right_order)):
            for a in range(self.atoms.size):
                if self.atoms[a].element == right_order[r]:
                    sorted_molecule.append(self.atoms[a])

        if len(sorted_molecule) != len(self.atoms):
            raise ValueError("Not all atoms were sorted correctly.")

        for idx in range(len(self.atoms)):
            self.atoms[idx] = sorted_molecule[idx]

    @classmethod
    def load_vasp_file(cls, molecule_name: str, file_name: str):
        logger.debug("Converting VASP File to Molecule")
        with open(file_name, "r") as f:
            file = f.read()

        scaling_factor_pattern = re.compile(r'^\s*([\d\.Ee\+\-]+)\s*$', re.MULTILINE)
        unit_cell_pattern = re.compile(r'^\s*([\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+)\s*$', re.MULTILINE)
        elements_info_pattern = re.compile(r'^(\s*[A-Za-z]+\s*)+\n(\s*\d+\s*)+$', re.MULTILINE)
        atoms_pattern = re.compile(r'^(Direct|Cartesian)\n((?:\s*[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s*\n)+)', re.MULTILINE)

        scaling_factor_match = scaling_factor_pattern.search(file)
        assert scaling_factor_match, "No scaling factor found"
        scaling_factor = float(scaling_factor_match.group(1))

        # Extract the three lines of the unit cell
        unit_cell_lines = re.findall(r'^\s*[\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+\s*$', file, re.MULTILINE)[:3]
        assert len(unit_cell_lines) == 3, "Unit cell should have exactly three lines"
        unit_cell = [float(value) for line in unit_cell_lines for value in line.split()]

        res_elements_info = elements_info_pattern.search(file)
        assert res_elements_info, "No element information found"

        element_info_lines = res_elements_info.group(0).strip().split("\n")
        assert len(element_info_lines) == 2, "Element information block should have two lines"

        element_names = element_info_lines[0].split()
        element_numbers = element_info_lines[1].split()
        logger.debug(f"Element names: {element_names}")
        logger.debug(f"Element numbers: {element_numbers}")

        element_info = []
        for e_name, e_number in zip(element_names, element_numbers):
            element_info.extend([e_name] * int(e_number))

        res_atoms_match = atoms_pattern.search(file)
        assert res_atoms_match, "No atom coordinates found"

        coord_type = res_atoms_match.group(1)
        atoms_lines = res_atoms_match.group(2).strip().split("\n")
        res_atoms = [line.split() for line in atoms_lines]
        logger.debug(f"Atoms: {res_atoms}")

        assert len(element_info) == len(res_atoms), "Mismatch in number of atoms"

        atom_obj = [
            Atom(element_info[idx], [float(x) for x in atom])
            for idx, atom in enumerate(res_atoms)
        ]

        molecule = cls(name=molecule_name, atoms=atom_obj)

        if coord_type == "Direct":
            molecule.get_cart_coord(scaling_factor, unit_cell)

        return molecule

    @classmethod
    def load_xyz_file(cls, molecule_name: str, file_name: str):
        logger.debug("Converting XYZ File to Molecule")
        with open(file_name, "r") as f:
            file = f.read()

        atoms_pattern = re.compile(r"([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
        atoms_info = atoms_pattern.findall(file)

        atom_obj = [
            Atom(element, [float(x), float(y), float(z)])
            for element, x, y, z in atoms_info
        ]

        return cls(name=molecule_name, atoms=atom_obj)

def get_input_type(file_name: str):
    patterns = {
        "vasp": re.compile(r"^(\s*[A-Za-z]+\s*)+\n(\s*\d+\s*)+$", re.MULTILINE),
        "xyz": re.compile(r"((\s?[A-Z][a-z]?)(\s*[-+]?\d+.?\d+){3})"),
    }

    with open(file_name, "r") as f:
        file = f.read()

    for name, pattern in patterns.items():
        if pattern.search(file):
            logger.debug(f"File {file_name} matches {name} pattern")
            return name
    logger.error(f"File {file_name} does not match any known format")
    return None

def get_molecule(molecule_name: str, file_name: str):
    input_type = get_input_type(file_name)
    if input_type:
        file_method = getattr(Molecule, f"load_{input_type}_file")
        return input_type, file_method(molecule_name=molecule_name, file_name=file_name)
    else:
        raise ValueError("Unsupported file type")

def print_coordinates(molecule: Molecule):
    print(f"Coordinates for molecule: {molecule.name}")
    for atom in molecule.atoms:
        print(f"{atom.element}: {atom.xyz_position}")

def write_xyz_out(mymolecule: Molecule):
    output_file_name = f"{mymolecule.name}.xyz"
    with open(output_file_name, "w") as output:
        output.write(f"{len(mymolecule.atoms)}\n{mymolecule.name}\n")
        for atom in mymolecule.atoms:
            output.write(f"{atom.element} {' '.join(f'{coord: 22.16f}' for coord in atom.xyz_position)}\n")
    logger.debug(f"XYZ file written to {output_file_name}")

def VASP_to_XYZinput(molecule_name: str, file_name: str):
    input_type, mymolecule = get_molecule(molecule_name=molecule_name, file_name=file_name)
    if input_type == "vasp":
        write_xyz_out(mymolecule)
    else:
        raise ValueError("The input file is not a VASP (POSCAR) file")


def histogram_distances_default(distance_list):
    hist, bin_edges = np.histogram(distance_list, bins=400, range=(0, 40))
    return hist, bin_edges

def plot_histogram(hist, bin_edges, title="Distance Histogram"):
    logger.debug(f"{hist} und {bin_edges}")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], edgecolor='black')
    plt.ylabel("N(r)")
    plt.xlabel("$r$")
    plt.title(title)
    plt.show()

def distance(c1, c2):
    x_dist = (c1[0] - c2[0]) ** 2
    y_dist = (c1[1] - c2[1]) ** 2
    z_dist = (c1[2] - c2[2]) ** 2
    return math.sqrt(x_dist + y_dist + z_dist)

def define_bond(mymolecule: Molecule):
    distances_of_atoms = np.array([])
    for a1 in range(mymolecule.atoms.size):
        for a2 in mymolecule.atoms[a1 + 1:]:
            distances_of_atoms = np.append(
                distances_of_atoms,
                distance(mymolecule.atoms[a1].xyz_position, a2.xyz_position),
            )
    logger.debug(f"{distances_of_atoms.size}")
    logger.debug(f"{np.amax(distances_of_atoms)}")
    dist_hist_1, bin_edges_1 = histogram_distances_default(distances_of_atoms)
    plot_histogram(dist_hist_1, bin_edges_1, title="All Atom Distances")

    element_list = np.array([])
    for e in mymolecule.atoms:
        element_list = np.append(element_list, e.element)
    unique, counts = np.unique(element_list, return_counts=True)
    l_unique = unique.tolist()
    mymolecule.sort_mol(l_unique)

    for idx1 in range(unique.size):
        for idx2 in unique[idx1:]:
            specific_distances = np.array([])
            for a1 in range(mymolecule.atoms.size):
                if unique[idx1] == mymolecule.atoms[a1].element:
                    for a2 in mymolecule.atoms[a1 + 1:]:
                        if (
                            idx2 == a2.element
                            and (
                                mymolecule.atoms[a1].xyz_position != a2.xyz_position
                            ).all()
                        ):
                            specific_distances = np.append(
                                specific_distances,
                                distance(
                                    mymolecule.atoms[a1].xyz_position, a2.xyz_position
                                ),
                            )
            d_name = f"sp_distance_{unique[idx1]}_{idx2}"
            globals()[d_name] = np.copy(specific_distances)
            logger.debug(f"{d_name}")
            dist_hist_1, bin_edges_1 = histogram_distances_default(globals()[d_name])
            plot_histogram(dist_hist_1, bin_edges_1, title=f"Distances: {unique[idx1]}-{idx2}")

def find_connectivity(mymolecule: Molecule):
  
    with open("dic_val_e.json", "r") as f:
        val_ele = json.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)

    bonds_from = config["bonds_from"]
    bonds_to = config["bonds_to"]
    cut_off = config["cut_off"]

    molecule_vdW_rad = []
    molecule_val_ele = []

    for a in mymolecule.atoms:
        for e in val_ele:
            if a.element == e:
                molecule_val_ele.append(val_ele[e])

    define_bonds = False
    if define_bonds:
        define_bond(mymolecule=mymolecule)

    not_saturated = np.array([])
    not_saturated_index = []

    for idx in range(len(bonds_from)):
        for a1 in range(mymolecule.atoms.size):
            count_con = 0
            if bonds_from[idx] == mymolecule.atoms[a1].element:
                for sub in range(len(bonds_to[idx])):
                    for a2 in mymolecule.atoms:
                        if bonds_to[idx][sub] == a2.element:
                            dis = distance(
                                mymolecule.atoms[a1].xyz_position, a2.xyz_position
                            )
                            if dis < cut_off[idx][sub]:
                                count_con += 1
                                if bonds_from[idx] == "O" and bonds_to[idx][sub] == "N":
                                    count_con += 1
            else:
                count_con = 10
            if count_con < val_ele[bonds_from[idx]][0]:
                not_saturated = np.append(not_saturated, mymolecule.atoms[a1])
                not_saturated_index.append(a1)

    return not_saturated_index



def create_groups(parts_of_mol: list[int], mymolecule: Molecule):
    with open("dic_val_e.json", "r") as f:
        val_ele = json.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)

    bonds_from = config["bonds_from"]
    bonds_to = config["bonds_to"]
    cut_off = config["cut_off"]

    groups_of_bonds = {}

    for a1 in range(len(parts_of_mol)):
        list_of_neigbours = []
        for idx1 in range(len(bonds_from)):
            if mymolecule.atoms[parts_of_mol[a1]].element == bonds_from[idx1]:
                for a2 in range(mymolecule.atoms.size):
                    for idx2 in range(len(bonds_to[idx1])):
                        if mymolecule.atoms[a2].element == bonds_to[idx1][idx2]:
                            dis = distance(
                                mymolecule.atoms[parts_of_mol[a1]].xyz_position,
                                mymolecule.atoms[a2].xyz_position,
                            )
                            if dis < cut_off[idx1][idx2]:
                                list_of_neigbours.append(a2)
        groups_of_bonds[parts_of_mol[a1]] = list_of_neigbours
    return groups_of_bonds


def saturate_structure(mymolecule: Molecule, unit_cell: list[float]):
    with open("dic_val_e.json", "r") as f:
        val_ele = json.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)

    to_saturate = {int(k): v for k, v in config["to_saturate"].items()}
    saturate_angle = {int(k): v for k, v in config["saturate_angle"].items()}
    saturate_bond = {int(k): v for k, v in config["saturate_bond"].items()}

    not_sat_index = find_connectivity(mymolecule=mymolecule)
    groups_of_bonds = create_groups(parts_of_mol=not_sat_index, mymolecule=mymolecule)

    vector_x = np.array([])
    vector_y = np.array([])
    vector_z = np.array([])

    for u in range(len(unit_cell)):
        if u < 3:
            vector_x = np.append(vector_x, unit_cell[u])
            vector_min_x = np.multiply(vector_x, -1)
        if u > 2 and u < 6:
            vector_y = np.append(vector_y, unit_cell[u])
            vector_min_y = np.multiply(vector_y, -1)
        if u > 5:
            vector_z = np.append(vector_z, unit_cell[u])
            vector_min_z = np.multiply(vector_z, -1)

    shifts_vectors = [
        vector_x,
        vector_min_x,
        vector_y,
        vector_min_y,
        vector_z,
        vector_min_z,
    ]

    bonds_from = config["bonds_from"]
    bonds_to = config["bonds_to"]
    cut_off = config["cut_off"]

    accepted_shifts = np.array([])

    for a1 in not_sat_index:
        for v in shifts_vectors:
            shifted_a = np.add(v, mymolecule.atoms[a1].xyz_position)
            shifted_n = list(
                map(
                    lambda n: np.add(v, mymolecule.atoms[n].xyz_position),
                    groups_of_bonds[a1],
                )
            )
            for idx1 in range(len(bonds_from)):
                if bonds_from[idx1] == mymolecule.atoms[a1].element:
                    connectivity = len(groups_of_bonds[a1])
                    initial_connectivity = len(groups_of_bonds[a1])
                    for a2 in mymolecule.atoms:
                        for idx2 in range(len(bonds_to[idx1])):
                            if a2.element == bonds_to[idx1][idx2]:
                                dis = distance(shifted_a, a2.xyz_position)
                                if dis < cut_off[idx1][idx2]:
                                    connectivity += 1
            accept_shift = True
            for id in range(len(groups_of_bonds[a1])):
                con_n = 1
                for idx1 in range(len(bonds_from)):
                    if bonds_from[idx1] == groups_of_bonds[a1][id]:
                        for m in mymolecule.atoms:
                            for idx2 in range(len(bonds_to[idx1])):
                                if m.element == bonds_to[idx1][idx2]:
                                    dis = distance(shifted_n[id], m.xyz_position)
                                    if dis < cut_off[idx1][idx2]:
                                        con_n += 1
                if (
                    con_n
                    < val_ele[mymolecule.atoms[groups_of_bonds[a1][id]].element][0]
                ):
                    accept_shift = False

            if (
                #connectivity == val_ele[mymolecule.atoms[a1].element][0]
                connectivity > initial_connectivity
                and accept_shift == True
            ):
                mymolecule.atoms[a1].xyz_position = shifted_a
                for gr in range(len(groups_of_bonds[a1])):
                    mymolecule.atoms[groups_of_bonds[a1][gr]].xyz_position = shifted_n[
                        gr
                    ]
    not_sat_index = find_connectivity(mymolecule=mymolecule)
    groups_of_bonds = create_groups(parts_of_mol=not_sat_index, mymolecule=mymolecule)

    while len(not_sat_index) > 0:
        for a1 in not_sat_index:
            valence = val_ele[mymolecule.atoms[a1].element][0]
            new_pos = [0, 0, 0]
            parallel_v = [0, 0, 0]
            perpendicular_v = [0, 0, 0]
            for a2 in groups_of_bonds[a1]:
                if valence == 2:
                    v_OSi = np.subtract(
                        mymolecule.atoms[a1].xyz_position,
                        mymolecule.atoms[a2].xyz_position,
                    )
                    v_Siz = [()]
                    for c in range(mymolecule.atoms[a2].xyz_position.size):
                        if c < 2:
                            v_Siz = np.append(v_Siz, 0)
                        else:
                            v_Siz = np.append(
                                v_Siz, mymolecule.atoms[a1].xyz_position[c]
                            )

                    perpendicular_v = np.cross(v_OSi, v_Siz)
                    parallel_v = np.add(parallel_v, v_OSi)

                if valence == 4:
                    if len(groups_of_bonds[a1]) == 3:
                        v_OSi = np.subtract(
                            mymolecule.atoms[a1].xyz_position,
                            mymolecule.atoms[a2].xyz_position,
                        )
                        new_pos = np.add(new_pos, v_OSi)
                    if len(groups_of_bonds[a1]) == 2:
                        v_OSi = np.subtract(
                            mymolecule.atoms[a1].xyz_position,
                            mymolecule.atoms[a2].xyz_position,
                        )
                        parallel_v = np.add(parallel_v, v_OSi)
                        if groups_of_bonds[a1].index(a2) == 0:
                            perpendicular_v = v_OSi
                        else:
                            perpendicular_v = np.cross(perpendicular_v, v_OSi)

            if valence == 2:
                length_parallel_step = (
                    math.cos(math.radians(180 - saturate_angle[valence])) * 0.95
                )
                length_perpendicular_step = (
                    math.sin(math.radians(180 - saturate_angle[valence])) * 0.95
                )

                scaling_parallel_step = length_parallel_step / np.linalg.norm(
                    parallel_v
                )
                scaling_perpendicular_step = length_perpendicular_step / np.linalg.norm(
                    perpendicular_v
                )

                parallel_v = np.multiply(parallel_v, scaling_parallel_step)
                perpendicular_v = np.multiply(
                    perpendicular_v, scaling_perpendicular_step
                )

                parallel_step = np.add(parallel_v, mymolecule.atoms[a1].xyz_position)
                perpendicular_step = np.add(perpendicular_v, parallel_step)

                mymolecule.atoms = np.append(
                    mymolecule.atoms,
                    Atom(element=to_saturate[valence], xyz_position=perpendicular_step),
                )

            if valence == 4 and len(groups_of_bonds[a1]) == 2:
                length_parallel_step = (
                    math.cos(math.radians(saturate_angle[valence] / 2)) * 1.65
                )
                length_perpendicular_step = (
                    math.sin(math.radians(saturate_angle[valence] / 2)) * 1.65
                )

                scaling_parallel_step = length_parallel_step / np.linalg.norm(
                    parallel_v
                )
                scaling_perpendicular_step = length_perpendicular_step / np.linalg.norm(
                    perpendicular_v
                )

                parallel_v = np.multiply(parallel_v, scaling_parallel_step)
                perpendicular_v = np.multiply(
                    perpendicular_v, scaling_perpendicular_step
                )
                perpendicular_v_neg = np.multiply(perpendicular_v, -1)

                parallel_step = np.add(parallel_v, mymolecule.atoms[a1].xyz_position)
                perpendicular_step = np.add(perpendicular_v, parallel_step)
                perpendicular_step_neg = np.add(perpendicular_v_neg, parallel_step)

                mymolecule.atoms = np.append(
                    mymolecule.atoms,
                    Atom(element=to_saturate[valence], xyz_position=perpendicular_step),
                )
                mymolecule.atoms = np.append(
                    mymolecule.atoms,
                    Atom(
                        element=to_saturate[valence],
                        xyz_position=perpendicular_step_neg,
                    ),
                )

            elif valence == 4 and len(groups_of_bonds[a1]) == 3:
                mymolecule.atoms = np.append(
                    mymolecule.atoms,
                    Atom(
                        element=to_saturate[valence],
                        xyz_position=np.add(new_pos, mymolecule.atoms[a1].xyz_position),
                    ),
                )
        not_sat_index = find_connectivity(mymolecule=mymolecule)
        groups_of_bonds = create_groups(
            parts_of_mol=not_sat_index, mymolecule=mymolecule
        )
        logger.debug(f"{not_sat_index} and {groups_of_bonds}")

    write_xyz_out(mymolecule=mymolecule)











def find_index_in_molecule(my_atoms: Atom, mymolecule: Molecule):
    for m in range(mymolecule.atoms.size):
        if np.array_equal(mymolecule.atoms[m].xyz_position, my_atoms.xyz_position):
            return m
        else:
            pass
    return "Not found"


def check_json_files():
    required_files = ["dic_val_e.json", "config.json"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' is missing.")

def get_unit_cell_from_user():
    print("Please enter the nine values for the unit cell, separated by spaces (e.g., 'a1 a2 a3 b1 b2 b3 c1 c2 c3'):")
    unit_cell_values = input().strip().split()
    if len(unit_cell_values) != 9:
        raise ValueError("You must enter exactly nine values for the unit cell.")
    return [float(value) for value in unit_cell_values]

def read_unit_cell_from_poscar(file_name: str):
    with open(file_name, "r") as f:
        file = f.read()
    unit_cell_lines = re.findall(r'^\s*[\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+\s+[\d\.\-\+Ee]+\s*$', file, re.MULTILINE)[:3]
    if len(unit_cell_lines) != 3:
        raise ValueError("Unit cell should have exactly three lines in POSCAR file")
    return [float(value) for line in unit_cell_lines for value in line.split()]
 


def main():
    parser = argparse.ArgumentParser(description="Load VASP or XYZ file and parse it into a Molecule object.")
    parser.add_argument("file_name", type=str, help="The name of the file to load (POSCAR or .xyz).")
    parser.add_argument("molecule_name", type=str, help="The name of the molecule.")
    parser.add_argument("--convert", action="store_true", help="Convert VASP to XYZ format.")
    parser.add_argument("--sort", action="store_true", help="Sort atoms by specified element order.")
    parser.add_argument("--define-bond", action="store_true", help="Define bonds and plot distance histograms.")
    parser.add_argument("--saturate", action="store_true", help="Saturate the molecule structure.")

    args = parser.parse_args()

    try:
        input_type = get_input_type(args.file_name)
        _, molecule = get_molecule(args.molecule_name, args.file_name)
        
        if args.sort or args.define_bond:
            print("Please enter the desired order of elements separated by spaces (e.g., 'C N O H Si'):")
            right_order = input().strip().split()
            molecule.sort_mol(right_order)
            print("Molecule sorted by the specified element order.")
        
        if args.convert:
            VASP_to_XYZinput(molecule_name=args.molecule_name, file_name=args.file_name)
            print(f"VASP file {args.file_name} converted to XYZ format.")
        
        if args.define_bond:
            define_bond(molecule)
        
        if args.saturate:
            check_json_files()
            if input_type == "vasp":
                unit_cell = read_unit_cell_from_poscar(args.file_name)
            else:
                unit_cell = get_unit_cell_from_user()
            saturate_structure(molecule, unit_cell)
            print(f"Molecule structure saturated.")

        print(molecule)
        print_coordinates(molecule)
    
    except ValueError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()

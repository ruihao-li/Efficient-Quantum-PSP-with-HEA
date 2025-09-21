from turn_decoder import read_xyz, bitstring_to_coords
import os
import numpy as np


def read_file(filename):
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def load_dataset(
    lattice: str,
    encoding: str,
    nearestneighbor: int = 1,
    matrix_year: str = "1996",
    directory: str = "classical_search/res/",
    size_of_set: None | int = None,
):
    """
    Input:
    matrix_year = '1985' or '1996' or '2003'
    """
    # directory = directory + lattice + "/"

    # if lattice == "fcc":
    #     directory = directory + "mj1996/"

    directory = directory + lattice + "/" + "mj" + matrix_year + "/"
    resname_list = next(os.walk(directory))[1]
    resname_list.sort()  # smallest number of aa first

    protein_dict = {}
    for resname in resname_list:
        # if resname[0] == "4" or resname[0] == "5" or resname[0] == "6":
        #    continue
        protein = ProteinData(resname, lattice, encoding)
        if not resname in protein_dict:
            protein_dict[resname] = protein

            # 1. Read the data
            # if lattice == "fcc":
            #     filenames = [f"{directory}{resname}/topobj.txt"]
            # elif lattice == "tetrahedral":
            #     filenames = [
            #         f"{directory}{resname}/{matrix_year}_energies/topobj.txt",
            #         f"{directory}{resname}/{str(int(matrix_year)-3)}_energies/topobj.txt",
            #         f"{directory}{resname}/{matrix_year}_matrix/topobj.txt",
            #         f"{directory}{resname}/{str(int(matrix_year)-3)}_matrix/topobj.txt",
            #         f"{directory}{resname}/topobj.txt",
            #     ]
            filenames = [f"{directory}{resname}/topobj.txt"]

            data = None
            for filename in filenames:
                data = read_file(filename)
                if data is not None:
                    break

            # 2. Split by line
            data_by_line = data.split("\n")

            # 3. Number of amino acids is on the second line
            protein.add_number_of_aa(int(data_by_line[1]))

            # if NN in resname then set protein.nn to the interger in front of NN in the resname, otherwise to 1
            if "1NN" in resname:
                protein.add_nn(1)
            elif "2NN" in resname:
                protein.add_nn(2)
            else:
                protein.add_nn(1)

            for line in data_by_line:
                # 4. Add folds in the dict folding_dict
                if len(line.split("\t")) == 3:  # tetrahedral
                    [energy, turns, bitstring] = line.split("\t")
                    protein.add_bit_txt(bitstring)

                    energy = float(energy) * 0.1  # some shift thing
                    turns = [int(turn) for turn in turns]
                    if (
                        turns[-1] == 1 and turns[-2] == 0
                    ):  # turns not ending with 01 are not allowed
                        protein.add_folding(energy, turns)

                elif len(line.split("\t")) == 2:  # fcc or bcc
                    [energy, turns] = line.split("\t")
                    energy = float(energy) * 0.1  # some shift thing
                    if lattice == "fcc":
                        turns = relabel_fcc_turns(turns)
                        if turns[-1] == 0:
                            if (
                                turns[-2] == 9
                                or turns[-2] == 8
                                or turns[-2] == 2
                                or turns[-2] == 0
                            ):
                                protein.add_folding(energy, turns)
                    elif lattice == "bcc":
                        turns = relabel_bcc_turns(turns)
                        if turns[-1] == 0:
                            if turns[-2] == 1 or turns[-2] == 3 or turns[-2] == 0:
                                protein.add_folding(energy, turns)

                elif len(line.split(" ")) == 4:
                    if len(protein.folding_dict) == 1:
                        protein.add_to_sequence(line[0])

            protein.add_ground_state()

    return_dict = {}
    # sort the proteins by number of amino acids
    protein_dict = dict(
        sorted(protein_dict.items(), key=lambda item: len(item[1].sequence))
    )

    for i, (key, protein) in enumerate(protein_dict.items()):
        if size_of_set is not None:
            if len(return_dict) >= size_of_set:
                break
        if nearestneighbor == protein.nn:
            return_dict[key] = protein

    return return_dict


def relabel_fcc_turns(turns: str) -> list[int]:
    """
    Convert the turns used in the classical exhaustive search to the turns used
    in the quantum encoding.
    0 to 0
    1 to 2
    2 to 4
    3 to 6
    4 to 8
    5 to 10
    6 to 1
    7 to 3
    8 to 5
    9 to 7
    a to 9
    b to 11
    """
    return_list = []
    for turn in turns:
        if turn == "0":
            return_list.append(0)
        elif turn == "1":
            return_list.append(2)
        elif turn == "2":
            return_list.append(4)
        elif turn == "3":
            return_list.append(6)
        elif turn == "4":
            return_list.append(8)
        elif turn == "5":
            return_list.append(10)
        elif turn == "6":
            return_list.append(1)
        elif turn == "7":
            return_list.append(3)
        elif turn == "8":
            return_list.append(5)
        elif turn == "9":
            return_list.append(7)
        elif turn == "a":
            return_list.append(9)
        elif turn == "b":
            return_list.append(11)
    return return_list


def relabel_bcc_turns(turns: str) -> list[int]:
    """
    Convert the turns used in the classical exhaustive search to the turns used
    in the quantum encoding.
    0 to 0
    1 to 1
    2 to 2
    3 to 3
    4 to 7
    5 to 6
    6 to 5
    7 to 4
    """
    return_list = []
    for turn in turns:
        if turn == "0":
            return_list.append(0)
        elif turn == "1":
            return_list.append(1)
        elif turn == "2":
            return_list.append(2)
        elif turn == "3":
            return_list.append(3)
        elif turn == "4":
            return_list.append(7)
        elif turn == "5":
            return_list.append(6)
        elif turn == "6":
            return_list.append(5)
        elif turn == "7":
            return_list.append(4)
    return return_list


class ProteinData:
    def __init__(self, resname, lattice, encoding):
        self.resname = resname
        self.sequence = ""
        self.lattice = lattice
        self.folding_dict = {}
        self.encoding = encoding
        self.number_of_aa = None
        self.nn = None
        self.ground_state_bitstring = None
        self.ground_energy = None
        self.bit_txt = None

    def __str__(self) -> str:
        return (
            "resname:"
            + self.resname
            + "\n"
            + "sequence:"
            + self.sequence
            + "\n"
            + "folding_dict:"
            + str(self.folding_dict)
            + "\n"
            + "number_of_aa:"
            + str(self.number_of_aa)
            + "\n"
            + "nn:"
            + str(self.nn)
            + "\n"
            + "ground_state_bitstring:"
            + str(self.ground_state_bitstring)
            + "\n"
            + "ground_energy:"
            + str(self.ground_energy)
            + "\n"
        )

    def add_to_sequence(self, aa):
        self.sequence += aa

    def add_number_of_aa(self, number_of_aa):
        self.number_of_aa = number_of_aa

    def add_folding(
        self,
        energy,
        turn_list,
    ):
        if str(turn_list) in self.folding_dict:
            self.folding_dict[str(turn_list)] = round(float(energy), 6)
            """
            raise ValueError(
                f"Folding {turn_list} already exists in the folding_dict with energy {self.folding_dict[str(turn_list)]}"
            )
            """
        else:
            self.folding_dict[str(turn_list)] = round(float(energy), 6)

    def add_nn(self, nn):
        self.nn = nn

    def add_bit_txt(self, bit_txt):
        if self.bit_txt == None:
            self.bit_txt = bit_txt

    def add_ground_state(self):
        exec("from turn_decoder import " + self.lattice + "_turns_to_bitstring")
        # go through the folding_dict and find the lowest energy
        # and the corresponding bitstring
        min_energy = 1e10
        min_turns = None
        for turns, energy in self.folding_dict.items():
            if energy < min_energy:
                min_energy = energy
                min_turns = turns
        self.ground_energy = min_energy
        self.ground_state_bitstring = eval(self.lattice + "_turns_to_bitstring")(
            min_turns, self.encoding
        )
        self.ground_state_turns = min_turns

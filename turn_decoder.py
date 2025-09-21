"""
Decoder

Input:
Take bitstring
Lattice structure: Tetrahedral, face-centered cubic (FCC), body-centered cubic
(BCC).

Out:
XYZ-file

TODO: add the lattice structures
"""

import numpy as np


def bitstring_to_coords(conf_bitstring: str, lattice: str, encoding: str) -> np.ndarray:
    """
    Convert a conformation bitstring to the Cartesian coordinates of the protein
    shape.

    Input:
        conf_bitstring (str): The turn sequence represented by binary numbers.
        lattice (str): The lattice structure ("tetrahedral", "fcc", or "bcc").
        encoding (str): The encoding of the bitstring. Unary or binary. Note
        that unary is not yet supported for FCC and BCC.

    Returns:
        np.ndarray: An array containing the coordinates of the amino acids in the sequence.
    """
    if lattice == "tetrahedral":
        # Coordinates of the 4 edges of a tetrahedron centered at 0. The vectors are normalized.
        coordinates = (
            3.8
            * (1.0 / np.sqrt(3))
            * np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]])
        )
        turn_seq = tetrahedral_bitstring_to_turns(conf_bitstring, encoding)[::-1]
        length_turns = len(turn_seq)
        positions = np.zeros((length_turns + 1, 3), dtype=float)

        for i in range(length_turns):
            positions[i + 1] = (-1) ** i * coordinates[turn_seq[i]]
        coords = positions.cumsum(axis=0)

    elif lattice == "fcc":
        # Coordinates of the 12 turns of a FCC lattice. The vectors are normalized.
        coordinates = (
            3.8
            * (1.0 / np.sqrt(2))
            * np.array(
                [
                    [1, 1, 0],
                    [-1, -1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [0, 1, 1],
                    [0, -1, -1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [1, 0, 1],
                    [-1, 0, -1],
                    [1, 0, -1],
                    [-1, 0, 1],
                ]
            )
        )
        turn_seq = fcc_bitstring_to_turns(conf_bitstring, encoding)[::-1]

        length_turns = len(turn_seq)
        positions = np.zeros((length_turns + 1, 3), dtype=float)
        for i in range(length_turns):
            # to avoid failing if the bitstring is outside of the feasible solution set
            if turn_seq[i] is not None:
                positions[i + 1] = (
                    positions[i] + coordinates[turn_seq[i]]
                )  # from fcc_protein_shape in fcc branch
            else:
                # If the turn is not feasible, the position is the same as the previous one (so that it can be penalized by the overlap penalty)
                positions[i + 1] = positions[i]
        # coords = positions.cumsum(axis=0)
        coords = positions

    elif lattice == "bcc":
        # Coordinates of the 8 turns of a BCC lattice. The vectors are normalized.
        coordinates = (
            3.8
            * (1.0 / np.sqrt(3))
            * np.array(
                [
                    [1, 1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                ]
            )
        )
        turn_seq = bcc_bitstring_to_turns(conf_bitstring, encoding)[::-1]

        length_turns = len(turn_seq)
        positions = np.zeros((length_turns + 1, 3), dtype=float)
        for i in range(length_turns):
            positions[i + 1] = positions[i] + coordinates[turn_seq[i]]
        coords = positions
    else:
        raise ValueError("Invalid lattice structure.")
    return coords


# TETRAHEDRAL
def tetrahedral_turns_to_bitstring(turns: list, encoding: str) -> list[int]:
    """
    Converts a list of integers representing the turns on a FCC lattice
    to a bitstring.

    Input:
        turns (list): A list of integers representing the turns.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    try:
        turns = [int(turn) for turn in turns]
    except:
        turns = eval(turns)
    turns = turns[::-1]
    if encoding == "unary":
        encoding = {0: "0001", 1: "0010", 2: "0100", 3: "1000"}
        bitstring = "".join([encoding[int(turn)] for turn in turns])
        bitstring = bitstring[8:]  # delete the first 8 bits that are locked
    elif encoding == "binary":
        encoding = {0: "00", 1: "01", 2: "10", 3: "11"}
        bitstring = "".join([encoding[int(turn)] for turn in turns])
        bitstring = bitstring[::-1]
        bitstring = bitstring[:-6] + bitstring[-5:]
        bitstring = bitstring[:-4]  # delete the first 4 bits that are locked
    else:
        raise ValueError("Invalid encoding.")
    return bitstring


def tetrahedral_bitstring_to_turns(bitstring: str, encoding: str) -> list[int]:
    """
    Converts a bitstring to a list of integers representing the turns on a
    tetrahedral lattice based on the following encoding:
    0: (1, 1, 1)   1: (-1, -1, 1)   2: (1, -1, -1)   3: (-1, 1, -1)
    -0: (-1, -1, -1)   -1: (1, 1, -1)   -2: (-1, 1, 1)   -3: (1, -1, 1)

    Input:
        bitstring (str): The bitstring to be converted.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    if encoding == "unary":
        encoding = {"0001": 0, "0010": 1, "0100": 2, "1000": 3}
        # Add the first two turns (to avoid rotational degeneracy) to the
        # beginning of the bitstring
        bitstring = bitstring + "10000100"
        bitstring = bitstring[::-1]
        # Check if length of bitstring is divisible by 4
        if len(bitstring) % 4 != 0:
            raise ValueError(
                "The bitstring length is not compatible with the unary encoding. It should be divisible by 4."
            )
        length_turns = len(bitstring) // 4
        turns = [encoding[bitstring[4 * i : 4 * (i + 1)]] for i in range(length_turns)]
        return turns[::-1]
    elif encoding == "binary":
        encoding = {"00": 0, "01": 1, "10": 2, "11": 3}
        # Add the first two turns (to avoid rotational degeneracy) to the
        # beginning of the bitstring
        bitstring = bitstring + "0010"
        # The amount of qubits needed to encode the turns will be 2(N-3) - 1 if
        # no side chain on second main bead or 2(N-3) otherwise.
        bitstring = bitstring[:-5] + "1" + bitstring[-5:]
        bitstring = bitstring[::-1]
        # Check if length of bitstring is divisible by 2
        if len(bitstring) % 2 != 0:
            raise ValueError(
                "The bitstring length is not compatible with the binary encoding. It should be divisible by 2."
            )
        length_turns = len(bitstring) // 2
        turns = [encoding[bitstring[2 * i : 2 * (i + 1)]] for i in range(length_turns)]
        return turns[::-1]
    else:
        raise ValueError("Invalid encoding.")


# FCC
def fcc_turns_to_bitstring(turns: list, encoding: str) -> list[int]:
    """
    Converts a list of integers representing the turns on a FCC lattice
    to a bitstring.

    Input:
        turns (list): A list of integers representing the turns.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    # if turns is a list then eval it to get the list of integers
    try:
        turns = [int(turn) for turn in turns]
    except:
        turns = eval(turns)
    turns = turns[::-1]
    if encoding == "unary":
        raise ValueError("Not implemented.")
    elif encoding == "binary":
        encoding_list = {
            0: "0000",
            1: "0011",
            2: "1100",
            3: "1111",
            4: "1001",
            5: "0101",
            6: "1010",
            7: "0110",
            8: "1000",
            9: "0100",
            10: "1011",
            11: "0111",
        }
        bitstring = "".join([encoding_list[int(turn)] for turn in turns])
        bitstring = bitstring[:6] + bitstring[8:]
        bitstring = bitstring[4:]
        bitstring = bitstring[::-1]
    else:
        raise ValueError("Invalid encoding.")
    return bitstring


def fcc_bitstring_to_turns(bitstring: str, encoding: str) -> list[int]:
    """
    Converts a bitstring to a list of integers representing the turns on a
    FCC lattice based on the following encoding:
    0: (1, 1, 0)   1: (-1, -1, 0)   2: (-1, 1, 0)   3: (1, -1, 0)
    4: (0, 1, 1)   5: (0, 1, -1)   6: (0, 1, -1)   7: (0, -1, 1)
    8: (1, 0, 1)   9: (-1, 0, -1)   10: (1, 0, -1)   11: (-1, 0, 1)

    Input:
        bitstring (str): The bitstring to be converted.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    if encoding == "unary":
        raise ValueError("Not implemented.")
    elif encoding == "binary":
        encoding = {
            "0000": 0,
            "0011": 1,
            "1100": 2,
            "1111": 3,
            "1001": 4,
            "0101": 5,
            "1010": 6,
            "0110": 7,
            "1000": 8,
            "0100": 9,
            "1011": 10,
            "0111": 11,
            "0001": None,
            "0010": None,
            "1101": None,
            "1110": None,
        }

        # Add the first 4 bits corresponding to the fixed first turn (0000)
        bitstring = bitstring + "0000"
        # Add the two qubits at positions 6 & 7 from the right (00)
        bitstring = bitstring[:-6] + "00" + bitstring[-6:]
        bitstring = bitstring[::-1]
        length_turns = len(bitstring) // 4
        turns = [encoding[bitstring[4 * i : 4 * (i + 1)]] for i in range(length_turns)]
        return turns[::-1]
    else:
        raise ValueError("Invalid encoding.")


# BCC
def bcc_turns_to_bitstring(turns: list, encoding: str) -> list[int]:
    """
    Converts a list of integers representing the turns on a BCC lattice
    to a bitstring.

    Input:
        turns (list): A list of integers representing the turns.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    # if turns is a list then eval it to get the list of integers
    try:
        turns = [int(turn) for turn in turns]
    except:
        turns = eval(turns)
    turns = turns[::-1]
    if encoding == "unary":
        raise ValueError("Not implemented.")
    elif encoding == "binary":
        encoding_list = {
            0: "000",
            1: "100",
            2: "001",
            3: "110",
            4: "111",
            5: "101",
            6: "011",
            7: "010",
        }
        bitstring = "".join([encoding_list[int(turn)] for turn in turns])
        bitstring = bitstring[:5] + bitstring[6:]
        bitstring = bitstring[3:]
        bitstring = bitstring[::-1]
    else:
        raise ValueError("Invalid encoding.")
    return bitstring


def bcc_bitstring_to_turns(bitstring: str, encoding: str) -> list[int]:
    """
    Converts a bitstring to a list of integers representing the turns on a
    BCC lattice based on the following encoding:
    0: (1, 1, -1)   1: (1, -1, -1)   2: (-1, 1, -1)   3: (-1, -1, -1)
    4: (1, 1, 1)   5: (1, -1, 1)   6: (-1, 1, 1)   7: (-1, -1, 1)

    Input:
        bitstring (str): The bitstring to be converted.
        encoding (str): The encoding of the bitstring. Unary or binary.

    Returns:
        list: A list of integers representing the turns.
    """
    if encoding == "unary":
        raise ValueError("Not implemented.")
    elif encoding == "binary":
        encoding = {
            "000": 0,
            "100": 1,
            "001": 2,
            "110": 3,
            "111": 4,
            "101": 5,
            "011": 6,
            "010": 7,
        }

        # Add the first 3 bits corresponding to the fixed first turn (000)
        bitstring = bitstring + "000"
        # Add one qubit at position 5 from the right (0)
        bitstring = bitstring[:-5] + "0" + bitstring[-5:]
        bitstring = bitstring[::-1]
        length_turns = len(bitstring) // 3
        turns = [encoding[bitstring[3 * i : 3 * (i + 1)]] for i in range(length_turns)]
        return turns[::-1]
    else:
        raise ValueError("Invalid encoding.")


def read_xyz(file_path: str, lattice: str) -> tuple[list[int], str]:
    """
    Reads an .xyz file and returns the turns as integers and the amino acid sequence.

    Input:
        file_path (str): Path to the .xyz file.
        lattice (str): The lattice structure ("fcc" or "tetrahedral").

    Returns:
        tuple[list[int], str]: A tuple containing a list of integers
        representing the turns and the amino acid sequence as a string.
    """
    # Read the .xyz file and extract coordinates and amino acids
    with open(file_path, "r") as file:
        lines = file.readlines()

    coords = []
    amino_acids = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:  # Ensure the line contains valid data
            amino_acids.append(parts[0])  # Extract the amino acid
            x, y, z = map(float, parts[1:])
            coords.append([x, y, z])

    coords = np.array(coords)
    sequence = "".join(amino_acids)  # Combine amino acids into a sequence string

    # Define the lattice-specific directions
    if lattice == "fcc":
        coordinates = (
            3.8
            * (1.0 / np.sqrt(2))
            * np.array(
                [
                    [1, 1, 0],
                    [-1, -1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [0, 1, 1],
                    [0, -1, -1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [1, 0, 1],
                    [-1, 0, -1],
                    [1, 0, -1],
                    [-1, 0, 1],
                ]
            )
        )
    elif lattice == "tetrahedral":
        coordinates = (
            3.8
            * (1.0 / np.sqrt(3))
            * np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]])
        )
    else:
        raise ValueError("Invalid lattice structure. Choose 'fcc' or 'tetrahedral'.")

    # Calculate the relative vectors (differences between consecutive coordinates)
    relative_vectors = np.diff(coords, axis=0)

    # Initialize the turn sequence
    turn_seq = []

    # Match each relative vector to the closest lattice vector
    for i, vector in enumerate(relative_vectors):
        # Adjust the vector based on the alternating (-1)^i factor
        # Ruihao: This is not needed for the FCC computations, right?
        adjusted_vector = (-1) ** i * vector

        # Find the closest lattice vector using Euclidean distance
        distances = np.linalg.norm(coordinates - adjusted_vector, axis=1)
        turn_index = np.argmin(distances)

        # Append the index of the closest lattice vector to the turn sequence
        turn_seq.append(turn_index)

    return turn_seq, sequence


def cubic_bitstring_to_turns():
    pass


# conf_bitstring = "01000110010"
# lattice = "bcc"
# encoding = "binary"
# print(bitstring_to_coords(conf_bitstring, lattice, encoding))

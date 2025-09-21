"""
The Folding problem
"""

import sys

sys.path.append("../")
import numpy as np
from scipy.spatial.distance import pdist, squareform

from turn_decoder import bitstring_to_coords
import os


def get_energy_matrix(
    main_seq: str,
    energy_file_name: str = "mj_matrix_1996",
    pair_energy_multiplier: float = 0.1,
) -> np.ndarray:
    """
    Get the energy matrix corresponding to the given main sequence from the energy matrix file.

    Args:
        main_seq (str): The main sequence of the protein.
        energy_file_name (str): The name of the energy matrix file.
        pair_energy_multiplier (float): The pair energy multiplier.

    Returns:
        np.ndarray: The energy matrix containing the interaction energies between the amino acids in the given sequence.
    """
    int_energy_matrix, list_aa = load_energy_matrix_file(energy_file_name)
    pair_energies = np.zeros((len(main_seq), len(main_seq)))
    for i in range(len(main_seq)):
        for j in range(i + 1, len(main_seq)):
            aa_i = list_aa.index(main_seq[i])
            aa_j = list_aa.index(main_seq[j])
            pair_energies[i, j] = int_energy_matrix[min(aa_i, aa_j), max(aa_i, aa_j)]
    pair_energies *= pair_energy_multiplier
    return pair_energies

def calculate_bitstring_energy_folding(
    conf_bitstring: str,
    lattice: str,
    encoding: str,
    penalty_param: float,
    pair_energies: np.ndarray,
    nearest_neighbors: int = 1,
    verbose: bool = False,
) -> float:
    """
    Calculate the energy of a protein shape given the bitstring indicating the conformation or turn sequence.

    Args:
        conf_bitstring (str): The bitstring indicating the conformation or turn sequence.
        lattice (str): The lattice structure of the protein ("tetrahedral",
        "fcc", or "bcc").
        encoding (str): The encoding of the bitstring.
        penalty_param (float): The penalty parameter.
        pair_energies (np.ndarray): The pairwise energy matrix of the protein.
        nearest_neighbors (int): The number of nearest neighbors to consider.

    Returns:
        float: The energy of the protein conformation.
    """
    # Possible nearest neighbor distances for different lattices
    if lattice == "tetrahedral":
        nn_distances = [3.8, 6.20537402, 7.27644602, 8.77572409, 9.56312362]
    elif lattice == "fcc":
        nn_distances = [3.8, 5.37401154, 6.58179307, 7.6, 8.49705831]
    elif lattice == "bcc":
        nn_distances = [3.8, 4.38786205, 6.20537402, 7.27644602, 7.6]
    else:
        raise ValueError("That lattice is not supported yet.")
    xyz_data = bitstring_to_coords(conf_bitstring, lattice, encoding)
    distances = pdist(xyz_data, metric="euclidean")  # Calculate pairwise distances
    n_overlaps = np.isclose(distances, 0).sum()
    overlap_penalty_energy = penalty_param * n_overlaps

    distance_matrix = squareform(distances)  # Convert pairwise distances to a matrix

    rows, cols = np.indices(distance_matrix.shape)
    if lattice == "fcc" or lattice == "bcc":
        # Only need to ignore any pair of row and column indices that are 1 AA
        # apart for any n-NN neighbors because they are always at distance r1 =
        # 3.8 Å
        mask = np.abs(rows - cols) == 1
        distance_matrix[mask] = np.nan
    elif lattice == "tetrahedral":
        # Need to ignore any pair of row and column indices that are 1 AA and 2
        # AAs apart when considering n-NN neighbors; e.g., pairs of 2 AAs apart
        # always at distance r2 = 6.20537402 Å or 0 (penalized above); pairs of
        # 3 AAs apart are not always at distance r3 = 7.27644602 Å
        mask = np.abs(rows - cols) == 1
        mask |= np.abs(rows - cols) == 2
        distance_matrix[mask] = np.nan

    interaction_energy = 0.0
    for nn in range(1, nearest_neighbors + 1):
        # Check for unit distance (3.8 Å) for first nearest neighbors
        if verbose:
            print("nn = ", nn)
            nn_pairs = ~np.isnan(distance_matrix) & np.isclose(
                nn_distances[nn - 1], distance_matrix
            )
            # Get the indices of the pairs that are first nearest neighbors
            nn_pairs_indices = np.argwhere(nn_pairs)
            # Filter out pairs that are not in the upper triangle of the matrix
            nn_pairs_indices = nn_pairs_indices[
                nn_pairs_indices[:, 0] < nn_pairs_indices[:, 1]
            ]
            print(f"{nn}-NN pairs: {nn_pairs_indices}")
            print(f"Number of {nn}-NN pairs: {len(nn_pairs_indices)}")

        nn_int_energy = pair_energies[
            ~np.isnan(distance_matrix)
            & np.isclose(nn_distances[nn - 1], distance_matrix)
        ].sum() * (
            3.8 / nn_distances[nn - 1]
        )  # 1/distance scaling
        if verbose:
            print(f"{nn}-NN interaction energy: {nn_int_energy}")
        interaction_energy += nn_int_energy
    if verbose:
        print("Total interaction energy: ", interaction_energy)
        print("Total overlapping penalty: ", overlap_penalty_energy)
    return round(overlap_penalty_energy + interaction_energy, 6)


def load_energy_matrix_file(file_name):
    """Returns the energy matrix from the Miyazawa-Jernigan potential file."""

    path = _construct_resource_path(file_name)
    matrix = np.loadtxt(fname=path, dtype=str)
    energy_matrix = _parse_energy_matrix(matrix)
    symbols = list(matrix[0, :])
    return energy_matrix, symbols


def _construct_resource_path(file_name):
    path = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            file_name + ".txt",
        )
    )
    return os.path.normpath(path)


def _parse_energy_matrix(matrix):
    """Parses a matrix loaded from the Miyazawa-Jernigan potential file."""
    energy_matrix = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
    for row in range(1, np.shape(matrix)[0]):
        for col in range(row - 1, np.shape(matrix)[1]):
            energy_matrix[row, col] = float(matrix[row, col])
    energy_matrix = energy_matrix[1:,]
    return energy_matrix
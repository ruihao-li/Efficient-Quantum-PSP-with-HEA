import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.circuit.library import RealAmplitudes
from train_qc import ProteinSolver, process_counts
from energy_functions import calculate_bitstring_energy_folding, get_energy_matrix
from dataset_class import load_dataset
from utils import *
from turn_decoder import *

import os
import glob
import json
import pickle
import psutil
import ray

# Path to save the results
current_file_directory = os.path.dirname(os.path.abspath(__file__))


def ratio_of_approximation(energy, ground_energy):
    """
    Calculate the ratio of approximation.
    """
    if ground_energy == 0:
        return float("inf")  # Avoid division by zero
    return energy / ground_energy

def relative_error(energy, ground_energy, sign_figures = 6):
    """
    Calculate the difference to the ground state.
    """
    if ground_energy == 0:
        return float("inf")  # Avoid division by zero
    return round(abs(energy - ground_energy) / abs(ground_energy), sign_figures)

def average_cost(prob_energy_pairs: list[tuple[float, float]]) -> float:
    return sum(prob * energy for prob, energy in prob_energy_pairs)

def cvar(
    prob_energy_pairs: list[tuple[float, float]], cvar_alpha: float = 0.1
) -> float:
    """
    Calculate the conditional value at risk (CVaR) energy of a protein shape.

    Args:
        prob_energy_pairs (list[tuple[float, float]]): List of 2-tuples. The
        first element of the tuple is the probability of a state. The seconds
        element of the tuple is the energy of that corresponding state.
        cvar_alpha (float): The alpha value for CVaR.

    Returns:
        float: The CVaR energy of the protein shape.
    """
    prob_energy_pairs = sorted(prob_energy_pairs, key=lambda x: x[1])
    accumulated_percent = 0.0
    cvar_energy = 0
    for prob, energy in prob_energy_pairs:
        cvar_energy += energy * min(prob, cvar_alpha - accumulated_percent)
        accumulated_percent += prob
        if accumulated_percent >= cvar_alpha:
            break
    return cvar_energy / cvar_alpha



def get_pdb_names(lattice = None):
    pdb_names = {
        "APRLRFY": "ABCP",
        "GCVLYPWC": "2M6C",
        "DRVYIHPFHL": "1N9U",
        "YYDPETGTWY": "5AWL",
        "VRRFDLLKRILK": "2N5R",
        "IFGAIAGFIKNIW": "2L24",
        "RGKWTYNGITYEGR": "1K43",
        "RHYYKFNSTGRHYHYY": "8T61",
        "GNLVS": "4QXX",
        "SNQNNF": "2OL9",
        "DLDALLADLE": "2K2R",
        "QYQFWKNFQT": "2MZX",
        "FATMRYPSDSDE": "1IXU",
        "INWLKLGKKIIASL": "6Q08",
        "WHMWNTVPNAKQVIAA": "8T63",
        "GGLRSLGRKILRAWKKYG": "2NDC",
        "IGLRGLGRKIALIHKKYG": "2NDE",
        "DAYAQWLKDGGPSSGRPPPS": "2JOF",
        "KKPGASLAALQALQALQAAQAAKKY": "8B1X",
        "YYHFWHRGVTKRSLSPHRPRHSRLQR": "6A8Y"
    }

    bcc_pdb_names = {
        "APRLRFY": "ABCP",
        "GCVLYPWC": "2M6C",
        "DRVYIHPFHL": "1N9U",
        "YYDPETGTWY": "5AWL",
        "VRRFDLLKRILK": "2N5R",
        "IFGAIAGFIKNIW": "2L24",
        "RGKWTYNGITYEGR": "1K43",
        "RHYYKFNSTGRHYHYY": "8T61"
        }
    fcc_pdb_names = {
        "GNLVS": "4QXX",
        "SNQNNF": "2OL9",
        "GCVLYPWC": "2M6C",
        "DLDALLADLE": "2K2R",
        "QYQFWKNFQT": "2MZX",
        "YYDPETGTWY": "5AWL",
        "FATMRYPSDSDE": "1IXU",
        "VRRFDLLKRILK": "2N5R",
        "IFGAIAGFIKNIW": "2L24",
        "RGKWTYNGITYEGR": "1K43"
        }
    tetrahedral_pdb_names = {
        "DLDALLADLE": "2K2R",
        "QYQFWKNFQT": "2MZX",
        "YYDPETGTWY": "5AWL",
        "FATMRYPSDSDE": "1IXU",
        "VRRFDLLKRILK": "2N5R",
        "IFGAIAGFIKNIW": "2L24",
        "INWLKLGKKIIASL": "6Q08",
        "RGKWTYNGITYEGR": "1K43",
        "RHYYKFNSTGRHYHYY": "8T61",
        "WHMWNTVPNAKQVIAA": "8T63",
        "GGLRSLGRKILRAWKKYG": "2NDC",
        "IGLRGLGRKIALIHKKYG": "2NDE",
        "DAYAQWLKDGGPSSGRPPPS": "2JOF",
        "KKPGASLAALQALQALQAAQAAKKY": "8B1X",
        "YYHFWHRGVTKRSLSPHRPRHSRLQR": "6A8Y"
        }
    if lattice == None:
        return pdb_names
    elif lattice == "bcc":
        return bcc_pdb_names
    elif lattice == "fcc":
        return fcc_pdb_names
    elif lattice == "tetrahedral":
        return tetrahedral_pdb_names


def set_plot_constants(font_size = 14):
    colors = (
        plt.rcParams["axes.prop_cycle"].by_key()["color"][:6] + plt.rcParams["axes.prop_cycle"].by_key()["color"][8:]
        + plt.rcParams["axes.prop_cycle"].by_key()["color"][:6] + plt.rcParams["axes.prop_cycle"].by_key()["color"][8:]
    )
    marker_list = ["o", "s", "D", "v", "^", ">", "<", "p", "h", "H", "d", "P", "X", "*", "1", "2", "3", "4", "8", "x", "+", "|", "_", "o", "s", "D", "v", "^", ">", "<", "p", "h", "H", "d", "P", "X", "*", "1", "2", "3", "4", "8", "x", "+", "|", "_"]

    marker_size = font_size - 3
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["figure.titlesize"] = font_size
    return colors, marker_list, marker_size


#for dir_i, directory_name in enumerate(directory_names):
def get_whole_dataset(lattice, nearest_neighbors, num_layers, encoding = "binary", matrix_year="1996"):
    directory_name = f"{lattice}_same_num_layers_{num_layers}_{100000}shots_cvar_1996_nn_{nearest_neighbors}_RealAmplitudes"
    load_directory = current_file_directory + "/results/" + directory_name + "/"
    all_files = get_filenames_os(load_directory)
    # load the constants from the file
    # some directories have a sub directory named a random number and some don't
    variables = load_constants_from_file(load_directory + "constants_dict.txt")
    locals().update(variables)
    print("Variables loaded: ", variables, "\n")

    # load the dataset
    whole_dataset = load_dataset(lattice, encoding, nearest_neighbors, str(matrix_year))
    # print(whole_dataset)
    # get the aa_list
    aa_list = []
    for protein_i, protein_name in enumerate(whole_dataset.copy().keys()):
        try:
            protein_info = whole_dataset[protein_name]
        except:
            protein_info = whole_dataset[
                str(len(protein_name))
                + "AA_"
                + protein_name
                + "_"
                + str(nearest_neighbors)
                + "NN_"
                + lattice
            ]
        # Verify that the protein is in the results directory
        if not any(protein_info.sequence in file_name for file_name in all_files):
            print(
                f"Warning: Protein {protein_info.sequence} not found in results directory {load_directory}"
            )
            whole_dataset.pop(protein_name)
        else:
            aa_list.append(protein_info.number_of_aa)

    num_proteins = len(whole_dataset)

    save_file_path = current_file_directory + "/plots/" + lattice + "/"
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    save_file_at = (
        save_file_path + str(num_layers) + "_layers_" + str(nearest_neighbors) + "nn"
    )
    return whole_dataset, aa_list, save_file_at, load_directory, variables

def get_sequence(protein_name, whole_dataset, lattice, nearest_neighbors):
    try:
        protein_info = whole_dataset[protein_name]
    except:
        protein_info = whole_dataset[
            str(len(protein_name))
            + "AA_"
            + protein_name
            + "_"
            + str(nearest_neighbors)
            + "NN_"
            + lattice
        ]
    sequence = protein_info.sequence
    return sequence

def load_protein_info(protein_name, whole_dataset, lattice, nearest_neighbors):
    try:
        protein_info = whole_dataset[protein_name]
    except:
        protein_info = whole_dataset[
            str(len(protein_name))
            + "AA_"
            + protein_name
            + "_"
            + str(nearest_neighbors)
            + "NN_"
            + lattice
        ]
    return protein_info

def get_protein_info(whole_dataset, protein_name, nearest_neighbors, lattice, num_layers, top_n_folds=50, encoding="binary"):
    directory_name = f"{lattice}_same_num_layers_{num_layers}_{100000}shots_cvar_1996_nn_{nearest_neighbors}_RealAmplitudes"
    load_directory = current_file_directory + "/results/" + directory_name + "/"
    protein_info = load_protein_info(protein_name, whole_dataset, lattice, nearest_neighbors)
    sequence = protein_info.sequence
    ground_energy = protein_info.ground_energy
    ground_state_bitstring = protein_info.ground_state_bitstring
    num_aa = protein_info.number_of_aa

    # look at best bitstring
    # compare if the best bitstring list with N best folds from the dataset
    folds_list = list(protein_info.folding_dict.keys())[:top_n_folds]
    # translate the folds to bitstrings
    classic_best_bitstrings = []
    for fold in folds_list:
        if lattice == "tetrahedral":
            turns = eval(fold)
            bitstring = tetrahedral_turns_to_bitstring(turns, encoding)
        elif lattice == "fcc":
            turns = eval(fold)
            bitstring = fcc_turns_to_bitstring(turns, encoding)
        elif lattice == "bcc":
            turns = eval(fold)
            bitstring = bcc_turns_to_bitstring(turns, encoding)
        classic_best_bitstrings.append(bitstring)

    # find how many average iterations were performed
    #av_files = glob.glob(load_directory + sequence + "*best_bitstrings*")
    average_iter = 10
    return protein_info, sequence, ground_energy, ground_state_bitstring, num_aa, average_iter

def get_number_of_qubits(lattice, sequence):
    if sequence is not int:
        num_aa = len(sequence)
    if lattice == "tetrahedral":
        num_qubits = 2 * (num_aa - 1) - 5
    elif lattice == "fcc":
        num_qubits = 4 * num_aa - 10
    elif lattice == "bcc":
        num_qubits = 3 * num_aa - 7
    else:
        raise ValueError(f"Unsupported lattice type: {lattice}")
    return num_qubits

def load_best_average_cost(average_iter, load_directory, sequence):
    lowest_average_cost_list = []
    for av_i in range(average_iter):
        try:
            cost_history_dict = pickle.load(
                open(
                    load_directory + sequence + f"_av{av_i}__cost_history_dict.pkl",
                    "rb",
                )
            )
        except:
            cost_history_dict = pickle.load(
                open(
                    load_directory + sequence + f"_cost_history_dict_av{av_i}.pkl",
                    "rb",
                )
            )
        try:
            # "iters": 0,            "prev_vector": [],            "cost_history": [],            "job_id": [],
            last_params = cost_history_dict["prev_vector"][-1]
            cost_history = cost_history_dict["cost_history"]
        except:
            # cost_trajectory and best_params
            cost_history = cost_history_dict["cost_trajectory"]
            last_params = cost_history_dict["best_params"]
        lowest_average_cost_list.append(cost_history[-1])

    # find which average iteration has the lowest cost
    lowest_average_cost = min(lowest_average_cost_list)
    best_average_i = lowest_average_cost_list.index(lowest_average_cost)

    return best_average_i, lowest_average_cost_list, lowest_average_cost

def load_best_bitstrings(average_iter, load_directory, sequence):
    best_bitstring_list = []
    for av_i in range(average_iter):
        # load the best bitstrings of the best average iteration
        with open(
            load_directory + sequence + f"_best_bitstrings_av{av_i}.json", "r"
        ) as f:
            best_bitstrings = json.load(f)
        best_bitstring = float(best_bitstrings[0][1])
        best_bitstring_list.append(best_bitstring)
    best_bitstring = min(best_bitstring_list)
    best_bitstring_i = best_bitstring_list.index(best_bitstring)
    return best_bitstring_i, best_bitstring_list, best_bitstring

def load_best_cost_history(best_average_i, load_directory, sequence):
    # load the best bitstrings of the best average iteration
    try:
        cost_history_dict = pickle.load(
            open(
                load_directory
                + sequence
                + f"_av{best_average_i}__cost_history_dict.pkl",
                "rb",
            )
        )
    except:
        cost_history_dict = pickle.load(
            open(
                load_directory
                + sequence
                + f"_cost_history_dict_av{best_average_i}.pkl",
                "rb",
            )
        )
    try:
        # "iters": 0,            "prev_vector": [],            "cost_history": [],            "job_id": [],
        last_params = cost_history_dict["prev_vector"][-1]
        cost_history = cost_history_dict["cost_history"]
    except:
        # cost_trajectory and best_params
        cost_history = cost_history_dict["cost_trajectory"]
        last_params = cost_history_dict["best_params"]
    return last_params, cost_history


def calculate_roa_mean_sterror(lowest_average_cost_list, ground_energy, average_iter):
    # Calculate the ratio of approximation
    ratio_of_approximation_list = []
    for average_cost in lowest_average_cost_list:
        ratio_of_approximation_list.append(
            ratio_of_approximation(average_cost, ground_energy)
        )

    approx_ratio_mean = np.mean(ratio_of_approximation_list)
    approx_ratio_sterror = np.std(ratio_of_approximation_list) / np.sqrt(average_iter)

    return approx_ratio_mean, approx_ratio_sterror

def calculate_RE_mean_sterror(lowest_bitstring, ground_energy, average_iter):
    # Calculate the relative error
    diff_list = []
    for energy in lowest_bitstring:
        diff_list.append(
            relative_error(energy, ground_energy)
        )

    diff_mean = np.mean(diff_list)
    diff_sterror = np.std(diff_list) / np.sqrt(average_iter)

    return diff_mean, diff_sterror

#------------------

def plot_cost_history(protein_i, cost_history, prob_energy_pairs, sequence, ground_energy, save_file_at, whole_dataset):
    # -------------------------- plot the cost history
    colors2 = plt.cm.coolwarm(np.linspace(0, 1, num_proteins + 1))
    plt.figure(20 + nearest_neighbors + dir_i)
    plt.plot(
        cost_history,prob_energy_pairs,
        label=sequence + ", " + str(round(ground_energy, 3)),
        color=colors2[protein_i],
    )
    plt.plot(
        [0, len(cost_history)],
        [ground_energy, ground_energy],
        color=colors2[protein_i],
        linestyle="--",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    if protein_i == len(whole_dataset) - 1:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.savefig(
            save_file_at + "_cost_history_plot.pdf", bbox_inches="tight"
        )
        plt.close()

def plot_diff_to_best_bitstrings(
    protein_i,
    diff_to_best_bitstrings_mean,
    diff_to_best_bitstrings_std,
    sequence,
    lowest_cost,
    ground_energy,
    save_file_at,
    colors,
    marker_list,
    marker_size,
    whole_dataset,
    aa_list
    ):
    # -------------------------- plot diff to best bitstrings
    plt.figure(30 + nearest_neighbors + dir_i)
    plt.errorbar(
        protein_i,
        diff_to_best_bitstrings_mean,
        yerr=diff_to_best_bitstrings_std,
        label=sequence,
        color=colors[protein_i],
        marker=marker_list[protein_i],
        capsize=5,
        markersize=marker_size,
    )
    # add a point with the lowest_cost-ground_energy
    plt.scatter(
        protein_i,
        round(abs(lowest_cost - ground_energy) / abs(ground_energy), 6),
        color=colors[protein_i],
        marker="o",
        s=50,
    )
    plt.xlabel("Number of amino acids")
    plt.ylabel("Difference to ground energy")
    plt.xticks(range(len(whole_dataset)), aa_list)
    plt.yticks([0.0, 0.25, 0.5])
    plt.ylim((0.0, 0.5))
    if protein_i == len(whole_dataset) - 1:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.savefig(
            save_file_at + "_diff_to_best_bitstrings_plot.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_overlap(protein_i, overlap_mean, overlap_std, sequence, best_overlap, save_file_at, colors, marker_list, marker_size, whole_dataset, aa_list):
    # -------------------------- plot the overlap
    plt.figure(50 + nearest_neighbors + dir_i)
    plt.errorbar(
        protein_i,
        overlap_mean,
        yerr=overlap_std,
        label=sequence,
        color=colors[protein_i],
        marker=marker_list[protein_i],
        capsize=5,
        markersize=marker_size,
    )
    plt.plot(
        protein_i,
        best_overlap,
        "o",
        color=colors[protein_i],
        markersize=marker_size - 2,
    )
    # Xticks is the pbd name, the number of aa in aa_list
    plt.xticks(range(len(whole_dataset)), [(pdb_names[seq], num_aa) for seq, num_aa in zip(aa_list, [len(seq) for seq in aa_list])])
    plt.xlabel("Number of amino acids")
    plt.ylabel("Overlap with best folds")
    plt.ylim((0.0, 1.0))
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if protein_i == len(whole_dataset) - 1:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.savefig(save_file_at + "_overlap_plot.pdf", bbox_inches="tight")
        plt.close()

def calculate_overlap(average_iter, load_directory, sequence, classic_best_bitstrings, top_n_folds):
    overlap_array = []
    for av_i in range(average_iter):
        # calculate the overlap with the best bitstrings
        with open(
            load_directory + sequence + f"_best_bitstrings_av{av_i}.json", "r"
        ) as f:
            best_bitstrings = json.load(f)
        best_bitstrings_list = [x[0] for x in best_bitstrings][:top_n_folds]
        overlap = 0
        for bitstring in best_bitstrings_list:
            if bitstring in classic_best_bitstrings:
                overlap += 1
        overlap = overlap / len(best_bitstrings_list)
        overlap_array.append(overlap)
    
    # calculate the mean and sterror of the overlap
    overlap_mean = np.mean(overlap_array)
    overlap_sterror = np.std(overlap_array) / np.sqrt(average_iter)

    # check the overlap between the best bitstrings and the best folds
    best_bitstrings_list = [x[0] for x in best_bitstrings][:top_n_folds]
    best_overlap = 0
    for bitstring in best_bitstrings_list:
        if bitstring in classic_best_bitstrings:
            best_overlap += 1
    best_overlap = best_overlap / len(best_bitstrings_list)
    return best_overlap, overlap_mean, overlap_sterror


def check_if_results_exist(load_directory, sequence):
    try:
        with open(load_directory + sequence + "_hitrates.json", "r") as f:
            hitrate_array = json.load(f)
    except:
        calculate_hitrate = True

    try:
        with open(load_directory + sequence + "_prob_energy_pairs.json", "r") as f:
            prob_energy_pairs = json.load(f)
    except:
        calculate_probs = True

    try:
        with open(load_directory + sequence + "_bitstrings_in_end.json", "r") as f:
            bitstrings_in_end = json.load(f)
        calculate_bitstrings_in_end = False
    except:
        calculate_bitstrings_in_end = True
    return calculate_hitrate, calculate_probs, calculate_bitstrings_in_end

def get_protein_solver(
    num_qubits,
    num_layers,
    lattice,
    sequence,
    nearest_neighbors,
    encoding = "binary",
    penalty_param = 10.0,
    pair_energy_multiplier = 0.1,
    energy_file_name = "mj_matrix_1996",
    default_shots=100_000,
    energy_function_name = "calculate_bitstring_energy_folding",
    cost_function_name = "cvar",
    ):
    ansatz = RealAmplitudes(num_qubits, reps=num_layers).decompose()

    mps = num_qubits >= 30  # Use matrix product state for larger circuits
    if mps:
        sampler = Sampler(
            default_shots=default_shots,
            options={"backend_options": {"method": "matrix_product_state"}},
        )
    else:
        sampler = Sampler(default_shots=default_shots)
    print(f"Sampler set up with {'MPS' if mps else 'SV'} method.")

    pair_energies = get_energy_matrix(
        sequence, energy_file_name, pair_energy_multiplier
    )
    energy_args = (
        lattice,
        encoding,
        penalty_param,
        pair_energies,
        nearest_neighbors,
    )
    energy_args = dict(
        zip(
            [
                "lattice",
                "encoding",
                "penalty_param",
                "pair_energies",
                "nearest_neighbors",
            ],
            energy_args,
        )
    )
    cost_args = (0.1,)
    cost_function = eval(cost_function_name)
    energy_function = eval(energy_function_name)

    protein_solver = ProteinSolver(
        ansatz,
        sampler,
        energy_function,
        energy_args,
        cost_function,
        cost_args,
        save_file_at="",
    )
    return protein_solver

def params_to_probs(protein_solver, last_params, load_directory, sequence):
    NUM_WORKERS = psutil.cpu_count()  # Use all available CPUs
    prob_energy_pairs = protein_solver.get_prob_energy_pairs(
        params=last_params,
        num_batches=NUM_WORKERS,
    )
    with open(
        load_directory + sequence + "_prob_energy_pairs.json", "w"
    ) as f:
        json.dump(prob_energy_pairs, f)
    return prob_energy_pairs

def counts_to_probs(protein_solver, counts):
    NUM_WORKERS = psutil.cpu_count()  # Use all available CPUs
    _, prob_energy_pairs = process_counts(
        counts,
        protein_solver.energy_function,
        protein_solver.energy_args,
        num_batches=NUM_WORKERS,
        global_bitstring_energies=protein_solver.global_bitstring_energies,
    )
    return prob_energy_pairs

def calculate_hitrate(average_iter, load_directory, sequence, protein_solver, ground_energy, default_shots):
    hitrate_array = np.zeros(average_iter)
    for av_i in range(average_iter):
        try:
            _cost_history_dict = pickle.load(
                open(
                    load_directory
                    + sequence
                    + f"_av{av_i}__cost_history_dict.pkl",
                    "rb",
                )
            )
        except:
            _cost_history_dict = pickle.load(
                open(
                    load_directory
                    + sequence
                    + f"_cost_history_dict_av{av_i}.pkl",
                    "rb",
                )
            )
        try:
            _last_params = _cost_history_dict["prev_vector"][-1]
        except:
            _last_params = _cost_history_dict["best_params"]

        hitrate = protein_solver.get_hitrate(
            params=_last_params,
            shots=default_shots,
            ground_energy=ground_energy,
        )
        hitrate_array[av_i] = hitrate

    with open(load_directory + sequence + "_hitrates.json", "w") as f:
        json.dump(hitrate_array.tolist(), f)

def calculate_depth(lattice, num_layers, nearest_neighbors, whole_dataset):
    depth_vector = []
    depth_vector_1qb = []
    depth_vector_2qb = []
    num_params_vector = []
    directory_name = f"{lattice}_same_num_layers_{num_layers}_{100000}shots_cvar_1996_nn_{nearest_neighbors}_RealAmplitudes"
    load_directory = current_file_directory + "/results/" + directory_name + "/"
    for protein_i, protein_name in enumerate(whole_dataset):
        num_qubits = get_number_of_qubits(lattice, get_sequence(protein_name, whole_dataset, lattice, nearest_neighbors))
        ansatz = RealAmplitudes(num_qubits, reps=num_layers).decompose()
        depth_vector.append(ansatz.depth())
        depth_vector_1qb.append(count_gates(ansatz)[1])
        depth_vector_2qb.append(count_gates(ansatz)[2])
        num_params_vector.append(ansatz.num_parameters)
        with open(load_directory + "_depth_vector.json", "w") as f:
            json.dump(depth_vector, f)
        with open(load_directory + "_depth_vector_1qb.json", "w") as f:
            json.dump(depth_vector_1qb, f)
        with open(load_directory + "_depth_vector_2qb.json", "w") as f:
            json.dump(depth_vector_2qb, f)
        with open(load_directory + "_num_params.json", "w") as f:
            json.dump(num_params_vector, f)
    return depth_vector, depth_vector_1qb, depth_vector_2qb, num_params_vector

def calculate_bitstrings_in_end(protein_solver, last_params, load_directory, sequence, default_shots, top_n_folds=50):
    bitstrings_in_end = protein_solver.get_top_bitstrings(
        params=last_params, shots=default_shots, top_n=top_n_folds
    )
    with open(
        load_directory + sequence + "_bitstrings_in_end.json", "w"
    ) as f:
        json.dump(bitstrings_in_end, f)

def plot_overlap_in_the_end():
    # calculate overlap with the ground state bitstring for after the run
    overlap_in_the_end = 0
    for bitstring in bitstrings_in_end:
        if bitstring in classic_best_bitstrings:
            overlap_in_the_end += 1
    overlap_in_the_end = overlap_in_the_end / len(bitstrings_in_end)
    if plot_overlap_in_the_end:
        plt.figure(70 + nearest_neighbors + dir_i)
        plt.plot(
            num_aa,
            overlap_in_the_end,
            "o",
            color=colors[protein_i],
            label=sequence,
            markersize=marker_size,
        )
        plt.xlabel("Number of amino acids")
        plt.ylabel("Overlap with best folds in the end")
        plt.xticks(aa_list)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylim((0.0, 1.0))
        if protein_i == len(whole_dataset) - 1:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.savefig(
                save_file_at + "_overlap_in_the_end_plot.pdf", bbox_inches="tight"
            )
            plt.close()

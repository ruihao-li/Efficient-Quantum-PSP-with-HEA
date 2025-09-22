import sys

sys.path.append("src/")
import numpy as np
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from train_qc import ProteinSolver
from utils import save_constants_to_file, save_text_to_file
from energy_functions import calculate_bitstring_energy_folding, get_energy_matrix
from cost_functions import cvar
from dataset_class import load_dataset
import ray
import psutil
import pickle
import json

NUM_WORKERS = psutil.cpu_count()  # Use all available CPUs
ray.init(
    num_cpus=NUM_WORKERS,
    ignore_reinit_error=True,
    log_to_driver=False,
    runtime_env={
        "py_modules": ["src"],
        "working_dir": ".",
        "env_vars": {"PYTHONPATH": "./src"},
    },
)
import time
import os

tick = time.time()

current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Constants
start_index = 0  # Index starting from 0
end_index = 1  # Controls how many proteins to run
maxiter = 50  # Max iterations for optimizer
average_iter = 5  # Number of times to run the optimization for each protein
verbose = True
num_layers = 1
lattice = "tetrahedral"  # "tetrahedral", "bcc", or "fcc"
default_shots = 100_000
encoding = "binary"
penalty_param = 10.0  # Penalty parameter for overlaps
energy_file_name = "mj_matrix_1996"
pair_energy_multiplier = 0.1  # Scaling factor for pair energies
energy_function_name = "calculate_bitstring_energy_folding"
energy_function = eval(energy_function_name)
nearest_neighbors = 1
cost_function_name = "cvar"
cost_function = eval(cost_function_name)
cost_args = (0.1,)
# ansatz_name = 'EfficientSU2'
ansatz_name = "RealAmplitudes"
donation_params = False

# Load the entire dataset in the directory
dataset = load_dataset(
    lattice=lattice,
    encoding=encoding,
    nearestneighbor=nearest_neighbors,
    matrix_year=matrix_year,
    directory="classical_search/res/",
)
resname_list = list(dataset.keys())
print(f"Dataset loaded with {len(dataset)} proteins.")
total_num_proteins = len(dataset)
if end_index >= total_num_proteins:
    print(
        f"End index {end_index} is not compatible with the dataset size {total_num_proteins}. Setting end index to {total_num_proteins - 1}."
    )
    end_index = total_num_proteins - 1

# Save the constants to txt
constants_to_save = {
    "start_index": start_index,
    "end_index": end_index,
    "total_num_proteins": total_num_proteins,
    "maxiter": maxiter,
    "average_iter": average_iter,
    "num_layers": num_layers,
    "lattice": lattice,
    "default_shots": default_shots,
    "encoding": encoding,
    "matrix_year": matrix_year,
    "penalty_param": penalty_param,
    "energy_file_name": energy_file_name,
    "pair_energy_multiplier": pair_energy_multiplier,
    "energy_function_name": energy_function_name,
    "nearest_neighbors": nearest_neighbors,
    "cost_function_name": cost_function_name,
    "cost_args": cost_args,
    "ansatz_name": ansatz_name,
    "dataset": dataset,
}

now_testing = (
    lattice
    + "_"
    + "same_num_layers_"
    + str(num_layers)
    + "_"
    + str(default_shots)
    + "shots_"
    + cost_function_name
    + "_"
    + matrix_year
    + "_nn_"
    + str(nearest_neighbors)
    + "_"
    + ansatz_name
)
save_file_at = current_file_directory + "/results/" + now_testing + "/"

# Ensure the directory exists
os.makedirs(os.path.dirname(save_file_at), exist_ok=True)

# Save constants to file
save_constants_to_file(save_file_at + "constants_dict.txt", constants_to_save)

exec(
    "from turn_decoder import "
    + lattice
    + "_turns_to_bitstring, "
    + lattice
    + "_bitstring_to_turns"
)

for data_i in range(start_index, end_index + 1):

    protein = dataset[resname_list[data_i]]
    sequence = protein.sequence
    ground_energy = protein.ground_energy
    ground_state = protein.ground_state_bitstring

    if lattice == "tetrahedral":
        num_qubits = 2 * (len(sequence) - 1) - 5
    elif lattice == "fcc":
        num_qubits = 4 * len(sequence) - 10
    elif lattice == "bcc":
        num_qubits = 3 * len(sequence) - 7
    else:
        raise ValueError(f"Unsupported lattice type: {lattice}")

    if verbose:
        print(f"Protein sequence: {sequence}")
        print(f"Number of AAs: {len(sequence)}")
        print(f"Number of qubits: {num_qubits}")
        print(f"Ground state energy: {ground_energy}")
        print(f"Ground state bitstring: {ground_state}")

    if ansatz_name == "RealAmplitudes":
        ansatz = RealAmplitudes(num_qubits, reps=num_layers).decompose()
    elif ansatz_name == "EfficientSU2":
        ansatz = EfficientSU2(num_qubits, reps=num_layers).decompose()

    save_text_to_file(
        save_file_at + sequence + "_ansatz_circuit.txt", str(ansatz.draw(output="text"))
    )

    # Set up sampler (sv if num_qubits < 30, mps otherwise)
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
    energy_args = (lattice, encoding, penalty_param, pair_energies, nearest_neighbors)
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

    # Run many times with random init_params, save the best one
    cost_lowest_array = np.zeros(average_iter)
    hitrate_array = np.zeros(average_iter)
    for average_i in range(average_iter):
        save_file_at_av = save_file_at + sequence + "_av" + str(average_i) + "_"
        if verbose:
            print(f"Running average iteration {average_i+1}/{average_iter}")

        # Train the quantum circuit

        if donation_params and data_i != start_index:
            # Use the parameters from the previous run
            protein_before = dataset[resname_list[data_i - 1]]
            sequence_before = protein_before.sequence
            temp_hitrate_array = np.genfromtxt(
                save_file_at
                + sequence_before
                + "_hitrate_av"
                + str(average_iter - 1)
                + ".out"
            )
            best_hitrate_i = np.argmax(temp_hitrate_array)
            if lattice == "tetrahedral":
                temp_qb = 2 * (len(sequence_before) - 1) - 5
            elif lattice == "fcc":
                temp_qb = 4 * len(sequence_before) - 10
            elif lattice == "bcc":
                temp_qb = 3 * len(sequence_before) - 7

            name_of_smaller = (
                save_file_at + sequence_before + "_av" + str(best_hitrate_i) + "_"
            )
            temp_cost_history_dict = pickle.load(
                open(name_of_smaller + "_cost_history_dict.pkl", "rb")
            )
            temp_params = temp_cost_history_dict["prev_vector"][-1]
            if len(sequence_before) == len(sequence):
                init_params = temp_params
            else:
                init_params = []
                cut = 0
                for num_layer_i in range(num_layers + 1):
                    for x in range(2):  # R and U
                        init_params.append(
                            np.concatenate(
                                (
                                    temp_params[temp_qb * cut : temp_qb * (cut + 1)],
                                    2 * np.pi * np.random.random(num_qubits - temp_qb),
                                ),
                                axis=0,
                            )
                        )
                        cut += 1
            init_params = np.reshape(init_params, (ansatz.num_parameters,))

        else:
            # Initialize the parameters randomly
            init_params = np.random.rand(ansatz.num_parameters)

        protein_solver = ProteinSolver(
            ansatz,
            sampler,
            energy_function,
            energy_args,
            cost_function,
            cost_args,
            save_file_at=save_file_at_av,
        )
        protein_solver.cost_trajectory = []
        parameters, _ = protein_solver.train(
            init_params,
            optimizer="COBYLA",
            maxiter=maxiter,
            num_batches=NUM_WORKERS,
            verbose=False,
        )

        # Calculate the energy of the 50 best bitstring
        sorted_bitstrings = sorted(
            protein_solver.global_bitstring_energies.items(), key=lambda x: x[1]
        )[:50]
        lowest_energy = sorted_bitstrings[0][1]
        best_bitstring = sorted_bitstrings[0][0]
        print(f"Lowest energy at iteration {average_i}: {lowest_energy}")
        print(f"Best bitstring at iteration {average_i}: {best_bitstring}")

        cost_lowest_array[average_i] = lowest_energy

        with open(
            save_file_at + sequence + "_best_bitstrings_av" + str(average_i) + ".json",
            "w",
        ) as f:
            json.dump(sorted_bitstrings, f)

        hitrate = protein_solver.get_hitrate(
            params=parameters, shots=default_shots, ground_energy=ground_energy
        )
        print(f"Hitrate at iteration {average_i}: {hitrate}")
        hitrate_array[average_i] = hitrate

    # Save the hitrate array and lowest cost array of multiple runs
    with open(save_file_at + sequence + "_lowest_costs.json", "w") as f:
        json.dump(cost_lowest_array.tolist(), f)
    with open(save_file_at + sequence + "_hitrates.json", "w") as f:
        json.dump(hitrate_array.tolist(), f)

    # Find the lowest cost
    lowest_energy_i = np.argmin(cost_lowest_array)
    print("cost_lowest_array ", cost_lowest_array)
    print("lowest_energy_i ", lowest_energy_i)
    lowest_energy = cost_lowest_array[lowest_energy_i]
    print("lowest_energy ", lowest_energy)
    print()


print("Done")

ray.shutdown()

tack = time.time()
duration = tack - tick

print(f" >> The whole calculation took {duration/60:.2f} minutes")
print("Files are here: ", save_file_at)

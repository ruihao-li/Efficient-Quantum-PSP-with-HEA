import numpy as np

from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Batch
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_ibm_runtime import SamplerV2 as Sampler
from dataset_class import load_dataset
from dataset_class import ProteinData
from utils import load_constants_from_file, get_filenames_os
from datetime import datetime, timezone
import os
import pickle
import json

# Path to save the results
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Set up Qiskit Runtime
service = QiskitRuntimeService(
    token="<token>",
    channel="ibm_quantum_platform",
    instance="<crn>",
)

TIMESTAMP = datetime.now(timezone.utc)
TEST = False

if TEST:
    lattice_list = ["tetrahedral"]
    nearest_neighbors_list = [1]
    num_layers = 1
    encoding = "binary"
    matrix_year = "1996"
    num_shots = 1000
    # Local testing mode
    backend = AerSimulator(method="matrix_product_state")

else:
    lattice_list = ["tetrahedral", "fcc", "bcc"]
    # lattice_list = ["tetrahedral"]
    nearest_neighbors_list = [1, 2]
    # nearest_neighbors_list = [1]
    num_layers = 1
    encoding = "binary"
    matrix_year = "1996"
    num_shots = 100_000
    backend = service.backend("ibm_kingston")  # Replace with any backend
    print(f"Backend: {backend.name}")


def aa_to_qubit(lattice, num_aa):
    """
    Convert amino acid count to qubit count based on the lattice type.
    """
    if lattice == "tetrahedral":
        return 2 * (num_aa - 1) - 5
    elif lattice == "fcc":
        return 4 * num_aa - 10
    elif lattice == "bcc":
        return 3 * num_aa - 7
    else:
        raise ValueError("Unknown lattice type")


# Get the results directories
directory_names = []
for lattice in lattice_list:
    for nn in nearest_neighbors_list:
        directory_name = f"{lattice}_same_num_layers_{num_layers}_{100000}shots_cvar_1996_nn_{nn}_RealAmplitudes"
        directory_names.append(directory_name)

print("Directory names: ", directory_names, "\n")

# Load proteins contained in each directory
full_dataset = {}
for dir_i, directory_name in enumerate(directory_names):
    print("Directory name: ", directory_name, " dir_i: ", dir_i, "\n")
    results_dir = current_file_directory + "/results/" + directory_name + "/"
    all_files = get_filenames_os(results_dir)
    variables = load_constants_from_file(results_dir + "constants_dict.txt")
    lattice = variables["lattice"]
    nearest_neighbors = variables["nearest_neighbors"]
    # load the dataset
    dataset = load_dataset(lattice, encoding, nearest_neighbors, matrix_year)
    # get the aa_list
    aa_list = []
    for protein_i, protein_name in enumerate(dataset.copy().keys()):
        try:
            protein_info = dataset[protein_name]
        except:
            protein_info = dataset[
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
                f"Warning: Protein {protein_info.sequence} not found in results directory {results_dir}"
            )
            dataset.pop(protein_name)
        else:
            aa_list.append(protein_info.number_of_aa)

    num_proteins = len(dataset)
    print("Number of proteins: ", num_proteins, "\n")
    full_dataset[directory_name] = dataset

print("Full dataset keys: ", full_dataset.keys(), "\n")

circuits = {}
# Create circuits for each protein in the dataset
for i, dir_name in enumerate(full_dataset.keys()):
    results_dir = current_file_directory + "/results/" + dir_name + "/"
    variables = load_constants_from_file(results_dir + "constants_dict.txt")
    lattice = variables["lattice"]
    nearest_neighbors = variables["nearest_neighbors"]
    average_iter = variables["average_iter"]
    print("Lattice: ", lattice, "\n")
    # Load the protein data
    print("Processing directory: ", dir_name, "\n")

    for protein_i, protein_name in enumerate(full_dataset[dir_name]):
        protein_info = full_dataset[dir_name][protein_name]
        sequence = protein_info.sequence
        print(
            f"Sequence: {sequence}, Protein index: {protein_i + 1} of {len(full_dataset[dir_name])}"
        )
        num_aa = protein_info.number_of_aa
        num_qubits = aa_to_qubit(
            lattice=lattice,
            num_aa=num_aa,
        )
        # Create the parameterized circuit
        qc = RealAmplitudes(
            num_qubits=num_qubits,
            reps=num_layers,
        ).decompose()
        qc.measure_all()

        # find the iteration with the lowest cost
        lowest_cost_array = []
        for av_i in range(average_iter):
            try:
                cost_history_dict = pickle.load(
                    open(
                        results_dir + sequence + f"_av{av_i}__cost_history_dict.pkl",
                        "rb",
                    )
                )
            except:
                cost_history_dict = pickle.load(
                    open(
                        results_dir + sequence + f"_cost_history_dict_av{av_i}.pkl",
                        "rb",
                    )
                )
            try:
                last_params = cost_history_dict["prev_vector"][-1]
                cost_history = cost_history_dict["cost_history"]
            except:
                # cost_trajectory and best_params
                cost_history = cost_history_dict["cost_trajectory"]
                last_params = cost_history_dict["best_params"]
            lowest_cost_array.append(cost_history[-1])

        # find which average iteration has the lowest cost
        lowest_cost = min(lowest_cost_array)
        best_average_i = lowest_cost_array.index(lowest_cost)
        print("Lowest cost: ", lowest_cost, " at average iteration: ", best_average_i)
        try:
            cost_history_dict = pickle.load(
                open(
                    results_dir
                    + sequence
                    + f"_av{best_average_i}__cost_history_dict.pkl",
                    "rb",
                )
            )
        except:
            cost_history_dict = pickle.load(
                open(
                    results_dir
                    + sequence
                    + f"_cost_history_dict_av{best_average_i}.pkl",
                    "rb",
                )
            )
        try:
            best_params = cost_history_dict["prev_vector"][-1]
        except:
            best_params = cost_history_dict["best_params"]

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=3,
            initial_layout=None,
            routing_method="none",
        )
        tp_qc = pm.run(qc)
        print(
            f"Transpiled circuit qubit layout: {tp_qc.layout.final_index_layout(filter_ancillas=True)}"
        )
        pub = (tp_qc, best_params)
        circ_name = lattice + "_" + str(nearest_neighbors) + "nn_" + sequence
        # Add the circuit to the circuits dictionary
        if circ_name not in circuits:
            circuits[circ_name] = [pub]
        else:
            circuits[circ_name].append(pub)
# Print the number of circuits created
print(f"Number of circuits created: {len(circuits)}\n")

# Set up the HW results directory
hw_results_dir = current_file_directory + "/hw_results/"
timestamp_str = TIMESTAMP.strftime("%Y-%m-%d-%H-%M-%S")
out_dir = hw_results_dir + timestamp_str + "/"
os.makedirs(out_dir, exist_ok=True)

# Submit the circuits to the backend in session mode
with Session(backend=backend) as session:
    print("📡 Starting Qiskit Runtime session... \n")
    sampler = Sampler(mode=session)
    sampler.options.default_shots = num_shots
    sampler.options.dynamical_decoupling.enable = True
    # Enable twirling
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    for i, item in enumerate(circuits.items()):
        circ_name, pub = item
        job = sampler.run(pub)
        print(f"Submitted circuit {i}: {circ_name} with job ID: {job.job_id()} \n")
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # Save the results to a json file
        with open(f"{out_dir}/{circ_name}_counts.json", "w") as f:
            json.dump(counts, f)

print(f"🎉 All runs completed! Results saved to {out_dir}")

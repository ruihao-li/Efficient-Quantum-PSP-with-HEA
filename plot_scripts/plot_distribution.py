import matplotlib.pyplot as plt
import numpy as np

from plot_utils import *
from train_qc import *

small = False

lattices_list = ["tetrahedral", "bcc", "fcc"]
nn_list = [1, 2]  # nearest neighbors
num_layers = 1  # default number of layers
font_size = 14

versions = []
# directory_names.append(
#     f"tetrahedral_same_num_layers_2_100000shots_cvar_1996_nn_1_RealAmplitudes"
# )
for lattice in lattices_list:
    for nn in nn_list:
        versions.append((lattice, nn))

#hw_load_directory = current_file_directory + "/hw_results/2025-06-19-14-24-46/"
hw_load_directories = ["2025-09-08-16-20-26",
                        "2025-09-08-19-04-58",
                        "2025-09-09-18-10-55",
                        "2025-09-09-21-10-23",
                        "2025-09-09-22-42-46"]


colors, marker_list, marker_size = set_plot_constants(font_size)

pdb_names = get_pdb_names(lattice=None)

save_plot_directory = "plots/energy_distributions/"
if not os.path.exists(save_plot_directory):
    os.makedirs(save_plot_directory, exist_ok=True)

if small:
    fig, axs = plt.subplots(
        5, 6, figsize=(20, 12), squeeze=False
    )
    tetrahedral_plot_list = ['2MZX', '2NDC', '2NDE', '8B1X', '6A8Y']
    bcc_plot_list = ['ABCP', '1N9U', '2N5R', '1K43', '8T61']
    fcc_plot_list = ['2OL9', '2M6C', '1IXU', '2L24', '1K43']
else:
    pass



for version_i, version in enumerate(versions):
    lattice = version[0]
    nearest_neighbors = version[1]

    whole_dataset, aa_list, save_file_at, load_directory, variables = get_whole_dataset(
        lattice, nearest_neighbors, num_layers
    )
    locals().update(variables)
    print(f"Default shots: {default_shots}")
    num_proteins = len(whole_dataset)

    plot_lattice_i = -1

    if not small and nearest_neighbors == 1:
        fig, axs = plt.subplots(
        len(whole_dataset), 2, figsize=(6.5, len(whole_dataset)*2.5), squeeze=False
        )

    for protein_i, protein_name in enumerate(whole_dataset):
        protein_info, sequence, ground_energy, ground_state_bitstring, num_aa, average_iter = get_protein_info(whole_dataset, protein_name, nearest_neighbors, lattice, load_directory, top_n_folds=50)
        pdb_name = pdb_names[sequence]
        print(pdb_name, sequence, num_aa, lattice, nearest_neighbors)
        if small:
            if (lattice == 'tetrahedral' and pdb_name not in tetrahedral_plot_list) or (lattice == 'bcc' and pdb_name not in bcc_plot_list) or (lattice == 'fcc' and pdb_name not in fcc_plot_list):
                continue
            else:
                plot_lattice_i += 1
        else:
            plot_lattice_i += 1
    
        try:
            with open(load_directory + sequence + "_prob_energy_pairs.json", "r") as f:
                prob_energy_pairs = json.load(f)
        except:
            best_average_i, _, _ = load_best_average_cost(
                average_iter, load_directory, sequence
            )
            last_params, cost_history = load_best_cost_history(
                best_average_i, load_directory, sequence
            )
            num_qubits = get_number_of_qubits(lattice, sequence)
            protein_solver = get_protein_solver(
                num_qubits, num_layers, lattice, sequence, nearest_neighbors
            )
            prob_energy_pairs = params_to_probs(
                protein_solver, last_params, load_directory, sequence
            )

        # plot energy distribution, one plot for each protein
        energies_to_plot = []
        probs_to_plot = []
        # find all unique energies
        prob_energy_pairs = sorted(prob_energy_pairs, key=lambda x: x[1])  # sort by energy

        # get the HW prob_energy_pair of the one with the lowest cost
        hw_cost_list = []
        for hw_load_directory in hw_load_directories:
            hw_load_directory = current_file_directory + "/hw_results/" + hw_load_directory + "/"
            with open(hw_load_directory + lattice + '_' + str(nearest_neighbors) + "nn_" + sequence + "_average_cost.json", 'r') as f:
                hw_average_cost = json.load(f)
            hw_cost_list.append(hw_average_cost)
        best_hw_i = np.argmin(hw_cost_list)
        print("Best hw run: ", hw_load_directories[best_hw_i])
        # load HW prob_energy_pairs
        hw_load_directory = hw_load_directories[best_hw_i]
        hw_load_directory = current_file_directory + "/hw_results/" + hw_load_directory + "/"
        with open(hw_load_directory + lattice + '_' + str(nearest_neighbors) + "nn_" + sequence + "_prob_energy_pairs.json", 'r') as f:
            hw_prob_energy_pairs = json.load(f)

        # get the HW prob_energy_pair of the one with the lowest cost
        hw_cost_list = []
        for hw_load_directory in hw_load_directories:
            hw_load_directory = (
                current_file_directory + "/hw_results/" + hw_load_directory + "/"
            )
            with open(
                hw_load_directory
                + lattice
                + "_"
                + str(nearest_neighbors)
                + "nn_"
                + sequence
                + "_average_cost.json",
                "r",
            ) as f:
                hw_average_cost = json.load(f)
            hw_cost_list.append(hw_average_cost)
        best_hw_i = np.argmin(hw_cost_list)
        print("Best hw run: ", hw_load_directories[best_hw_i])
        # load HW prob_energy_pairs
        hw_load_directory = hw_load_directories[best_hw_i]
        hw_load_directory = (
            current_file_directory + "/hw_results/" + hw_load_directory + "/"
        )
        with open(
            hw_load_directory
            + lattice
            + "_"
            + str(nearest_neighbors)
            + "nn_"
            + sequence
            + "_prob_energy_pairs.json",
            "r",
        ) as f:
            hw_prob_energy_pairs = json.load(f)

        hw_energies_to_plot = []
        hw_prob_energy_pairs = sorted(hw_prob_energy_pairs, key=lambda x: x[1])

        for state in prob_energy_pairs:
            if state[1] >= 0.0:
                continue
            energies_to_plot.extend(
                [state[1] / abs(ground_energy)] * int(state[0] * default_shots)
            )

        weights = np.ones_like(energies_to_plot) / default_shots

        for state in hw_prob_energy_pairs:
            if state[1] >= 0.0:
                continue
            hw_energies_to_plot.extend(
                [state[1] / abs(ground_energy)] * int(state[0] * default_shots)
            )

        hw_weights = np.ones_like(hw_energies_to_plot) / default_shots

        # For reference, generate `num_shots` random bitstrings of length of `num_qubits` and plot their energy distribution

        # First try loading precomputed random distribution
        random_dir = "random_dist/"
        try:
            with open(
                random_dir
                + lattice
                + "_"
                + str(nearest_neighbors)
                + "nn_"
                + sequence
                + "_random_prob_energy_pairs.json",
                "r",
            ) as f:
                random_prob_energy_pairs = json.load(f)
            print("Loaded precomputed random distribution.")

        except:
            print("Generating random distribution...")

            num_qubits = get_number_of_qubits(lattice, sequence)
            print("Number of qubits: ", num_qubits)
            num_shots = 100_000
            protein_solver = get_protein_solver(
                num_qubits, num_layers, lattice, sequence, nearest_neighbors
            )
            random_bitstrings = [
                "".join(np.random.choice(["0", "1"], size=num_qubits))
                for _ in range(num_shots)
            ]
            # Create counts dictionary of unique bitstrings
            counts = {}
            for bitstring in random_bitstrings:
                if bitstring in counts:
                    counts[bitstring] += 1
                else:
                    counts[bitstring] = 1
            random_prob_energy_pairs = counts_to_probs(protein_solver, counts)
            # Save to file
            if not os.path.exists(random_dir):
                os.makedirs(random_dir, exist_ok=True)
            with open(
                random_dir
                + lattice
                + "_"
                + str(nearest_neighbors)
                + "nn_"
                + sequence
                + "_random_prob_energy_pairs.json",
                "w",
            ) as f:
                json.dump(random_prob_energy_pairs, f)
            print("Saved random distribution for future use.")

        random_energies_to_plot = []
        for state in random_prob_energy_pairs:
            # print(f"Energy of random state: {state[1]}")
            if state[1] >= 0.0:
                continue
            random_energies_to_plot.extend(
                [state[1] / abs(ground_energy)] * int(state[0] * default_shots)
            )
        # print(random_energies_to_plot)
        random_weights = np.ones_like(random_energies_to_plot) / default_shots


        width_bin = 0.07
        bin_edges = np.arange(-1.025, 0.05, 0.05)

        # Create subplots: one subplot per protein in this directory

        if not small:
            version_i = nearest_neighbors - 1

        axs[plot_lattice_i, version_i].hist(
            energies_to_plot,
            bins=bin_edges,
            weights=weights,
            color="green",
            #color=colors[protein_i],
            edgecolor="green",
            alpha=0.5,
            label = "Simulated run",
        )
        # hw
        axs[plot_lattice_i, version_i].hist(
            hw_energies_to_plot,
            bins=bin_edges,
            weights=hw_weights,
            color="red",
            edgecolor="red",
            alpha=0.5,
            label="Hardware run",
        )
        # random
        axs[plot_lattice_i, version_i].hist(
            random_energies_to_plot,
            bins=bin_edges,
            weights=random_weights,
            color="blue",
            edgecolor="blue",
            alpha=0.5,
            label="Random sampling",
        )

        axs[plot_lattice_i, version_i].axvline(
            x=ground_energy / abs(ground_energy),
            color="r",
            linestyle="--",
            label="Ground energy",
        )
        axs[plot_lattice_i, version_i].set_xlabel("")
        axs[plot_lattice_i, version_i].set_xlim((-1.02, 0.0))
        axs[plot_lattice_i, version_i].set_ylim((0.0, 0.31))
        axs[plot_lattice_i, version_i].grid()

        if small:
            if lattice == 'tetrahedral' and nearest_neighbors == 1:
                axs[plot_lattice_i, version_i].set_yticks(np.arange(0.0, 0.31, 0.1))
            else: 
                axs[plot_lattice_i, version_i].set_yticks(np.arange(0.0, 0.31, 0.1), ['', '', '', ''])

            if plot_lattice_i == 4:
                axs[plot_lattice_i, version_i].set_xticks(np.arange(-1.0, 0.1, 0.25),  ['-1.0', '', '-0.5', '', '0.0'])
            else:
                axs[plot_lattice_i, version_i].set_xticks(np.arange(-1.0, 0.1, 0.25), ['', '', '', '', ''])# ['-1.0', '', '-0.5', '', '0.0'], fontsize=font_size)
        else:
            if nearest_neighbors == 1:
                axs[plot_lattice_i, version_i].set_yticks(np.arange(0.0, 0.31, 0.1))
            else: 
                axs[plot_lattice_i, version_i].set_yticks(np.arange(0.0, 0.31, 0.1), ['', '', '', ''])

            if plot_lattice_i == len(whole_dataset)-1:
                axs[plot_lattice_i, version_i].set_xticks(np.arange(-1.0, 0.1, 0.25),  ['-1.0', '', '-0.5', '', '0.0'])
            else:
                axs[plot_lattice_i, version_i].set_xticks(np.arange(-1.0, 0.1, 0.25), ['', '', '', '', ''])# ['-1.0', '', '-0.5', '', '0.0'], fontsize=font_size)

        # put the title in the top right coner in a box
        axs[plot_lattice_i, version_i].text(
            0.95,
            0.95,
            f"{pdb_names[sequence]}, {num_aa}-AA",
            transform=axs[plot_lattice_i, version_i].transAxes,
            fontsize=font_size,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

    if not small and nearest_neighbors == 2:
        axs[len(whole_dataset)-1,1].legend(loc="upper left", fontsize=font_size-2)
        axs[len(whole_dataset)-1,0].set_xlabel("Normalized Energy", fontsize=font_size)
        axs[0,0].set_ylabel("Probability", fontsize=font_size)
        if lattice == 'tetrahedral':
            axs[0, 0].set_title(
                "Tetrahedral", fontsize=font_size
            )
        elif lattice == 'fcc':
            axs[0, 0].set_title(
                "FCC", fontsize=font_size
            )
        elif lattice == 'bcc':
            axs[0, 0].set_title(
            "BCC", fontsize=font_size
            )
        plt.tight_layout()
        plt.savefig(
            current_file_directory + "/plots/" + "distribution_plot_" + lattice + ".pdf", bbox_inches="tight"
        )
        plt.close()

if small:
    axs[4,5].legend(loc="upper right", fontsize=font_size-2)
    axs[4,0].set_xlabel("Normalized Energy", fontsize=font_size)
    axs[2,0].set_ylabel("Probability", fontsize=font_size)
            
    axs[0, 4].set_title(
        "FCC", fontsize=font_size
    )
    axs[0, 2].set_title(
        "BCC", fontsize=font_size
    )
    axs[0, 0].set_title(
        "Tetrahedral", fontsize=font_size
    )
    plt.tight_layout()
    plt.savefig(
        current_file_directory + "/plots/" + "distribution_plot.pdf", bbox_inches="tight"
    )
        
print("Done!")

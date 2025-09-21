import matplotlib.pyplot as plt
import numpy as np
import os

from plot_utils import *
from train_qc import *

# initialize Ray
NUM_WORKERS = psutil.cpu_count()  # Use all available CPUs
if not ray.is_initialized():
    ray.init(
        num_cpus=NUM_WORKERS,
        ignore_reinit_error=True,
        log_to_driver=False,
    )

recompute_prob_energy_pairs = False

lattices_list = ["tetrahedral", "bcc", "fcc"]
nn_list = [1, 2]  # nearest neighbors
num_layers = 1  # default number of layers
font_size = 14
current_file_directory = os.path.dirname(os.path.abspath(__file__))

versions = []
# directory_names.append(
#     f"tetrahedral_same_num_layers_2_100000shots_cvar_1996_nn_1_RealAmplitudes"
# )
for lattice in lattices_list:
    for nn in nn_list:
        versions.append((lattice, nn))


# testing = "2025-08-15-21-31-19_XpXm"
# testing = "2025-08-15-21-22-55_rerun"
# testing = "2025-08-08-13-43-03_XY4"
# testing = "2025-08-17-14-17-24_XpXm"
# testing = "2025-08-17-14-15-06_rerun"
### New data from September 2025 ###
# testing = "2025-09-08-16-20-26"
# testing = "2025-09-08-19-04-58"
# testing = "2025-09-09-18-10-55"
# testing = "2025-09-09-21-10-23"
testing = "2025-09-09-22-42-46"

hw_load_directory = current_file_directory + "/hw_results/" + testing + "/"

colors, marker_list, marker_size = set_plot_constants(font_size)

pdb_names = get_pdb_names(lattice=None)

cvar_alpha_list = [0.01, 0.05, 0.1, 0.2]


for to_plot in ["roa", "diff"]:
    labeled = False
    fig, axs2 = plt.subplots(
        len(nn_list), len(lattices_list), figsize=(20, 12), squeeze=False
    )
    for ax in axs2.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    lattice_i = 0
    for version_i, version in enumerate(versions):
        lattice = version[0]
        if lattice == "fcc":
            lattice_i = 3
        elif lattice == "bcc":
            lattice_i = 2
        elif lattice == "tetrahedral":
            lattice_i = 1
        nearest_neighbors = version[1]

        whole_dataset, aa_list, save_file_at, load_directory, variables = (
            get_whole_dataset(lattice, nearest_neighbors, num_layers)
        )
        locals().update(variables)
        num_proteins = len(whole_dataset)
        print(load_directory)

        sequence_list = [
            get_sequence(protein_name, whole_dataset, lattice, nearest_neighbors)
            for protein_name in whole_dataset
        ]

        xticks_labels = []
        for seq in sequence_list:
            xticks_labels.append(
                str(pdb_names[seq] + ", " + str(aa_list[sequence_list.index(seq)]))
            )
        for protein_i, protein_name in enumerate(whole_dataset):
            (
                protein_info,
                sequence,
                ground_energy,
                ground_state_bitstring,
                num_aa,
                average_iter,
            ) = get_protein_info(
                whole_dataset,
                protein_name,
                nearest_neighbors,
                lattice,
                load_directory,
                top_n_folds=50,
            )
            average_iter = 10

            # get the hw
            try:
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_counts.json",
                    "r",
                ) as f:
                    counts = json.load(f)
            except:
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_counts_error_miti.json",
                    "r",
                ) as f:
                    counts = json.load(f)
                # resave
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_counts.json",
                    "w",
                ) as f:
                    json.dump(counts, f)

            if recompute_prob_energy_pairs:
                num_qubits = get_number_of_qubits(lattice, sequence)
                protein_solver = get_protein_solver(
                    num_qubits, num_layers, lattice, sequence, nearest_neighbors
                )
                hw_prob_energy_pairs = counts_to_probs(protein_solver, counts)
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_prob_energy_pairs.json",
                    "w",
                ) as f:
                    json.dump(hw_prob_energy_pairs, f)

                hw_average_cost = cvar(hw_prob_energy_pairs)
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_average_cost.json",
                    "w",
                ) as f:
                    json.dump(hw_average_cost, f)

                hw_best_bitstring = sorted(hw_prob_energy_pairs, key=lambda x: x[1])[0][1]
                with open(
                    hw_load_directory
                    + lattice
                    + "_"
                    + str(nearest_neighbors)
                    + "nn_"
                    + sequence
                    + "_best_bitstring.json",
                    "w",
                ) as f:
                    json.dump(hw_best_bitstring, f)
            else:
                try:
                    # with open(
                    #     hw_load_directory
                    #     + lattice
                    #     + "_"
                    #     + str(nearest_neighbors)
                    #     + "nn_"
                    #     + sequence
                    #     + "_average_cost.json",
                    #     "r",
                    # ) as f:
                    #     hw_average_cost = json.load(f)
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
                    hw_average_cost_list = []
                    for cvar_alpha in cvar_alpha_list:
                        hw_average_cost_list.append(cvar(hw_prob_energy_pairs, cvar_alpha=cvar_alpha))
                    with open(
                        hw_load_directory
                        + lattice
                        + "_"
                        + str(nearest_neighbors)
                        + "nn_"
                        + sequence
                        + "_best_bitstring.json",
                        "r",
                    ) as f:
                        hw_best_bitstring = json.load(f)
                except:
                    num_qubits = get_number_of_qubits(lattice, sequence)
                    protein_solver = get_protein_solver(
                        num_qubits, num_layers, lattice, sequence, nearest_neighbors
                    )
                    hw_prob_energy_pairs = counts_to_probs(protein_solver, counts)
                    with open(
                        hw_load_directory
                        + lattice
                        + "_"
                        + str(nearest_neighbors)
                        + "nn_"
                        + sequence
                        + "_prob_energy_pairs.json",
                        "w",
                    ) as f:
                        json.dump(hw_prob_energy_pairs, f)

                    hw_average_cost = cvar(hw_prob_energy_pairs)
                    with open(
                        hw_load_directory
                        + lattice
                        + "_"
                        + str(nearest_neighbors)
                        + "nn_"
                        + sequence
                        + "_average_cost.json",
                        "w",
                    ) as f:
                        json.dump(hw_average_cost, f)

                    hw_best_bitstring = sorted(hw_prob_energy_pairs, key=lambda x: x[1])[0][1]
                    with open(
                        hw_load_directory
                        + lattice
                        + "_"
                        + str(nearest_neighbors)
                        + "nn_"
                        + sequence
                        + "_best_bitstring.json",
                        "w",
                    ) as f:
                        json.dump(hw_best_bitstring, f)

            if to_plot == "roa":
                best_average_i, lowest_average_cost_list, lowest_average_cost = (
                    load_best_average_cost(average_iter, load_directory, sequence)
                )
                value_to_plot_mean, value_to_plot_sterror = calculate_RE_mean_sterror(
                    lowest_average_cost_list, ground_energy, average_iter
                )

                value_to_plot_lowest = relative_error(
                    lowest_average_cost, ground_energy
                )
                value_to_plot_hw = [relative_error(hw_average_cost, ground_energy) for hw_average_cost in hw_average_cost_list]
            if to_plot == "diff":
                best_bitstring_i, best_bitstring_list, best_bitstring = (
                    load_best_bitstrings(average_iter, load_directory, sequence)
                )
                value_to_plot_mean, value_to_plot_sterror = calculate_RE_mean_sterror(
                    best_bitstring_list, ground_energy, average_iter
                )

                value_to_plot_lowest = relative_error(best_bitstring, ground_energy)
                value_to_plot_hw = relative_error(hw_best_bitstring, ground_energy)

            best_bitstring_i, best_bitstring_list, best_bitstring = (
                load_best_bitstrings(average_iter, load_directory, sequence)
            )

            print(sequence)
            print("\n\nground_energy, lowest_average_cost, best_bitstring")
            print(ground_energy, lowest_average_cost, best_bitstring)
            print("\n\nhw_average_cost, hw_best_bitstring")
            print(hw_average_cost_list, hw_best_bitstring)
            print("\n\nvalue_to_plot_mean, value_to_plot_lowest, value_to_plot_hw")
            print(value_to_plot_mean, value_to_plot_lowest, value_to_plot_hw)
            print()

            # plot
            if not labeled:
                axs2[nearest_neighbors - 1, lattice_i - 1].errorbar(
                    protein_i,
                    value_to_plot_mean,
                    yerr=value_to_plot_sterror,
                    # label=sequence,
                    color="black",
                    marker="x",
                    capsize=5,
                    markersize=marker_size,
                    label="Simulated average value",
                )
                axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                    protein_i,
                    value_to_plot_lowest,
                    color="green",
                    marker="o",
                    s=50,
                    label="Simulated lowest value",
                )
                if isinstance(value_to_plot_hw, list):
                    for idx, cvar_alpha in enumerate(cvar_alpha_list):
                        axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                            protein_i,
                            value_to_plot_hw[idx],
                            color=colors[idx],
                            marker=marker_list[idx],
                            s=100,
                            label=f"Hardware run value (CVaR α={cvar_alpha})",
                        )
                else:
                    axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                        protein_i,
                        value_to_plot_hw,
                        color="red",
                        marker="*",
                        s=100,
                        label="Hardware run value",
                    )
                labeled = True
            else:
                axs2[nearest_neighbors - 1, lattice_i - 1].errorbar(
                    protein_i,
                    value_to_plot_mean,
                    yerr=value_to_plot_sterror,
                    # label=sequence,
                    color="black",
                    marker="x",
                    capsize=5,
                    markersize=marker_size,
                )
                axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                    protein_i,
                    value_to_plot_lowest,
                    color="green",
                    marker="o",
                    s=50,
                )
                if isinstance(value_to_plot_hw, list):
                    for idx, cvar_alpha in enumerate(cvar_alpha_list):
                        axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                            protein_i,
                            value_to_plot_hw[idx],
                            color=colors[idx],
                            marker=marker_list[idx],
                            s=100,
                        )
                else:   
                    axs2[nearest_neighbors - 1, lattice_i - 1].scatter(
                        protein_i,
                        value_to_plot_hw,
                        color="red",
                        marker="*",
                        s=100,
                    )
            # plt.xlabel("Number of amino acids")

            axs2[nearest_neighbors - 1, lattice_i - 1].set_xticks(
                np.arange(len(whole_dataset)), xticks_labels, rotation=45
            )

            # axs2[nearest_neighbors - 1, lattice_i - 1].set_ylim((-0.03, 2.03))
            # axs2[nearest_neighbors - 1, lattice_i - 1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if nearest_neighbors == 1:
                if lattice == "fcc":
                    axs2[nearest_neighbors - 1, lattice_i - 1].set_title(
                        "FCC", fontsize=font_size
                    )
                elif lattice == "bcc":
                    axs2[nearest_neighbors - 1, lattice_i - 1].set_title(
                        "BCC", fontsize=font_size
                    )
                elif lattice == "tetrahedral":
                    axs2[nearest_neighbors - 1, lattice_i - 1].set_title(
                        "Tetrahedral", fontsize=font_size
                    )
            if protein_i == len(whole_dataset) - 1:
                axs2[nearest_neighbors - 1, lattice_i - 1].grid()

    axs2[0, 0].legend()
    if to_plot == "roa":
        axs2[0, 0].set_ylabel("Average relative error")
        plt.savefig(
            current_file_directory
            + "/plots/"
            + "approximation_ratio_plot_"
            + testing
            + ".pdf",
            bbox_inches="tight",
        )
        plt.close()
    if to_plot == "diff":
        axs2[0, 0].set_ylabel("Best case relative error")
        plt.savefig(
            current_file_directory
            + "/plots/"
            + "difference_to_groundstate_plot_"
            + testing
            + ".pdf",
            bbox_inches="tight",
        )
        plt.close()
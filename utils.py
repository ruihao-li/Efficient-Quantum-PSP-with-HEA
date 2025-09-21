import sys
from collections import defaultdict

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import re
import os


def plot_cost(
    cost_vector,
    name,
    pdf: bool = False,
    new_fig: bool = True,
    fig_num: int = None,
    ground_energy: float = None,
    label: str = "Cost",
    color: str = None,
):
    if new_fig:
        plt.figure()
    if not fig_num == None:
        plt.figure(fig_num)
    plt.plot(cost_vector, label=label, color=color)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if ground_energy is not None:
        plt.plot(
            np.full(len(cost_vector), ground_energy),
            label=label + " ground energy",
            color=color,
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if pdf:
        plt.savefig(name + ".pdf", bbox_inches="tight")
    else:
        print("saving as jpg")
        plt.savefig(name + ".jpg")


def save_constants_to_file(file_path: str, constants_dict: dict) -> None:
    with open(file_path, "w") as f:
        for key, value in constants_dict.items():
            f.write(f"{key}: {value}\n")
    f.close()


def load_constants_from_file(file_path: str) -> str:
    return_dict = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line.strip("\n")
            if not line.startswith("dataset: "):
                key, value = line.split(": ")
                value = value.strip("\n")
                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        pass
                return_dict[key] = value
            else:  # its the dataset # dataset: {'7AA_APRLRFY_1NN_BCC': <dataset_class.ProteinData object at 0x14c85a7fd1f0>, '8AA_GCVLYPWC_1NN_BCC': <dataset_class.ProteinData object at 0x14c850e20350>, '10AA_DRVYIHPFHL_1NN_BCC': <dataset_class.ProteinData object at 0x14c850de0830>, '10AA_YYDPETGTWY_1NN_BCC': <dataset_class.ProteinData object at 0x14c85438c4d0>, '12AA_VRRFDLLKRILK_1NN_BCC': <dataset_class.ProteinData object at 0x14c8540449b0>, '13AA_IFGAIAGFIKNIW_1NN_BCC': <dataset_class.ProteinData object at 0x14c85a7fe420>, '14AA_RGKWTYNGITYEGR_1NN_BCC': <dataset_class.ProteinData object at 0x14c853dfe8a0>, '16AA_RHYYKFNSTGRHYHYY_1NN_BCC': <dataset_class.ProteinData object at 0x14c853dffd40>}
                # save all data eclosed by ''
                key = "dataset"
                value = re.findall(r"'(.*?)'", line)
                return_dict[key] = value

    f.close()
    return return_dict


def save_text_to_file(file_path: str, text: str) -> None:
    with open(file_path, "w") as f:
        f.write(text)
    f.close()


def count_gates(circuit):
    counts = defaultdict()
    counts[1] = 0
    counts[2] = 0
    for inst in circuit.data:
        counts[inst.operation.num_qubits] += 1
    return counts


def get_filenames_os(directory_path):
    """
    Gets all file names under a given directory using os.listdir().

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of file names, or None if the directory doesn't exist.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return None

    filenames = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            filenames.append(entry)
    return sorted(filenames)

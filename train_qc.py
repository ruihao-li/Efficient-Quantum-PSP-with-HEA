import time
import numpy as np

import ray
import psutil
from itertools import chain
from qiskit.result import Counts
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit import QuantumCircuit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle


class ProteinSolver:
    """
    Class for training a quantum circuit to analyze a protein.

    Args:
        ansatz (QuantumCircuit): The ansatz circuit from Qiskit.
        sampler (Sampler): The quantum sampler from Qiskit Aer or hardware.
        energy_function (Callable): The function to calculate the energy of a bitstring.
        energy_args (dict): The arguments of the energy function.
        cost_function (Callable): The function to calculate the cost of a list of energies.
        cost_args (tuple): The arguments of the cost function.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        sampler: Sampler,
        energy_function: callable,
        energy_args: dict,
        cost_function: callable,
        cost_args: tuple,
        save_file_at: str = "",
        sequence: str = "",
    ):
        self.ansatz = ansatz
        self.sampler = sampler
        self.energy_function = energy_function
        self.energy_args = energy_args
        self.cost_function = cost_function
        self.cost_args = cost_args
        self.global_bitstring_energies: dict[str, float] = {}
        self.cost_history_dict = {
            "iters": 0,
            "prev_vector": [],
            "cost_history": [],
            "job_id": [],
        }
        self.sequence = sequence
        self.save_file_at = save_file_at

        # Initialize Ray
        # ray.init(ignore_reinit_error=True, num_cpus=psutil.cpu_count())

    def qc_cost_function(
        self,
        params: np.ndarray,
        num_batches: int | None = None,
        save: bool = True,
        verbose: bool = False,
        #train: bool = True,
    ) -> float:
        """
        Compute the cost for a given set of parameters of the ansatz.

        Args:
            params (np.ndarray): The parameters of the ansatz.
            num_batches (int, optional): The number of batches to split the
            unique states into. Defaults to None. If None, it will be set to the
            total number of CPUs (may not be desirable).
            verbose (bool): Whether to print the duration of the job and processing.

        Returns:
            float: The cost function of the protein folding problem.
        """
        # 1. Run the quantum circuit as a random number generator
        tic0 = time.time()
        if self.ansatz.num_clbits == 0:
            self.ansatz.measure_all()
        pub = (self.ansatz, params)
        job = self.sampler.run(pubs=[pub])
        primitive_result = job.result()
        pub_result = primitive_result[0].data
        counts = pub_result.meas.get_counts()  # count occurrences of each bitstring
        toc1 = time.time()
        job_duration = toc1 - tic0

        # 2. Process the counts, calculating the energy of unique bitstrings that have not been processed before
        tic2 = time.time()

        state_wise_energies, prob_energy_pairs = process_counts(
            counts,
            self.energy_function,
            self.energy_args,
            num_batches=num_batches,
            global_bitstring_energies=self.global_bitstring_energies,
        )

        # 3. Calculate the cost of the measurements
        cost = self.cost_function(prob_energy_pairs, *self.cost_args)
        toc2 = time.time()
        process_duration = toc2 - tic2

        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["prev_vector"].append(params)
        self.cost_history_dict["cost_history"].append(cost)
        self.cost_history_dict["job_id"].append(job.job_id())
        if save:
            with open(self.save_file_at + "_cost_history_dict.pkl", "wb") as f:
                pickle.dump(self.cost_history_dict, f)

        if verbose:
            print(f" >> Sampler job took {job_duration:.2f} seconds")
            print(
                f" >> Processing {len(state_wise_energies)} unique bitstrings took {process_duration:.2f} seconds"
            )
        return cost

    def get_prob_energy_pairs(
        self,
        params: np.ndarray,
        num_batches: int | None = None,
        #train: bool = True,
    ) -> float:
        """
        Compute the cost for a given set of parameters of the ansatz.

        Args:
            params (np.ndarray): The parameters of the ansatz.
            num_batches (int, optional): The number of batches to split the
            unique states into. Defaults to None. If None, it will be set to the
            total number of CPUs (may not be desirable).
            verbose (bool): Whether to print the duration of the job and processing.

        Returns:
            list[tuple[float, float]]: A list of tuples containing the probability and energy of each unique bitstring.
        """
        # 1. Run the quantum circuit as a random number generator
        if self.ansatz.num_clbits == 0:
            self.ansatz.measure_all()
        pub = (self.ansatz, params)
        job = self.sampler.run(pubs=[pub])
        primitive_result = job.result()
        pub_result = primitive_result[0].data
        counts = pub_result.meas.get_counts()  # count occurrences of each bitstring

        _, prob_energy_pairs = process_counts(
            counts,
            self.energy_function,
            self.energy_args,
            num_batches=num_batches,
            global_bitstring_energies=self.global_bitstring_energies,
        )
        return prob_energy_pairs


    def train(
        self,
        init_params: np.ndarray,
        optimizer: str,
        maxiter: int,
        num_batches: int | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Train the quantum circuit to fold a protein.

        Args:
            init_params (np.ndarray): The initial parameters of the ansatz.
            optimizer (str): The optimizer to use.
            maxiter (int): The maximum number of iterations for the optimizer.
            num_batches (int, optional): The number of batches to split the
            unique states into. Defaults to None. If None, it will be set to the
            total number of CPUs.
            verbose (bool): Whether to print the duration of the job and processing.

        Returns:
            tuple[np.ndarray, float]: The optimized parameters and the final cost.
        """
        optimizer_result = minimize(
            self.qc_cost_function,
            init_params,
            method=optimizer,
            args=(num_batches, verbose),
            options={"maxiter": maxiter, "disp": True},
        )
        return optimizer_result.x, optimizer_result.fun

    def plot_cost_trajectory(
        self, name: str = "cost_trajectory", pdf: bool = False, new_fig: bool = True
    ):
        """
        Plot the cost trajectory.

        Args:
            name (str): The name of the plot.
            pdf (bool): Whether to save the plot as a PDF.
            new_fig (bool): Whether to create a new figure.
        """
        if new_fig:
            plt.figure()
        cost_trajectory = self.cost_history_dict["cost_history"]
        plt.plot(cost_trajectory, label=self.sequence)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        if pdf:
            plt.savefig(name + ".pdf")
        else:
            plt.show()

    def get_hitrate(
        self, params: np.ndarray, shots: int, ground_energy: float
    ) -> float:
        """
        Get the hitrate of the quantum circuit.

        Args:
            params (np.ndarray): The parameters of the ansatz.
            shots (int): The number of shots to run.

        Returns:
            float: The hitrate of the quantum circuit.
        """
        if self.ansatz.num_clbits == 0:
            self.ansatz.measure_all()
        pub = (self.ansatz, params)
        job = self.sampler.run(pubs=[pub], shots=shots)
        primitive_result = job.result()
        pub_result = primitive_result[0].data
        counts = pub_result.meas.get_counts()
        tot_counts = sum(counts.values())
        hit = 0
        for string, count in counts.items():
            try:
                e = self.global_bitstring_energies[string]
            except:
                e = self.energy_function(string, **self.energy_args)
            if np.isclose(e, ground_energy, atol=1e-3):
                hit += count

            tot_counts += count
        hit /= tot_counts
        return hit

    def get_top_bitstrings(
        self, params: np.ndarray, shots: int, top_n: int = 50
    ) -> list[tuple[str, float]]:
        """
        Get the top bitstrings with their energies.

        Args:
            params (np.ndarray): The parameters of the ansatz.
            shots (int): The number of shots to run.
            top_n (int): The number of top bitstrings to return.

        Returns:
            list[tuple[str, float]]: List of tuples containing the bitstring and its energy.
        """
        if self.ansatz.num_clbits == 0:
            self.ansatz.measure_all()
        pub = (self.ansatz, params)
        job = self.sampler.run(pubs=[pub], shots=shots)
        primitive_result = job.result()
        pub_result = primitive_result[0].data
        counts = pub_result.meas.get_counts()

        # Sort the counts by energy
        bitstring_energies = {
            string: self.global_bitstring_energies.get(
                string, self.energy_function(string, **self.energy_args)
            )
            for string in counts.keys()
        }

        sorted_bitstrings = sorted(bitstring_energies.items(), key=lambda x: x[1])

        return sorted_bitstrings[:top_n]


@ray.remote
def calculate_batch_energy(
    conf_bitstring_list: list[str],
    energy_function: callable,
    energy_args: tuple,
) -> list[float]:
    """
    Calculate the energy of a batch of bitstrings.

    Args:
        conf_bitstring_list (list[str]): List of conformation bitstrings.
        energy_function (callable): The energy function.
        energy_args (tuple): The arguments of the energy function.

    Returns:
        list[float]: List of energies of the batch of bitstrings.
    """
    return [
        energy_function(bitstring, **energy_args) for bitstring in conf_bitstring_list
    ]


def process_counts(
    counts: Counts | dict[str, int],
    energy_function: callable,
    energy_args: tuple,
    num_batches: int | None = None,
    global_bitstring_energies: dict[str, float] = {},
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    """
    Process a Counts distribution in parallel batches. First, it splits all
    unique states (bitstrings) in the Counts distribution into specified number
    of batches. Then, for each batch of states, it computes the energy per
    unique state. It also converts integer count of a state to probability
    (float) by dividing the count by total number of shots.

    Args:
        counts (Counts): The counts of a quantum circuit run.
        energy_function (callable): The energy function.
        energy_args (tuple): The arguments of the energy function.
        num_batches (int, optional): The number of batches to split the unique
        states into. Defaults to None.
        global_bitstring_energies (dict[str, float], optional): The global
        dictionary that keeps track of all bitstrings ever measured. Bitstrings
        that are already in this dictionary will not be processed. Defaults to
        empty dictionary.

    Returns:
        state_wise_energies (dict[str, float]): Dictionary where keys are unique
        bitstrings (states) from `Counts` (with energy below a certain value)
        and values are corresponding energy of the respective state.
        prob_energy_pairs (list[tuple[float, float]]): List of 2-tuples. The
        first element of the tuple is the probability of a state. The seconds
        element of the tuple is the energy of that corresponding state.
    """
    total_shots = sum(counts.values())
    states = list(counts.keys())
    state_counts = np.array(list(counts.values()))
    state_probs = state_counts / total_shots
    # Get the unique states that are not already in the global dictionary
    unique_states = [
        state for state in states if state not in global_bitstring_energies
    ]
    num_unique_states = len(unique_states)

    if num_unique_states == 0:
        all_energies = [global_bitstring_energies[state] for state in states]
        prob_energy_pairs: list[tuple[float, float]] = list(
            zip(state_probs, all_energies)
        )
        return {}, prob_energy_pairs

    if num_batches is None or num_batches > psutil.cpu_count():
        num_batches = psutil.cpu_count()
    if num_batches > num_unique_states:
        num_batches = num_unique_states
    batch_size = num_unique_states // num_batches

    energy_function_id = ray.put(energy_function)
    energy_args_id = ray.put(energy_args)
    doubled_batch_refs = []
    for i in range(0, num_unique_states, batch_size):
        batch_states = unique_states[i : i + batch_size]
        doubled_batch_refs.append(
            calculate_batch_energy.remote(
                batch_states, energy_function_id, energy_args_id
            )
        )
    unique_energies = list(chain(*ray.get(doubled_batch_refs)))

    assert num_unique_states == len(unique_energies)

    state_wise_energies: dict[str, float] = {
        state: energy for state, energy in zip(unique_states, unique_energies)
    }
    # Update the global dictionary
    global_bitstring_energies.update(state_wise_energies)
    all_energies = [global_bitstring_energies[state] for state in states]
    prob_energy_pairs: list[tuple[float, float]] = list(zip(state_probs, all_energies))

    return state_wise_energies, prob_energy_pairs

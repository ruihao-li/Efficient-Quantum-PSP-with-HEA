# Energy/cost functions
"""
Input: list of 2-tuples, where the first element of the tuple is the probability of a state and the second element of the tuple is the energy of that corresponding state.
    args for the functions, e.g. cvar_alpha
Output: float of the cost
"""
import numba as nb


@nb.jit(nopython=True)
def average_cost(prob_energy_pairs: list[tuple[float, float]]) -> float:
    return sum(prob * energy for prob, energy in prob_energy_pairs)


def min_sample(prob_energy_pairs: list[tuple[float, float]]) -> float:
    return sorted(prob_energy_pairs, key=lambda x: x[1])[0][1]


@nb.jit(nopython=True)
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

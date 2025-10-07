# Efficient Quantum Protein Structure Prediction with Hardware-Efficient Ansatz (HEA)

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.12+-blue.svg) ![Qiskit](https://img.shields.io/badge/qiskit-1.4+-green.svg)

## 🔬 Overview

Hamiltonian-free quantum protein structure prediction workflow leveraging problem-agnostic ansatzes, such as a hardware-efficient ansatz (HEA).

Link to the paper: [Efficient Quantum Protein Structure Prediction with Problem-Agnostic Ansatzes](https://arxiv.org/abs/2509.18263)

## 🏗️ Code Structure

```
├── src/                          # Core source code
│   ├── train_qc.py              # Quantum circuit training and optimization
│   ├── energy_functions.py      # Energy calculations using MJ potentials
│   ├── dataset_class.py         # Protein dataset loading and management
│   ├── cost_functions.py        # Optimization cost functions (CVaR)
│   ├── turn_decoder.py          # Bitstring-turn-coordinate conversion
│   ├── utils.py                 # Utility functions
│   └── mj_matrix_1996.txt       # Miyazawa-Jernigan interaction matrix
├── classical_search/            # Reference classical search results
│   └── res/                     # Classical exhaustive search results
│       ├── tetrahedral/         # Tetrahedral lattice results
│       ├── fcc/                 # Face-centered cubic lattice results
│       └── bcc/                 # Body-centered cubic lattice results
├── plot_scripts/                # Visualization and analysis tools
├── example_run.py               # Main execution script
├── hardware_run.py              # Hardware execution script
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Qiskit 1.4+
- Ray for parallel processing
- NumPy, SciPy, Matplotlib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HannaLinn/Efficient-Quantum-PSP-with-HEA.git
cd Efficient-Quantum-PSP-with-HEA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Code

**Basic Example for running noise-free simulation:**
```bash
python example_run.py
```

**Hardware sampling (IBM Quantum credentials required):**
```bash
python hardware_run.py
```

## ⚙️ Configuration

Key parameters in `example_run.py`:

```python
# Protein and lattice settings
lattice = "tetrahedral"                # "tetrahedral", "bcc", or "fcc"
encoding = "binary"                    # Encoding method
energy_file_name = "mj_matrix_1996"    # Statistical potential matrix file
pair_energy_multiplier = 0.1           # Scaling factor for interaction energies
penalty_param = 10.0                   # Overlap penalty parameter

# Quantum circuit settings
num_layers = 1                         # Ansatz depth
ansatz_name = "RealAmplitudes"         # "RealAmplitudes" or "EfficientSU2"
default_shots = 100_000                # Number of quantum measurements

# Optimization settings
cost_function_name = "cvar"            # Cost function: "cvar"
maxiter = 100                          # Maximum optimizer iterations
average_iter = 5                       # Number of optimization runs per protein
```

## 🤝 Main contributors

- [Hanna Linn](https://github.com/HannaLinn) (hannlinn@chalmers.se)
- [Ruihao Li](https://github.com/ruihao-li) (lir9@ccf.org)

We also thank Abdullah Ash Saki (IBM), Frank DiFilippo (Cleveland Clinic), and Tomas Radivoyevitch (Cleveland Clinic) for their valuable contributions to this codebase.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**For questions or support, please reach out to the developers or open an issue in this repository.**

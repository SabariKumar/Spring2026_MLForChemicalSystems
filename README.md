# CSU Spring 2026 — Intro to ML + Molecular Dynamics Simulation

This repository contains theory + practial demonstrations of several ML algorithms, as well as
hands-on demonstration of computing the self-diffusion coefficient of liquid ethanol from MD simulations using two
different force field approaches.

# MD - Ethanol Self-Diffusion Coefficient via Molecular Dynamicss
1. **GAFF2** — a classical, empirically parameterized force field (General AMBER Force Field 2)
2. **MACE-OFF23** — a machine-learned interatomic potential (MLIP) trained on quantum-mechanical data

Both simulations are run with [OpenMM](https://openmm.org/), and the
self-diffusion coefficient is extracted from the mean-squared displacement
(MSD) using [MDAnalysis](https://www.mdanalysis.org/).

## Folder structure

```
├── environment.yml              Conda environment specification
├── 01_gaff2_ethanol_md.ipynb    GAFF2 simulation (build, minimize, equilibrate, production)
├── 02_mace_ethanol_md.ipynb     MACE-OFF23 simulation (ML potential production run)
├── 03_analysis_msd.ipynb        MSD analysis and diffusion coefficient comparison
├── 04_MD_for_VAMPnets.ipynb     Alanine dipeptide 60 ns MD for VAMPnets training data
├── alanine-dipeptide.pdb        Solvated alanine dipeptide input structure
├── models/
│   └── MACE-OFF23_small.model   Pre-downloaded MACE-OFF23 model
└── README.md                    This file
```

## Environment setup

### Prerequisites

- **pixi** — a fast, standalone package manager (see below).
- **Git** (to clone this repository).
- **(Optional) NVIDIA GPU + CUDA drivers** for accelerated simulations.
  Run `nvidia-smi` to check — if it prints driver/GPU info you are good.

### Step 1 — Install pixi

Visit https://pixi.prefix.dev/latest/ and follow the installation steps for your operating system

### Step 2 — Clone the repository

```bash
git clone git@github.com:SabariKumar/Spring2026_MLForChemicalSystems.git
cd CSU_26SP_MolecularDynamics
```

### Step 3 — Check your CUDA version and update `pixi.toml`

The CUDA packages in the environment **must** match the CUDA version
supported by your NVIDIA driver.  If there is a mismatch you will get a
cryptic `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` error at runtime.

Run `nvidia-smi` and look at the top-right of the output:

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
...
```

The number after **CUDA Version** (here `12.8`) is the maximum CUDA
toolkit version your driver can support.  Now open `pixi.toml` and
make sure the `system-requirements cuda` pin matches.  The file ships with:

```
  - cuda =  12.4
```

If your `nvidia-smi` shows a different version (e.g. `12.6`), change the
line accordingly:

```yaml
  - cuda-version 12.6.*
```

**No NVIDIA GPU?**  Remove the `system-requirements` section
entirely — the environment will install CPU-only builds and the notebooks
will fall back to the CPU platform automatically.

### Step 4 — Create the environment

The file `pixi.toml` defines an environment called **md_demo**
with all required packages (OpenMM, openmmforcefields, OpenFF Toolkit,
openmm-ml, mace-torch, MDAnalysis, etc.).

```bash
pixi install
```

### Step 5 — Activate the environment

```bash
pixi shell
```

### Step 6 — Start the jupyter server

If you're running this on a remote server:
Run the following command, replacing the port number (10001) with your favorite number
between 9000 and 12000:
```bash
pixi run jupyter lab --port 10001
```
From the output in the terminal, note the a URL with a port number was printed. Copy this URL, including the token in the URL.
Note the port that the server started on - this may be different from what you specified, if the port you chose was already in use
In a new terminal window, run ssh -L port_of_server:localhost:port_of_server your_user_account@your_remote_machine.chem.colostate.edu 
In a web browser tab, navigate to the URL you copied.
  
  
If you're running this on your machine, just run the first command:
```bash
pixi run jupyter lab --port 10001
``` 

**Using the kernel:** When you open any of the notebooks in Jupyter, select
**Kernel → Change Kernel → Python (md-demo)** from the menu bar.

## Quick start

### 1. Run the notebooks in order

| Notebook | What it does |
|---|---|
| `01_gaff2_ethanol_md.ipynb` | Builds 200 ethanol molecules, parameterizes with GAFF2, runs 500 ps NPT equilibration + 2 ns NVT production |
| `02_mace_ethanol_md.ipynb` | Loads equilibrated box from Notebook 1, runs 200 ps NVT with MACE-OFF23 |
| `03_analysis_msd.ipynb` | Computes MSD from both trajectories, fits diffusion coefficient, compares methods vs. experiment |
| `04_MD_for_VAMPnets.ipynb` | Runs 60 ns NVT of alanine dipeptide at 400 K (AMBER ff14SB + TIP3P), saves 600-frame PDB for VAMPnets training |

### 2. Expected output

The analysis notebook produces:

- MSD vs. lag-time plots
- Linear fits to the diffusive regime
- A bar chart comparing D(GAFF2) vs. D(MACE-OFF23) vs. D(experiment)

The experimental self-diffusion coefficient of ethanol at 25 °C is
approximately **1.06 × 10⁻⁹ m²/s**.

## Background

### Self-diffusion coefficient

The Einstein relation connects the mean-squared displacement to the
self-diffusion coefficient:

```
MSD(τ) = ⟨|r(t+τ) − r(t)|²⟩ → 2dDτ   as τ → ∞
```

where *d* = 3 for three-dimensional diffusion.  In practice, we fit the
linear portion of the MSD curve and compute D = slope / (2d).

### GAFF2

The General AMBER Force Field 2 is a classical force field that uses
fixed-charge Lennard-Jones + Coulomb interactions with bonded terms
(bonds, angles, dihedrals).  Partial charges are assigned via the AM1-BCC
method.  GAFF2 is widely used for small organic molecules.

### MACE-OFF23

MACE-OFF23 is a transferable machine-learning force field for organic
molecules built on the MACE (Multi-ACE) equivariant message-passing
architecture.  It is trained on DFT-level quantum-mechanical data and
captures polarization, charge transfer, and many-body effects that
classical force fields cannot represent.

## Key packages

| Package | Role |
|---|---|
| [OpenMM](https://openmm.org/) | MD simulation engine |
| [openmmforcefields](https://github.com/openmm/openmmforcefields) | GAFF2 parameterization |
| [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit) | Molecule creation and topology |
| [openmm-ml](https://github.com/openmm/openmm-ml) | ML potential interface for OpenMM |
| [mace-torch](https://github.com/ACEsuit/mace) | MACE neural-network potential |
| [MDAnalysis](https://www.mdanalysis.org/) | Trajectory analysis and MSD calculation |

# MIMICO - Schelling Active Learning & XAI Framework

This repository contains the Active Learning (AL) framework designed to optimize the exploration of the Schelling Agent-Based Model (ABM). The framework uses an Explanatory Diversity Sampling (EDS) strategy, merging surrogate modeling with post-hoc Explainable AI (SHAP) to maximize the discovery of novel behaviors and tipping points.

## Project Structure

- `src/`: Core Python modules, including the surrogate models (`models_jfsma.py`), database handlers (`database_jfsma.py`), and GAMA simulator connectors (`gama_jfsma.py`).
- `scripts/`: Main execution scripts for running the active learning loops (e.g., `al_tp_v3.py`).
- `notebooks/`: Jupyter Notebooks for XAI tooling and analysis.
- `data/`: Contains raw data and processed outputs from the active learning loop (like `.csv` databases and `.npy` SHAP values). *Ignored by git for large files.*
- `models/`: Exported proxy models (`.pkl`, `.pmml`, `.jar`).
- `figures/`: Generated analytical plots and visualizations.
- `scratch/`: Experimental scripts and scratchpads.

## Usage

Navigate to the `Schelling_ABS` root and run the execution scripts from the `scripts/` directory:

```bash
python scripts/al_tp_v3.py
```
Outputs and plots will automatically be routed to their respective `data/processed` and `figures` folders.

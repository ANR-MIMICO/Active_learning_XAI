# ANR-JCJC MIMICO: Active Learning & XAI for Complex Systems

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

**MIMICO** (Exploring the synergies between machine learning and agent-based modelling) is an ANR-JCJC funded project (02/2025 - 02/2028).

This repository contains the codebase for the **Explanatory Diversity Sampling (EDS)** methodology, a novel Active Learning framework designed to discover "new explanations" and behavioral regimes (like tipping points) in Agent-Based Models (ABMs). 

Rather than standard variance reduction, this derivative-free optimization framework integrates **Surrogate Modeling (Kriging/Multi-Fidelity)** with **post-hoc Explainable Artificial Intelligence (SHAP)** within an iterative Bayesian loop to target points that are distinct in their explanatory structure.

## Repository Structure

The methodology has been applied and validated on two distinct complex systems, structured in separate directories:

### 1. [Schelling_ABS](./Schelling_ABS/)
Application of the EDS framework to the classic Schelling Segregation Model. This directory contains the surrogate models, the Active Learning optimization loops, and the KDE-based density maximization logic mapped onto a 5-dimensional design space.

### 2. [Cireco](./cireco/)
Application of the EDS framework to the Cireco simulation platform (a market environment simulating buyers and sellers exchanging products). This directory contains both the core Cireco ABM and the advanced Sobol/Surrogate sensitivity analysis scripts.

### Documentation & Reports
- `docs/`: Contains the project presentation (`v0.pdf`), and comprehensive technical evaluation reports detailing the mathematical algorithms and operations research insights.

## Methodology High-Level Overview

1. **Initial LHS Sampling:** Sample the complex system's features using a static Design of Experiments.
2. **Explanatory Latent Space:** Extract feature contribution profiles using `shap.ExactExplainer` on a surrogate model.
3. **Density Target:** Apply Kernel Density Estimation (KDE) with boundary soft-folding and repulsion to map out the density of explanations.
4. **Active Optimization:** Minimize the density using Expected Improvement on a Multi-Fidelity Gaussian Process, driving the search into rare behavioral regimes.
5. **Update:** Evaluate the expensive black-box simulator at the discovered optimal point.

## License
Refer to the individual subdirectories for licensing details (MIT License for Cireco).

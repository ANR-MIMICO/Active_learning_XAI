# 3 Mathematically Rigorous Improvements to the Acquisition Optimization and Surrogate Modeling

Based on the review of `al_xai_optimizer.py` and the context of the Derivative-Free Optimization (DFO) Symposium, here are three mathematically rigorous improvements designed to elevate the active learning methodology. The current implementation relies on a linearly-scalarized acquisition function optimized via L-BFGS-B, and standard multi-fidelity Kriging. The proposed enhancements address critical limitations in these areas.

## 1. Replacing Local Gradient-Based Optimization (L-BFGS-B) with a Global Derivative-Free Optimizer (e.g., CMA-ES)

**Current Limitation:**
The acquisition function combines Expected Improvement (EI) on SHAP novelty and standard deviation (exploration) into a single scalar objective: `obj = alpha * ei_norm + (1 - alpha) * sigma_norm`. This landscape is highly non-convex, multimodal, and oscillatory. Currently, the code optimizes this using `scipy.optimize.minimize` with `L-BFGS-B` from 15 Latin Hypercube Sampling (LHS) starting points. `L-BFGS-B` relies on finite-difference approximations of gradients, which are notoriously unstable on multi-modal surrogate surfaces and prone to getting trapped in local optima, particularly when the Kriging variance approaches zero.

**Mathematical Improvement:**
Replace `L-BFGS-B` with **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** or **DIRECT (DIviding RECTangles)**. CMA-ES is a state-of-the-art DFO algorithm that is invariant to rank-preserving transformations of the objective function and scales well with dimensionality. It adapts a multivariate normal distribution to the topography of the objective function.
*   **Why it elevates the paper:** In a black-box context, proving that the active learning loop strictly bounds the error in finding the true maximum of the acquisition function is crucial. CMA-ES ensures global exploration of the acquisition landscape without relying on finite differences. This strengthens the claim that the selected candidate point is genuinely the global optimal trade-off between SHAP novelty and spatial exploration.

## 2. Resolving Linear Scalarization Limitations via Multi-Objective DFO (Pareto Front Exploration)

**Current Limitation:**
The hybrid objective uses a linear weighting scheme (`alpha`) that decays over iterations. Mathematically, a linear combination (scalarization) of objectives can only identify solutions on the convex hull of the Pareto optimal front. If the true trade-off surface between `ei_norm` (novelty) and `sigma_norm` (exploration) is non-convex, linear scalarization will completely miss optimal points in the concave regions, leading to sub-optimal sampling.

**Mathematical Improvement:**
Formulate the acquisition optimization as a rigorous **Multi-Objective Optimization (MOO)** problem:
$$\max_{x \in \mathcal{X}} \begin{bmatrix} \text{EI}_{MFK}(x) \\ \sigma_{KRG}(x) \end{bmatrix}$$
Instead of a priori scalarization via `alpha`, utilize a Multi-Objective Evolutionary Algorithm (MOEA) such as **NSGA-II** or **MOEA/D**, or a multi-objective acquisition metric like **Expected Hypervolume Improvement (EHVI)**.
*   **Why it elevates the paper:** By finding the Pareto front of the acquisition function, the framework can mathematically guarantee non-dominance in candidate selection. This enables batch active learning (selecting a diverse set of points from the Pareto front) instead of sequentially adding one point per loop. Discussing the topological structure of the Pareto front in the SHAP-vs-Spatial exploration context adds significant theoretical depth suitable for a DFO symposium.

## 3. Trust-Region Multi-Fidelity Kriging (MFK) for Non-Stationary SHAP Density Landscapes

**Current Limitation:**
The `MFK` surrogate is trained on a density matrix derived from soft-folding KDE of SHAP values. Density landscapes of feature attributions are inherently non-stationary (exhibiting sharp peaks and flat plateaus). The current `MFK` setup uses a standard stationary Gaussian kernel (`theta0=[1e-2]`). Stationary kernels assume uniform smoothness across the entire design space, which leads to poor calibration and high predictive variance near sharp density transitions or design boundaries (Runge's phenomenon).

**Mathematical Improvement:**
Integrate a **Trust-Region Bayesian Optimization (e.g., TuRBO)** framework for the MFK, or employ **Non-Stationary/Deep Gaussian Processes**. 
A mathematically sound approach for DFO is to restrict the acquisition optimization to a localized Trust Region $\mathcal{T}_k$ centered around the best explanation-density points. Inside $\mathcal{T}_k$, the stationarity assumption holds locally. The size of the Trust Region is dynamically expanded or contracted based on the success of the surrogate predictions (via the ratio of actual vs. predicted improvement).
*   **Why it elevates the paper:** Trust-region methods provide strong mathematical convergence guarantees for DFO. By limiting the MFK's domain to a local trust region, the active learning loop naturally avoids boundary exploitation issues and accurately models the highly non-linear SHAP density function locally. This presents a robust, theoretically backed solution to handling complex explanation spaces in black-box systems.

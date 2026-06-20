import numpy as np
import shap
import pandas as pd
from scipy.stats import entropy
from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
from smt.sampling_methods import LHS
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings("ignore")

def compute_metrics(X, shap_values, design_space):
    X_norm = np.copy(X)
    for i, var in enumerate(design_space.design_variables):
        b_min, b_max = var.lower, var.upper
        X_norm[:, i] = (X[:, i] - b_min) / (b_max - b_min)

    n_bins = 10
    entropies_x = []
    for i in range(X.shape[1]):
        hist, _ = np.histogram(X_norm[:, i], bins=n_bins, range=(0, 1), density=False)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropies_x.append(entropy(hist))
    entropy_x = (np.mean(entropies_x) / np.log(n_bins))

    if len(X_norm) > 1:
        nn_x = NearestNeighbors(n_neighbors=2).fit(X_norm)
        distances_x, _ = nn_x.kneighbors(X_norm)
        nnd_x = distances_x[:, 1]
        cv_nnd_x = np.std(nnd_x) / np.mean(nnd_x) if np.mean(nnd_x) > 0 else 0
    else:
        cv_nnd_x = 0.0

    shap_norm = np.copy(shap_values)
    entropies_shap = []
    for i in range(shap_values.shape[1]):
        b_min, b_max = np.min(shap_values[:, i]), np.max(shap_values[:, i])
        if b_max > b_min:
            shap_norm[:, i] = (shap_values[:, i] - b_min) / (b_max - b_min)
        else:
            shap_norm[:, i] = 0
        hist, _ = np.histogram(shap_norm[:, i], bins=n_bins, range=(0, 1), density=False)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropies_shap.append(entropy(hist))
    entropy_shap = (np.mean(entropies_shap) / np.log(n_bins))

    if len(shap_norm) > 1:
        nn_shap = NearestNeighbors(n_neighbors=2).fit(shap_norm)
        distances_shap, _ = nn_shap.kneighbors(shap_norm)
        nnd_shap = distances_shap[:, 1]
        cv_nnd_shap = np.std(nnd_shap) / np.mean(nnd_shap) if np.mean(nnd_shap) > 0 else 0
    else:
        cv_nnd_shap = 0.0

    return entropy_x, cv_nnd_x, entropy_shap, cv_nnd_shap

def explicit_ei(x, sm, y_min):
    pred = sm.predict_values(x)
    var = sm.predict_variances(x)
    sigma = np.sqrt(np.maximum(var, 1e-9))
    u = (y_min - pred) / sigma
    ei = (y_min - pred) * norm.cdf(u) + sigma * norm.pdf(u)
    return ei

class ActiveLearningXAI:
    def __init__(self, simulator_func, design_space, x_initial, y_initial, alpha_start=0.0, alpha_end=1.0, total_loops=50, mode='v4'):
        self.simulator_func = simulator_func
        self.design_space = design_space
        self.X = np.array(x_initial)
        self.y = np.array(y_initial)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_loops = total_loops
        self.mode = mode
        self.samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
        
    def run(self, output_dir="data/processed/v4"):
        os.makedirs(output_dir, exist_ok=True)
        metrics_csv_path = os.path.join(output_dir, "al_metrics_history.csv")
        
        with open(metrics_csv_path, 'w') as f:
            f.write("Loop,N_Points,Alpha,Entropy_Input,CV_NND_Input,Entropy_SHAP,CV_NND_SHAP\n")

        for loop in range(self.total_loops):
            print(f"\n--- LOOP {loop + 1}/{self.total_loops} ---")
            
            if self.total_loops > 1:
                alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (loop / (self.total_loops - 1))
            else:
                alpha = self.alpha_end
            print(f"Hybrid Alpha: {alpha:.2f} (Exploration vs Novelty)")

            # 1. Primary Surrogate (Physics Model)
            sm = KRG(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-8, design_space=self.design_space)
            sm.set_training_values(self.X, self.y)
            sm.train()
            
            # 2. Extract SHAP values
            n_added_kde = 50
            x_val = self.samp(n_added_kde)
            x_val_all = np.concatenate((x_val, self.X))
            
            explainer = shap.ExactExplainer(sm.predict_values, self.X)
            shap_values = explainer(x_val_all).values
            
            historical_shap = shap_values[n_added_kde:]
            candidate_shap = shap_values[:n_added_kde]
            
            # Save Metrics
            ent_x, cvnnd_x, ent_shap, cvnnd_shap = compute_metrics(self.X, historical_shap, self.design_space)
            with open(metrics_csv_path, 'a') as f:
                f.write(f"{loop},{len(self.X)},{alpha:.2f},{ent_x:.4f},{cvnnd_x:.4f},{ent_shap:.4f},{cvnnd_shap:.4f}\n")

            # 3. Fast Density Proxy via Cosine Distance (Replaces Slow KDE)
            # Distance in SHAP space represents novelty (High distance = High Novelty = Low Density)
            # We convert distance to a "density proxy" so MFK can minimize it.
            if getattr(self, 'mode', 'v4') == 'v3':
                # --- V3 MODE: 1D Marginal KDE Soft Folding ---
                from scipy.stats import gaussian_kde
                n_features = historical_shap.shape[1]
                density_candidates = np.zeros(candidate_shap.shape[0])
                density_hist = np.zeros(historical_shap.shape[0])
                
                for l in range(n_features):
                    data_l = historical_shap[:, l]
                    b_min, b_max = np.min(data_l), np.max(data_l)
                    # Soft folding
                    dist = b_max - b_min
                    folded_data = np.concatenate([data_l, 2*b_min - data_l, 2*b_max - data_l])
                    
                    try:
                        kde = gaussian_kde(folded_data, bw_method='scott')
                        density_candidates += kde(candidate_shap[:, l])
                        density_hist += kde(historical_shap[:, l])
                    except:
                        pass
                
                density_low_fid = density_candidates / n_features
                density_high_fid = density_hist / n_features
            else:
                # --- V4 MODE: Multivariate Cosine Distance Proxy ---
                nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(historical_shap)
                distances_candidates, _ = nn.kneighbors(candidate_shap)
                density_low_fid = np.exp(-distances_candidates.flatten())  # Proxies density
                
                nn_hist = NearestNeighbors(n_neighbors=2, metric='cosine').fit(historical_shap)
                distances_hist, _ = nn_hist.kneighbors(historical_shap)
                density_high_fid = np.exp(-distances_hist[:, 1])

            y_min_observed = np.min(np.concatenate((density_low_fid, density_high_fid))) + 0.05
            
            # 4. Multi-Fidelity Kriging (MFK) on Novelty
            surro = MFK(theta0=[1e-2], print_global=False, eval_noise=True, nugget=1e-7, design_space=self.design_space)
            surro.set_training_values(x_val, density_low_fid, name=0) # Low fidelity
            surro.set_training_values(self.X, density_high_fid)       # High fidelity
            surro.train()
            
            # 5. Hybrid Acquisition Function with Dynamic Alpha
            x_start = self.samp(15)
            ei_samples = explicit_ei(np.atleast_2d(x_start), surro, y_min_observed)
            var_samples = sm.predict_variances(np.atleast_2d(x_start))
            sigma_samples = np.sqrt(np.maximum(var_samples, 1e-9))
            
            max_ei = np.max(ei_samples) if np.max(ei_samples) > 1e-12 else 1.0
            max_sigma = np.max(sigma_samples) if np.max(sigma_samples) > 1e-12 else 1.0
            
            def objective(x_opt):
                if x_opt.ndim == 1:
                    x_opt = np.atleast_2d(x_opt)
                ei = explicit_ei(x_opt, surro, y_min_observed)
                var = sm.predict_variances(x_opt)
                sigma = np.sqrt(np.maximum(var, 1e-9))
                
                ei_norm = ei / max_ei
                sigma_norm = sigma / max_sigma
                obj = alpha * ei_norm + (1 - alpha) * sigma_norm
                return -obj.flatten()

            bounds = [(v.lower, v.upper) for v in self.design_space.design_variables]
            
            from scipy.optimize import differential_evolution
            
            res = differential_evolution(objective, bounds=bounds, seed=42, popsize=15, maxiter=50)
            best_x = res.x
                    
            # Evaluate new point
            best_x = np.atleast_2d(best_x)
            new_y = self.simulator_func(best_x)
            
            self.X = np.vstack((self.X, best_x))
            self.y = np.append(self.y, new_y)
            
            df_combined = pd.DataFrame(self.X, columns=[f"Var_{i}" for i in range(self.X.shape[1])])
            df_combined["Target"] = self.y.flatten()
            df_combined.to_csv(os.path.join(output_dir, f"al_database_loop_{loop}.csv"), index=False)
            
        return self.X, self.y

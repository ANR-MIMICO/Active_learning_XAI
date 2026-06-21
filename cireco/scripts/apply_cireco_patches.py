import os

scripts_dir = r'C:\Users\psaves\Desktop\Active_learning_XAI\cireco\scripts'

# 1. Fix 1D Histograms bounds
file_1d = os.path.join(scripts_dir, 'plot_1d_histograms.py')
with open(file_1d, 'r', encoding='utf-8') as f: content = f.read()

bins_code = """        if var_name == "Price_Dispose":
            bins_to_use = np.linspace(0, 200, 21)
            xticks = np.linspace(0, 200, 5)
        elif var_name == "Scarcity":
            bins_to_use = np.linspace(-2, 2, 21)
            xticks = np.linspace(-2, 2, 5)
        elif var_name == "Density":
            bins_to_use = np.linspace(0, 0.5, 21)
            xticks = np.linspace(0, 0.5, 6)
        elif var_name == "Cluster":
            bins_to_use = np.linspace(0, 10, 21)
            xticks = np.linspace(0, 10, 6)
        elif var_name == "Km_Cost":
            bins_to_use = np.linspace(-4, -1, 16)
            xticks = np.linspace(-4, -1, 4)"""

old_code = """        min_v = min(np.min(X_lhs), np.min(X_sur), np.min(X_v5), np.min(X_sur_shap))
        max_v = max(np.max(X_lhs), np.max(X_sur), np.max(X_v5), np.max(X_sur_shap))
        bins_to_use = np.linspace(min_v, max_v, 15)
        xticks = None"""

content = content.replace(old_code, bins_code)
with open(file_1d, 'w', encoding='utf-8') as f: f.write(content)

# 2. Fix Hybrid Diff bounds
file_diff = os.path.join(scripts_dir, 'plot_hybrid_diff_histograms.py')
with open(file_diff, 'r', encoding='utf-8') as f: content = f.read()

old_code_diff = """        min_v = min(np.min(X_lhs), np.min(X_v5))
        max_v = max(np.max(X_lhs), np.max(X_v5))
        bins_to_use = np.linspace(min_v, max_v, 15)
        xticks = None"""

content = content.replace(old_code_diff, bins_code)
with open(file_diff, 'w', encoding='utf-8') as f: f.write(content)

# 3. Fix PCA GIF script
file_gif = os.path.join(scripts_dir, 'generate_pca_gif.py')
with open(file_gif, 'r', encoding='utf-8') as f: content = f.read()

content = content.replace('lhs_file = os.path.join(results_dir, f"tmp_lhs_{seed}", "al_database.csv")', 'lhs_file = os.path.join(results_dir, f"LHS_seed_{seed}.csv")')
content = content.replace('mlp, mlp_scaler = prepare_simulator()', 'simulator = prepare_simulator()')
content = content.replace('Z = mlp.predict_proba(mlp_scaler.transform(grid_5d_real))[:, 1]', 'Z = simulator(grid_5d_real)')
content = content.replace('levels=[0.5]', 'levels=[70]')
content = content.replace('levels=20, cmap=\'coolwarm\'', 'levels=20, cmap=\'viridis\'')
content = content.replace('gif_path = os.path.join(results_dir, "..", "..", "figures", "analysis", f"{method.lower()}_evolution_pca.gif")', 'gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "figures", "analysis", f"{method.lower()}_evolution_pca.gif"))')

with open(file_gif, 'w', encoding='utf-8') as f: f.write(content)

print("Applied patches to Histograms and GIF scripts.")

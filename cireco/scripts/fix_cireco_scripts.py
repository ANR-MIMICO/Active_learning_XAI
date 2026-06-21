import os

scripts_dir = r'C:\Users\psaves\Desktop\Active_learning_XAI\cireco\scripts'

# Fix 1: plot_all_metrics.py
file_path = os.path.join(scripts_dir, 'plot_all_metrics.py')
with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
# Make sure to handle empty dfs properly, and skip LHS correctly.
content = content.replace("            if not dfs: continue", "            if not dfs:\n                continue")
content = content.replace("""            for f in csv_files:
                df = pd.read_csv(f)
                if 'N_Points' not in df.columns: continue
                df = df.set_index('N_Points')
                dfs.append(df[metric_col])""", """            for f in csv_files:
                df = pd.read_csv(f)
                if 'N_Points' not in df.columns or metric_col not in df.columns: continue
                df = df.set_index('N_Points')
                dfs.append(df[metric_col])
            if not dfs: continue""")
with open(file_path, 'w', encoding='utf-8') as f: f.write(content)

# Fix 2: plot_dual_space_efficiency.py
file_path = os.path.join(scripts_dir, 'plot_dual_space_efficiency.py')
with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace("df = pd.read_csv(f).set_index('N_Points')", "df = pd.read_csv(f)\n        if 'N_Points' not in df.columns: continue\n        df = df.set_index('N_Points')")
with open(file_path, 'w', encoding='utf-8') as f: f.write(content)

# Fix 3: plot_hybrid_diff_histograms.py
file_path = os.path.join(scripts_dir, 'plot_hybrid_diff_histograms.py')
with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
content = content.replace("path_lhs = os.path.join(results_dir, \"tmp_lhs_42\", \"al_database.csv\")", "path_lhs = os.path.join(results_dir, \"LHS_seed_42.csv\")")
content = content.replace("if not os.path.exists(path_lhs): path_lhs = os.path.join(results_dir, \"tmp_lhs_42\", \"al_database_loop_49.csv\")", "")
with open(file_path, 'w', encoding='utf-8') as f: f.write(content)

# Fix 4: plot_pca_metrics.py and generate_pca_gif.py (Imports)
for fname in ['plot_pca_metrics.py', 'generate_pca_gif.py']:
    file_path = os.path.join(scripts_dir, fname)
    if not os.path.exists(file_path): continue
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    content = content.replace("from cireco.scripts.paper_benchmark import prepare_simulator", "from cireco.scripts.cireco_paper_benchmark import prepare_simulator")
    # Also in plot_pca_metrics and generate_pca_gif, the simulator is used. It expects X to be 5D, and it returns Price.
    # In Schelling it returned Probability. Let's make sure it doesn't crash on "predict_proba".
    content = content.replace("krg = prepare_simulator()", "simulator = prepare_simulator()")
    content = content.replace("krg.predict_values(grid_points)", "simulator(grid_points)")
    content = content.replace("Tipping Point Probability", "Simulated Price")
    # For PCA grid, we should generate points properly
    with open(file_path, 'w', encoding='utf-8') as f: f.write(content)

print("Fixed the Python files!")

import os

src = r'C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\scripts'
dst = r'C:\Users\psaves\Desktop\Active_learning_XAI\cireco\scripts'

files = [
    'plot_1d_histograms.py', 'plot_all_metrics.py', 'plot_boundary_distribution.py',
    'plot_dual_space_efficiency.py', 'plot_hybrid_diff_histograms.py', 'plot_pca_metrics.py',
    'generate_pca_gif.py', 'generate_all_paper_figures.py'
]

os.makedirs(dst, exist_ok=True)

for f in files:
    src_file = os.path.join(src, f)
    if not os.path.exists(src_file): continue
    with open(src_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Base replacements
    content = content.replace('Schelling_ABS', 'cireco')
    content = content.replace('paper_results_2', 'paper_results_2')
    
    # 2. Path correction: force output to cireco\data\figures\analysis
    # In Schelling, out_img was usually:
    # out_img = os.path.join(results_dir, "..", "..", "figures", "analysis", ...)
    # If results_dir is cireco\data\processed\paper_results
    # ..\..\figures\analysis -> cireco\data\figures\analysis
    # Which perfectly aligns with the user's request! (C:\Users\psaves\Desktop\Active_learning_XAI\cireco\data\figures)
    
    # 3. Variable replacements
    content = content.replace('["Radius", "Ratio", "Tolerance", "Grid Size", "N Groups"]', '["Price_Dispose", "Scarcity", "Density", "Cluster", "Km_Cost"]')
    content = content.replace('"Radius"', '"Price_Dispose"').replace('"Ratio"', '"Scarcity"').replace('"Tolerance"', '"Density"').replace('"Grid Size"', '"Cluster"').replace('"N Groups"', '"Km_Cost"')
    
    # 4. Fix specific files
    if 'boundary' in f:
        content = content.replace('Probability of Convergence (Simulated Output)', 'Simulated Price (Target)')
        content = content.replace('Tipping Point Boundary (P=0.5)', 'Price Threshold')
        content = content.replace('plt.axvline(0.5', 'plt.axvline(70')
        content = content.replace('plt.xlim(0, 1)', '')
        
        # We need to skip the MLP loading and just use df['Target']
        lines = content.split('\n')
        new_lines = []
        skip_mlp = False
        for line in lines:
            if 'def get_simulator()' in line:
                skip_mlp = True
            if skip_mlp and 'return mlp, scaler' in line:
                skip_mlp = False
                continue
            if skip_mlp:
                continue
            
            if 'mlp, scaler = get_simulator()' in line: continue
            
            if 'X_scaled = scaler.transform(X_added)' in line: continue
            
            if 'probs = mlp.predict_proba(X_scaled)[:, 1]' in line:
                new_lines.append('                    probs = df["Target"].values[30:]')
                continue
            
            new_lines.append(line)
        content = '\n'.join(new_lines)
        
    if 'plot_all_metrics' in f:
        content = content.replace("df = pd.read_csv(f).set_index('N_Points')", 
                                "df = pd.read_csv(f)\n                if 'N_Points' not in df.columns: continue\n                df = df.set_index('N_Points')")
                                
    if '1d_histograms' in f or 'hybrid_diff_histograms' in f:
        # Erase custom bins
        old_bins_1d = """        if var_name == "Price_Dispose":
            bins_to_use = np.arange(1.5, 6.5, 1)
            xticks = [2, 3, 4, 5]
        elif var_name == "Km_Cost":
            bins_to_use = np.arange(0.5, 11.5, 1)
            xticks = range(1, 11)
        elif var_name == "Cluster":
            bins_to_use = np.linspace(10, 40, 11)
            xticks = None
        else:
            bins_to_use = np.linspace(0, 1, 11)
            xticks = None"""
            
        new_bins_1d = """        min_v = min(np.min(X_lhs), np.min(X_sur), np.min(X_v5), np.min(X_sur_shap))
        max_v = max(np.max(X_lhs), np.max(X_sur), np.max(X_v5), np.max(X_sur_shap))
        bins_to_use = np.linspace(min_v, max_v, 15)
        xticks = None"""
        
        new_bins_diff = """        min_v = min(np.min(X_lhs), np.min(X_v5))
        max_v = max(np.max(X_lhs), np.max(X_v5))
        bins_to_use = np.linspace(min_v, max_v, 15)
        xticks = None"""
        
        if '1d' in f: content = content.replace(old_bins_1d, new_bins_1d)
        if 'diff' in f: content = content.replace(old_bins_1d, new_bins_diff)

    with open(os.path.join(dst, f), 'w', encoding='utf-8') as file:
        file.write(content)

print("Restoration script executed!")

import os
import shutil

src = r'C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\scripts'
dst = r'C:\Users\psaves\Desktop\Active_learning_XAI\cireco\scripts'
files = ['plot_all_metrics.py', 'plot_1d_histograms.py', 'plot_hybrid_diff_histograms.py']

os.makedirs(dst, exist_ok=True)

for f in files:
    src_file = os.path.join(src, f)
    dst_file = os.path.join(dst, f)
    if not os.path.exists(src_file):
        continue
    with open(src_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Adapt to Cireco context
    content = content.replace('Schelling_ABS', 'cireco')
    content = content.replace('paper_results_2', 'paper_results_2')
    
    # Change Variable Names
    content = content.replace('["Radius", "Ratio", "Tolerance", "Grid Size", "N Groups"]', '["Price_Dispose", "Scarcity", "Density", "Cluster_Spread", "Km_Cost"]')
    content = content.replace('"Radius"', '"Price_Dispose"').replace('"Ratio"', '"Scarcity"').replace('"Tolerance"', '"Density"').replace('"Grid Size"', '"Cluster_Spread"').replace('"N Groups"', '"Km_Cost"')
    
    # Remove custom Schelling bins logic and use simple linspace for continuous variables
    if f == 'plot_1d_histograms.py':
        old_bins = """        # Custom bins for each variable
        if var_name == "Price_Dispose":
            bins_to_use = np.arange(1.5, 6.5, 1)
            xticks = [2, 3, 4, 5]
        elif var_name == "Km_Cost":
            bins_to_use = np.arange(0.5, 11.5, 1)
            xticks = range(1, 11)
        elif var_name == "Cluster_Spread":
            bins_to_use = np.linspace(10, 40, 11)
            xticks = None
        else:
            bins_to_use = np.linspace(0, 1, 11)
            xticks = None"""
            
        new_bins = """        # Continuous bounds for Cireco
        bins_to_use = np.linspace(np.min(X_lhs), np.max(X_lhs), 15)
        xticks = None"""
        content = content.replace(old_bins, new_bins)

    if f == 'plot_hybrid_diff_histograms.py':
        old_bins_diff = """        # Custom bins for each variable
        if var_name == "Price_Dispose":
            bins_to_use = np.arange(1.5, 6.5, 1)
            xticks = [2, 3, 4, 5]
        elif var_name == "Km_Cost":
            bins_to_use = np.arange(0.5, 11.5, 1)
            xticks = range(1, 11)
        elif var_name == "Cluster_Spread":
            bins_to_use = np.linspace(10, 40, 11)
            xticks = None
        else:
            bins_to_use = np.linspace(0, 1, 11)
            xticks = None"""
            
        new_bins_diff = """        # Continuous bounds for Cireco
        min_v = min(np.min(X_lhs), np.min(X_v5))
        max_v = max(np.max(X_lhs), np.max(X_v5))
        bins_to_use = np.linspace(min_v, max_v, 15)
        xticks = None"""
        content = content.replace(old_bins_diff, new_bins_diff)
        # Fix v4 -> v5
        content = content.replace('tmp_v4_', 'tmp_v5_')
        content = content.replace('X_v4', 'X_v5')
        content = content.replace('df_v4', 'df_v5')
        content = content.replace('hist_v4', 'hist_v5')
        content = content.replace('path_v4', 'path_v5')
        
    with open(dst_file, 'w', encoding='utf-8') as file:
        file.write(content)

print('Copied plotting scripts to cireco/scripts!')

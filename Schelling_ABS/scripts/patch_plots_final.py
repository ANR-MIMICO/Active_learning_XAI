import os

scripts_dir = r"C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\scripts"

# 1. plot_all_metrics.py
path = os.path.join(scripts_dir, "plot_all_metrics.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('methods = ["LHS", "SUR", "SUR_SHAP", "V5", "V6_SUR", "V6_DYN"]', 'methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]')
content = content.replace('colors = {"LHS": "black", "SUR": "blue", "SUR_SHAP": "green", "V5": "red", "V6_SUR": "orange", "V6_DYN": "purple"}', 'colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}')
content = content.replace('''    labels = {
        "LHS": "LHS", 
        "SUR": "Space-US", 
        "SUR_SHAP": "SHAP-US",
        "V5": "Dynamic-US",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "IMSE-Hybrid"
    }''', '''    labels = {
        "LHS": "LHS", 
        "SUR_SHAP": "SHAP-CS",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "Dynamic-US"
    }''')
with open(path, "w", encoding="utf-8") as f: f.write(content)

# 2. plot_dual_space_efficiency.py
path = os.path.join(scripts_dir, "plot_dual_space_efficiency.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('methods = ["LHS", "SUR", "SUR_SHAP", "V5", "V6_SUR", "V6_DYN"]', 'methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]')
content = content.replace('colors = {"LHS": "black", "SUR": "blue", "SUR_SHAP": "green", "V5": "red", "V6_SUR": "orange", "V6_DYN": "purple"}', 'colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}')
content = content.replace('''    labels = {
        "LHS": "LHS", 
        "SUR": "Space-US", 
        "SUR_SHAP": "SHAP-US",
        "V5": "Dynamic-US",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "IMSE-Hybrid"
    }''', '''    labels = {
        "LHS": "LHS", 
        "SUR_SHAP": "SHAP-CS",
        "V6_SUR": "IMSE-US",
        "V6_DYN": "Dynamic-US"
    }''')
content = content.replace('cmap_dict = {"LHS": "Greys", "SUR": "Blues", "SUR_SHAP": "Greens", "V5": "Reds", "V6_SUR": "Oranges", "V6_DYN": "Purples"}', 'cmap_dict = {"LHS": "Greys", "SUR_SHAP": "Greens", "V6_SUR": "Blues", "V6_DYN": "Reds"}')
with open(path, "w", encoding="utf-8") as f: f.write(content)

# 3. plot_boundary_distribution.py
path = os.path.join(scripts_dir, "plot_boundary_distribution.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('methods = ["LHS", "SUR", "V5", "SUR_SHAP", "V6_SUR", "V6_DYN"]', 'methods = ["LHS", "SUR_SHAP", "V6_SUR", "V6_DYN"]')
content = content.replace('colors = {"LHS": "black", "SUR": "blue", "V5": "red", "SUR_SHAP": "green", "V6_SUR": "orange", "V6_DYN": "purple"}', 'colors = {"LHS": "black", "SUR_SHAP": "green", "V6_SUR": "blue", "V6_DYN": "red"}')
content = content.replace('labels = {"LHS": "LHS", "SUR": "Space-US", "V5": "Dynamic-US", "SUR_SHAP": "SHAP-US", "V6_SUR": "IMSE-US", "V6_DYN": "IMSE-Hybrid"}', 'labels = {"LHS": "LHS", "SUR_SHAP": "SHAP-CS", "V6_SUR": "IMSE-US", "V6_DYN": "Dynamic-US"}')
with open(path, "w", encoding="utf-8") as f: f.write(content)

# 4. plot_pca_metrics.py
path = os.path.join(scripts_dir, "plot_pca_metrics.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('subplots(1, 6, figsize=(36, 6))', 'subplots(1, 4, figsize=(24, 6))')
content = content.replace('''    methods = [("lhs", "LHS"), 
               ("sur", "Space-US"), 
               ("sur_shap", "SHAP-US"), 
               ("v5", "Dynamic-US (V5)"),
               ("v6_sur", "IMSE-US (V6)"),
               ("v6_dyn", "IMSE-Hybrid (V6)")]''', '''    methods = [("lhs", "LHS"), 
               ("sur_shap", "SHAP-CS"), 
               ("v6_sur", "IMSE-US"),
               ("v6_dyn", "Dynamic-US")]''')
content = content.replace("seed_colors = ['magenta', 'cyan', 'lime', 'orange', 'purple']", "seed_colors = ['magenta', 'cyan', 'lime', 'orange', 'purple']")
with open(path, "w", encoding="utf-8") as f: f.write(content)

# 5. plot_1d_histograms.py
path = os.path.join(scripts_dir, "plot_1d_histograms.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace('methods = ["lhs", "sur", "sur_shap", "v5", "v6_sur", "v6_dyn"]', 'methods = ["lhs", "sur_shap", "v6_sur", "v6_dyn"]')
content = content.replace('titles = {"lhs": "LHS", "sur": "Space-US", "sur_shap": "SHAP-US", "v5": "Dynamic-US", "v6_sur": "IMSE-US", "v6_dyn": "IMSE-Hybrid"}', 'titles = {"lhs": "LHS", "sur_shap": "SHAP-CS", "v6_sur": "IMSE-US", "v6_dyn": "Dynamic-US"}')
content = content.replace('colors = {"lhs": "black", "sur": "blue", "sur_shap": "green", "v5": "red", "v6_sur": "orange", "v6_dyn": "purple"}', 'colors = {"lhs": "black", "sur_shap": "green", "v6_sur": "blue", "v6_dyn": "red"}')
with open(path, "w", encoding="utf-8") as f: f.write(content)

# 6. generate_pca_gif.py
path = os.path.join(scripts_dir, "generate_pca_gif.py")
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("for method in ['lhs', 'sur', 'sur_shap', 'v5', 'v6_sur', 'v6_dyn']:", "for method in ['lhs', 'sur_shap', 'v6_sur', 'v6_dyn']:")
content = content.replace('''        if method == 'lhs': m_title = "LHS"
        elif method == 'sur': m_title = "Space-US"
        elif method == 'sur_shap': m_title = "SHAP-US"
        elif method == 'v5': m_title = "Dynamic-US"
        elif method == 'v6_sur': m_title = "IMSE-US"
        elif method == 'v6_dyn': m_title = "IMSE-Hybrid"
        color = 'magenta' if method == 'lhs' else 'cyan' if method == 'sur' else 'lime' if method == 'sur_shap' else 'red' if method == 'v5' else 'orange' if method == 'v6_sur' else 'purple'
''', '''        if method == 'lhs': m_title = "LHS"
        elif method == 'sur_shap': m_title = "SHAP-CS"
        elif method == 'v6_sur': m_title = "IMSE-US"
        elif method == 'v6_dyn': m_title = "Dynamic-US"
        color = 'magenta' if method == 'lhs' else 'lime' if method == 'sur_shap' else 'cyan' if method == 'v6_sur' else 'red'
''')
with open(path, "w", encoding="utf-8") as f: f.write(content)

print("Final patch complete!")

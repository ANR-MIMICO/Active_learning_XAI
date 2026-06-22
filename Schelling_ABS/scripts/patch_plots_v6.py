import os
import glob

scripts_dir = r"C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\scripts"
files = glob.glob(os.path.join(scripts_dir, "plot_*.py")) + glob.glob(os.path.join(scripts_dir, "generate_pca*.py"))

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
        
    # Standard lists
    content = content.replace('["LHS", "SUR", "SUR_SHAP", "V5"]', '["LHS", "SUR", "SUR_SHAP", "V5", "V6_SUR", "V6_DYN"]')
    content = content.replace('["lhs", "sur", "sur_shap", "v5"]', '["lhs", "sur", "sur_shap", "v5", "v6_sur", "v6_dyn"]')
    content = content.replace("['lhs', 'sur', 'sur_shap', 'v5']", "['lhs', 'sur', 'sur_shap', 'v5', 'v6_sur', 'v6_dyn']")
    
    # Dicts
    content = content.replace('"V5": "red"}', '"V5": "red", "V6_SUR": "orange", "V6_DYN": "purple"}')
    content = content.replace('"V5": "Reds"}', '"V5": "Reds", "V6_SUR": "Oranges", "V6_DYN": "Purples"}')
    
    content = content.replace('"V5": "Dynamic-US"\n    }', '"V5": "Dynamic-US",\n        "V6_SUR": "IMSE-US",\n        "V6_DYN": "IMSE-Hybrid"\n    }')
    
    # Tuples for pca_metrics
    content = content.replace('("v5", "Dynamic-US")]', '("v5", "Dynamic-US"), \n               ("v6_sur", "IMSE-US"), \n               ("v6_dyn", "IMSE-Hybrid")]')

    # Color mapping for PCA metrics (if they exist)
    # The PCA metrics plot has fig, axes = plt.subplots(1, 4, figsize=(24, 6)) -> Needs to be 1, 6
    content = content.replace('subplots(1, 4, figsize=(24, 6))', 'subplots(1, 6, figsize=(36, 6))')

    with open(f, "w", encoding="utf-8") as file:
        file.write(content)

print("Patched successfully!")

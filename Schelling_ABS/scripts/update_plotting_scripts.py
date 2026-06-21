import glob
import os

scripts_dir = r"C:\Users\psaves\Desktop\Active_learning_XAI\Schelling_ABS\scripts"
files_to_update = glob.glob(os.path.join(scripts_dir, "plot_*.py")) + glob.glob(os.path.join(scripts_dir, "generate_pca*.py"))

for fpath in files_to_update:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace the results directory
    content = content.replace('"paper_results"', '"paper_results_2"')
    
    # Replace V4 with V5 in dictionaries and lists
    content = content.replace('"V4"', '"V5"')
    content = content.replace("'V4'", "'V5'")
    content = content.replace("tmp_v4_", "tmp_v5_")
    
    # Replace Hybrid-US with Dynamic-US if needed for V5 context
    content = content.replace('"Hybrid-US"', '"Dynamic-US"')
    content = content.replace("'Hybrid-US'", "'Dynamic-US'")
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Updated {len(files_to_update)} scripts for results_2 and V5.")

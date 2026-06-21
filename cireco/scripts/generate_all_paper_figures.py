import subprocess
import os

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
    else:
        print(f"Success: {result.stdout.strip()}")

if __name__ == "__main__":
    print("Generating all camera-ready figures for the paper...")
    
    scripts = [
        "plot_all_metrics.py",            # 1. 2x2 Grid (Entropy & CV NND for Input/SHAP)
        "plot_dual_space_efficiency.py",  # 2. Dual Space Scatter Plot
        "plot_boundary_distribution.py",  # 3. Tipping Point Distance KDE
        "plot_pca_metrics.py",            # 4. Static PCA 1x4 Grid
        "plot_1d_histograms.py",          # 5. 1D Histograms per Variable
        "plot_hybrid_diff_histograms.py", # 5.5 Hybrid Diff Histograms
        "generate_pca_gif.py"             # 6. PCA Real-Time Evolution GIFs
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\nALL POST-PROCESSING COMPLETE! Check the figures/analysis folder.")

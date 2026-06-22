import os, sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cireco.scripts.cireco_paper_benchmark import prepare_simulator

def generate_pca_gif_for_method(results_dir, method, seed=42):
    folder = os.path.join(results_dir, f"tmp_{method}_{seed}")
    if not os.path.exists(folder): return
        
    lhs_file = os.path.join(results_dir, f"tmp_lhs_{seed}", "al_database_loop_0.csv")
    if not os.path.exists(lhs_file): return
        
    df_lhs = pd.read_csv(lhs_file)
    X_lhs = df_lhs.iloc[:, :5].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lhs)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # --- GET DYNAMIC BOUNDARIES FROM FINAL LOOP ---
    final_loop = 50
    final_csv = os.path.join(folder, f"al_database_loop_{final_loop}.csv")
    if os.path.exists(final_csv):
        df_final = pd.read_csv(final_csv)
        X_final_proj = pca.transform(scaler.transform(df_final.iloc[:, :5].values))
        x_min, x_max = X_final_proj[:, 0].min() - 1, X_final_proj[:, 0].max() + 1
        y_min, y_max = X_final_proj[:, 1].min() - 1, X_final_proj[:, 1].max() + 1
    else:
        x_min, x_max = -3, 4
        y_min, y_max = -3, 3

    # --- GET TRUE SIMULATOR FOR BACKGROUND CONTOUR ---
    simulator = prepare_simulator()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_5d_scaled = pca.inverse_transform(grid_2d)
    grid_5d_real = scaler.inverse_transform(grid_5d_scaled)
    Z = simulator(grid_5d_real)
    Z = Z.reshape(xx.shape)
    
    frames = []
    tmp_dir = os.path.join(folder, "gif_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    
    titles = {'lhs': 'LHS', 'sur': 'Space-US', 'sur_shap': 'SHAP-US', 'v5': 'Dynamic-US'}
    method_title = titles.get(method.lower(), method.upper())
    color_map = {'lhs': 'black', 'sur': 'blue', 'sur_shap': 'green', 'v5': 'red'}
    main_color = color_map.get(method.lower(), 'red')

    for loop in range(51):
        csv_file = os.path.join(folder, f"al_database_loop_{loop}.csv")
        if not os.path.exists(csv_file): continue
            
        df = pd.read_csv(csv_file)
        X_all = df.iloc[:, :5].values
        X_proj = pca.transform(scaler.transform(X_all))
        
        plt.figure(figsize=(10, 8))
        plt.title(f"PCA Projection - {method_title} Evolution (Loop {loop})", fontsize=16, fontweight='bold')
        
        # Plot background contour
        plt.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.4, zorder=1)
        plt.contour(xx, yy, Z, levels=[70], colors='black', linewidths=2, linestyles='--', zorder=2)
        
        plt.scatter(X_proj[:30, 0], X_proj[:30, 1], c='grey', alpha=0.6, label='Initial DoE (30 pts)', s=50, edgecolors='white', zorder=10)
        
        if len(X_proj) > 30:
            plt.scatter(X_proj[30:, 0], X_proj[30:, 1], c=main_color, alpha=0.9, 
                        label=f'Active Learning ({len(X_proj)-30} pts)', s=100, edgecolors='black', zorder=11)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(loc="upper right")
        
        frame_path = os.path.join(tmp_dir, f"frame_{loop:03d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100)
        plt.close()
        frames.append(frame_path)
        
    if frames:
        gif_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "figures", "analysis", f"{method.lower()}_evolution_pca.gif"))
        images = [Image.open(f) for f in frames]
        for _ in range(10): images.append(images[-1])
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
        print(f"Saved GIF: {gif_path}")
        
    for f in frames: os.remove(f)
    try: os.rmdir(tmp_dir)
    except: pass

def generate_all_gifs():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "paper_results"))
    for method in ['lhs', 'sur', 'sur_shap', 'v5']:
        generate_pca_gif_for_method(results_dir, method, seed=42)

if __name__ == "__main__":
    generate_all_gifs()

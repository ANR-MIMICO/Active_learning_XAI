import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_entropy_curves(csv_path, output_image_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Normalized Entropy
    axes[0].plot(df['N_Points'], df['Entropy_Input'], label='Input Space Entropy', color='blue', marker='o')
    axes[0].plot(df['N_Points'], df['Entropy_SHAP'], label='SHAP Space Entropy', color='orange', marker='s')
    axes[0].set_title('Evolution of Entropy vs Iterations', fontweight='bold')
    axes[0].set_xlabel('Number of Points in DoE')
    axes[0].set_ylabel('Normalized Shannon Entropy (Higher is Better)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Right subplot: CV NND
    axes[1].plot(df['N_Points'], df['CV_NND_Input'], label='Input Space CV NND', color='blue', marker='o')
    axes[1].plot(df['N_Points'], df['CV_NND_SHAP'], label='SHAP Space CV NND', color='orange', marker='s')
    axes[1].set_title('Evolution of Uniformity (CV NND)', fontweight='bold')
    axes[1].set_xlabel('Number of Points in DoE')
    axes[1].set_ylabel('Coefficient of Variation NND (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Curves successfully plotted and saved to: {output_image_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Entropy Curves from AL V4 Metrics")
    parser.add_argument("--app", type=str, choices=["cireco", "schelling"], required=True, help="Which application to plot")
    args = parser.parse_args()

    app_dir = "cireco" if args.app == "cireco" else "Schelling_ABS"
    csv_file = os.path.join(app_dir, "data", "processed", "v4_al_results", "al_metrics_history.csv")
    out_img = os.path.join(app_dir, "figures", "analysis", "entropy_evolution.png")
    
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plot_entropy_curves(csv_file, out_img)

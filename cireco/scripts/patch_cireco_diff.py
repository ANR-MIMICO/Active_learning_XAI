import os
import re
path = r"C:\Users\psaves\Desktop\Active_learning_XAI\cireco\scripts\plot_hybrid_diff_histograms.py"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f: content = f.read()
    content = content.replace("tmp_v4_", "tmp_v6_dyn_")
    content = content.replace("tmp_v5_", "tmp_v6_dyn_")
    content = content.replace("df_v4", "df_v6")
    content = content.replace("X_v4", "X_v6")
    content = content.replace("hist_v4", "hist_v6")
    content = content.replace("Hybrid-US", "Dynamic-US")
    content = content.replace("IMSE-Hybrid", "Dynamic-US")
    with open(path, "w", encoding="utf-8") as f: f.write(content)
print("Diff patched!")

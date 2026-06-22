import os
path = r'cireco\scripts\plot_1d_histograms.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('titles = {"lhs": "LHS", "sur": "Space-US", "sur_shap": "SHAP-US", "v5": "Dynamic-US"}', 'titles = {"lhs": "LHS", "sur_shap": "SHAP-CS", "v6_sur": "IMSE-US", "v6_dyn": "Dynamic-US"}')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("1D histograms patched!")

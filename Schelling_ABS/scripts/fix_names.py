import glob
import os

scripts = glob.glob('Schelling_ABS/scripts/plot_*.py') + glob.glob('Schelling_ABS/scripts/generate_*.py')

for f in scripts:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # We want the methods array to be exactly: ["LHS", "SUR_SHAP", "V6_SUR", "V6_Dynamic"]
    # So we replace occurrences of V5 and V6_DYN with V6_Dynamic.
    content = content.replace('"V5"', '"V6_Dynamic"')
    content = content.replace('"V6_DYN"', '"V6_Dynamic"')
    content = content.replace("'V5'", "'V6_Dynamic'")
    content = content.replace("'V6_DYN'", "'V6_Dynamic'")
    content = content.replace('V6_DYN_seed', 'V6_Dynamic_seed')
    content = content.replace('V5_seed', 'V6_Dynamic_seed')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print("Plot scripts patched!")

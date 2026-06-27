import glob
import os

scripts = glob.glob('cireco/scripts/plot_*.py') + glob.glob('cireco/scripts/generate_*.py')

for f in scripts:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Adapt to the V6_Dynamic naming convention
    content = content.replace('"V5"', '"V6_Dynamic"')
    content = content.replace('"V6_DYN"', '"V6_Dynamic"')
    content = content.replace("'V5'", "'V6_Dynamic'")
    content = content.replace("'V6_DYN'", "'V6_Dynamic'")
    content = content.replace('V6_DYN_seed', 'V6_Dynamic_seed')
    content = content.replace('V5_seed', 'V6_Dynamic_seed')
    content = content.replace('v6_dynamic', 'v6_dyn') # Fallback for folders if lowercased
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print("Cireco plot scripts successfully patched to V6_Dynamic!")

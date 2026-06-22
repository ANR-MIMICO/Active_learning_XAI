import os, glob
scripts = glob.glob(r'cireco\scripts\*.py')
for s in scripts:
    with open(s, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'paper_results' in content and 'cireco_paper_benchmark.py' not in s:
        content = content.replace('"paper_results"', '"paper_results_2"')
        content = content.replace("'paper_results'", "'paper_results_2'")
        with open(s, 'w', encoding='utf-8') as f:
            f.write(content)
print("Directory paths updated!")

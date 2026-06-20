import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess

def run_benchmarks():
    print("--- STARTING BENCHMARK: 10 LOOPS ---")
    
    # 1. V4 (Cosine + Pareto DE + MFK + Alpha 0->1)
    print("\nRunning V4 (New Framework)...")
    subprocess.run(["python", "al_v4_schelling.py"])
    
    # 2. V3 (KDE + Soft Folding + L-BFGS-B + Alpha 0->1)
    print("\nRunning V3 (Old KDE Framework)...")
    subprocess.run(["python", "al_tp_v3.py"])
    
    print("\n--- BENCHMARK FINISHED ---")
    print("Check the generated CSVs in data/processed/ and plot them.")

if __name__ == '__main__':
    run_benchmarks()

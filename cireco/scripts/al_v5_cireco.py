import sys, os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from smt.design_space import DesignSpace, FloatVariable
from smt.sampling_methods import LHS

# Add the unified src directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.al_xai_optimizer import ActiveLearningXAI

def main():
    print("==================================================")
    print("   V5 Active Learning Framework : CIRECO (Dynamic) ")
    print("==================================================")
    
    # 1. Load Initial Database
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    csv_file = os.path.join(folder_path, 'Newbounds.csv')
    
    if not os.path.exists(csv_file):
        print(f"Error: Database not found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    df["density"] = np.log10(df["density"])
    df["scarcity"] = np.log2(df["scarcity"])
    
    DATABASEx = np.array(df)[:,:5]
    DATABASEy = np.array(df)[:,5] # TARGET IS PRICE
    
    # 2. Train Surrogate Simulator (MLP Proxy)
    print("Training surrogate MLP on real data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(DATABASEx)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_scaled, DATABASEy)
    
    def cireco_simulator(x):
        """Proxy simulator evaluating points on the trained MLP"""
        x_scaled = scaler.transform(x)
        return mlp.predict(x_scaled)
        
    # 3. Define Design Space (Corrected Column Order)
    design_space = DesignSpace([
        FloatVariable(0, 200),  # 0. price_to_dispose
        FloatVariable(-2, 2),   # 1. scarcity (log2)
        FloatVariable(-4, -1),  # 2. density (log10)
        FloatVariable(0, 0.5),  # 3. cluster_spread
        FloatVariable(0, 10),   # 4. km_cost
    ])
    
    # 4. Generate starting DoE
    print("Generating Initial LHS Design of Experiments (30 points)...")
    np.random.seed(42)
    samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
    x_initial = samp(30)
    y_initial = cireco_simulator(x_initial)
    
    # 5. Launch Active Learning Framework V5
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "v5_al_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # V5 MAGIC: Dynamic Annealing from 0.0 (Pure Explanatory) to 1.0 (Pure Variance)
    print("Launching Active Learning Loop (V5 - Dynamic Alpha)...")
    al_framework = ActiveLearningXAI(
        simulator_func=cireco_simulator,
        design_space=design_space,
        x_initial=x_initial,
        y_initial=y_initial,
        alpha_start=0.0,
        alpha_end=1.0,
        total_loops=50,
        mode='v4' # Use Cosine Distance
    )
    
    X_final, y_final = al_framework.run(output_dir=output_dir)
    print(f"\\nDone! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

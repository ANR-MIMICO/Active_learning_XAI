import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from smt.design_space import DesignSpace, FloatVariable
from smt.sampling_methods import LHS

# Import our new V4 Framework!
from src.al_xai_optimizer import ActiveLearningXAI

def main():
    print("--- V4 Active Learning Framework : CIRECO ---")
    
    # 1. Load Initial Database
    # We use the existing LHS DoE or Sobol sequence from Cireco
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    csv_file = os.path.join(folder_path, 'Newbounds.csv')
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    # Transformations from Cireco
    df["density"] = np.log10(df["density"])
    df["scarcity"] = np.log2(df["scarcity"])
    
    DATABASEx = np.array(df)[:,:5]
    DATABASEy = np.array(df)[:,5]
    
    # 2. Train Surrogate Simulator (MLP)
    # Since Cireco is slow, we use an MLP trained on 100 points as the "true simulator"
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
    
    # Wrapper function for the simulator
    def cireco_simulator(x):
        # Scale input, predict, and return
        x_scaled = scaler.transform(x)
        return mlp.predict(x_scaled)
        
    # 3. Define Design Space
    design_space = DesignSpace([
        FloatVariable(0, 200),
        FloatVariable(-2, 2), 
        FloatVariable(0, 0.5),
        FloatVariable(0, 10),
        FloatVariable(-4, -1),
    ])
    
    # 4. Generate starting DoE for Active Learning
    # We start with 30 points from LHS
    samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
    x_initial = samp(30)
    y_initial = cireco_simulator(x_initial)
    
    # 5. Launch Active Learning Framework
    al_framework = ActiveLearningXAI(
        simulator_func=cireco_simulator,
        design_space=design_space,
        x_initial=x_initial,
        y_initial=y_initial,
        total_loops=50  # 50 iterations
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "v4_al_results")
    print("Launching Active Learning Loop...")
    X_final, y_final = al_framework.run(output_dir=output_dir)
    print("Done! Data and Entropy Curves saved in:", output_dir)

if __name__ == "__main__":
    main()

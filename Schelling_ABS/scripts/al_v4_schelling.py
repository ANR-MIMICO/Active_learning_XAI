import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable
from smt.sampling_methods import LHS

# Import our new V4 Framework!
from src.al_xai_optimizer import ActiveLearningXAI

def main():
    print("--- V4 Active Learning Framework : SCHELLING ABS ---")
    
    # 1. Load Initial Database from doe_5_200
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'doe_5_200')
    DATABASEx = []
    DATABASEy = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                last_line = file.readlines()[-1]
                DATABASEx.append(np.array(filename[:-11].split('_'), dtype=np.float64))
                DATABASEy.append(np.array([
                    bool(last_line.split(',')[0].lower() == "true"), 
                    float(last_line.split(',')[1])
                ]))
                
    DATABASEx = np.array(DATABASEx)
    DATABASEy = np.array(DATABASEy)
    
    # 2. Train Surrogate Simulator (MLP Classifier & Regressor)
    # Schelling predicts a boolean (is segregated?) and a float. 
    # For AL SHAP, we usually target the float or the probability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(DATABASEx)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    # Train on the boolean target (or use the regressor for the float)
    y_target = DATABASEy[:, 0].astype(int)
    mlp.fit(X_scaled, y_target)
    
    # Wrapper function for the simulator
    def schelling_simulator(x):
        # We return the probability of being class 1 for continuous SHAP analysis
        x_scaled = scaler.transform(x)
        return mlp.predict_proba(x_scaled)[:, 1]
        
    # 3. Define Design Space
    design_space = DesignSpace([
        IntegerVariable(2, 5),
        FloatVariable(0.01, 1.0),
        FloatVariable(0.0, 1.0),
        IntegerVariable(10, 40),
        IntegerVariable(1, 10),
    ])
    
    # 4. Generate starting DoE for Active Learning
    # We start with 30 points from LHS
    samp = LHS(xlimits=design_space.get_num_bounds(), criterion="ese")
    x_initial = samp(30)
    
    # Floor integer variables
    for i, var in enumerate(design_space.design_variables):
        if isinstance(var, IntegerVariable):
            x_initial[:, i] = np.round(x_initial[:, i])
            
    y_initial = schelling_simulator(x_initial)
    
    # 5. Launch Active Learning Framework
    al_framework = ActiveLearningXAI(
        simulator_func=schelling_simulator,
        design_space=design_space,
        x_initial=x_initial,
        y_initial=y_initial,
        total_loops=10  # 10 iterations for testing
    )
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "v4_al_results")
    print("Launching Active Learning Loop...")
    X_final, y_final = al_framework.run(output_dir=output_dir)
    print("Done! Data and Entropy Curves saved in:", output_dir)

if __name__ == "__main__":
    main()

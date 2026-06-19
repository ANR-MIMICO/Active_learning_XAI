# smt_manual_pmml_nyoka.py
"""
Train an SMT LS surrogate and export it to PMML using Nyoka's core API.
This bypasses skl_to_pmml and nyoka.pmml import issues by using the
exact Nyoka PMML44 module path.

Requirements:
    pip install smt scikit-learn nyoka numpy
"""

import numpy as np
from smt.surrogate_models import LS
from sklearn.base import BaseEstimator, RegressorMixin

# Nyoka core PMML 4.4 API
from nyoka.PMML44 import PMML, Header, DataDictionary, DataField, MiningSchema, MiningField, RegressionModel, RegressionTable
from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
from sklearn.gaussian_process.kernels import RBF

if __name__ == "__main__":
    # 1) Define a scikit-learn wrapper for SMT LS
    class SMTSurrogateLS(BaseEstimator, RegressorMixin):
        def __init__(self,
                     print_global=True, print_training=True,
                     print_prediction=True, print_problem=True,
                     print_solver=True, data_dir=None):
            self.print_global     = print_global
            self.print_training   = print_training
            self.print_prediction = print_prediction
            self.print_problem    = print_problem
            self.print_solver     = print_solver
            self.data_dir         = data_dir
            self.theta = 0
        def fit(self, X, y):
            X_arr = np.atleast_2d(X)
            y_arr = np.atleast_1d(y)
            self.sm_ = LS(
                print_global     = self.print_global,
                print_training   = self.print_training,
                print_prediction = self.print_prediction,
                print_problem    = self.print_problem,
                print_solver     = self.print_solver,
            )
            self.sm_.set_training_values(X_arr, y_arr)
            self.sm_.train()
            return self

        def predict(self, X):
            X_arr = np.atleast_2d(X)
            return self.sm_.predict_values(X_arr).ravel()

    # 2) Generate synthetic data and train
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.sin(X[:, 0]) + (X[:, 1]**2) + 0.1 * np.random.randn(100)

    model = SMTSurrogateLS().fit(X, y)

    # 3) Define feature/target names
    feature_names = ["x1", "x2"]
    target_name   = "y"

    # 4) Build the DataDictionary
    data_fields = [
        DataField(name=fn, dataType="double", optype="continuous")
        for fn in feature_names
    ] + [
        DataField(name=target_name, dataType="double", optype="continuous")
    ]
    data_dict = DataDictionary(
        numberOfFields=len(data_fields),
        DataField=data_fields
    )

    # 5) Build the MiningSchema
    mining_fields = [
        MiningField(name=fn, usageType="active")
        for fn in feature_names
    ] + [
        MiningField(name=target_name, usageType="predicted")
    ]
    mining_schema = MiningSchema(MiningField=mining_fields)

    # 6) Extract SMT LS coefficients & intercept
    #    SMT LS stores weights in `theta`, intercept in `beta`
    coeffs    = model.sm_.theta          # array of length 2
    intercept = float(getattr(model.sm_, "beta", 0.0))

    # 7) Build the RegressionTable
    regression_table = RegressionTable(
        intercept=intercept,
        Parameter=[
            # name / value pairs for each feature
            pmml_param
            for pmml_param in [
                {"name": feature_names[i], "value": float(coeffs[i])}
                for i in range(len(coeffs))
            ]
        ]
    )

    # Wrap parameters into actual PMML objects
    # RegressionTable expects a list of nyoka.PMML44.pmml44.Parameter
    # but the constructor’s introspection will handle dict->object conversion.

    # 8) Build the RegressionModel
    reg_model = RegressionModel(
        modelName="SMT_LS_Surrogate",
        functionName="regression",
        algorithmName="SMT Least Squares",
        MiningSchema=mining_schema,
        Output=Output(
            OutputField=[
                OutputField(
                    name=f"predicted_{target_name}",
                    feature="predictedValue"
                )
            ]
        ),
        RegressionTable=[regression_table]
    )

    # 9) Assemble the PMML document
    pmml = PMML(
        version=PMML.VERSION.value,
        Header=Header(
            description="SMT LS surrogate exported via Nyoka core API",
            Timestamp=Timestamp(),
            Application=Application(name="Nyoka", version="latest")
        ),
        DataDictionary=data_dict,
        RegressionModel=[reg_model]
    )

    # 10) Write out XML file
    with open("smt_model.pmml", "w") as f:
        f.write(pmml.export(indent=2))

    print("Successfully wrote PMML to 'smt_model.pmml'")

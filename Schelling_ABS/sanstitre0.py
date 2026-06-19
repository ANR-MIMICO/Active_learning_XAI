# smt_nyoka_export.py
"""
Example script showing how to wrap an SMT LS surrogate model in a scikit-learn API and export it to PMML using Nyoka.
Requirements:
    pip install smt scikit-learn nyoka numpy
"""
import numpy as np
from smt.surrogate_models import LS
from sklearn.base import BaseEstimator, RegressorMixin
from nyoka import skl_to_pmml
from sklearn.linear_model import LinearRegression as _LR
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
from sklearn.gaussian_process.kernels import RBF

class GaussianProcessRegressor(_GPR):
    """
    scikit-learn wrapper for SMT Least Squares surrogate (LS) model.
    """
    def __init__(
        self,
        print_global=True,
        print_training=True,
        print_prediction=True,
        print_problem=True,
        print_solver=True,
        data_dir=None,
    ):
        super().__init__()         # Initialize base LinearRegression
        self.print_global = print_global
        self.print_training = print_training
        self.print_prediction = print_prediction
        self.print_problem = print_problem
        self.print_solver = print_solver
        self.data_dir = data_dir

    def fit(self, X, y):
        X_arr = np.atleast_2d(X)
        y_arr = np.atleast_1d(y)
        self.sm_ = LS(
            print_global=self.print_global,
            print_training=self.print_training,
            print_prediction=self.print_prediction,
            print_problem=self.print_problem,
            print_solver=self.print_solver,
        )
        self.coef_      = [0,0]
        self.intercept_ = 0
        self.n_features_in_ = 0
        
        self.sm_.set_training_values(X_arr, y_arr)
        self.sm_.train()
        return self

    def predict(self, X):
        X_arr = np.atleast_2d(X)
        return self.sm_.predict_values(X_arr).ravel()


if __name__ == "__main__":
    # 1) Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = np.sin(X[:, 0]) + X[:, 1]**2 + np.random.randn(100) * 0.1

    # 2) Instantiate and wrap in an sklearn Pipeline
    smt_model = GaussianProcessRegressor()
    pipeline = Pipeline([
        ("smt_ls", smt_model)
    ])  # Nyoka's exporter expects a Pipeline instance :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

    # 3) Fit pipeline
    pipeline.fit(X, y)

    # 4) Define feature/target names
    feature_names = ["x1", "x2"]
    target_name   = "y"

    # 5) Export to PMML
    skl_to_pmml(
        pipeline=pipeline,
        col_names=feature_names,
        target_name=target_name,
        pmml_f_name="smt_model.pmml"
    )  # writes smt_model.pmml :contentReference[oaicite:3]{index=3}

    print("Successfully exported PMML to 'smt_model.pmml'")
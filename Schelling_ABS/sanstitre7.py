import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KernelDensity
from mapie.regression import MapieRegressor
from smt.surrogate_models import KRG
# 1) Définir un wrapper KRGRegressor pour qu’il offre fit(X, y) et predict(X)

class KRGRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper scikit-learn pour utiliser le Kriging (KRG) de SMT comme régressseur.
    - fit(X, y) entraîne le modèle KRG sur (X, y).
    - predict(X) renvoie la prédiction de la moyenne du Kriging pour chaque point.
    """
    def __init__(self, theta0=[0.1], noise0=[0.001], corr='matern52', print_global=True, n_start=5):
        """
        Paramètres (tous optionnels, à adapter selon vos besoins) :
        
        - theta0   : initial guess pour les hyperparamètres de corrélation (taille d'array = d)
        - thetaL   : borne inférieure pour l'optimisation de theta (array de taille d)
        - thetaU   : borne supérieure pour l'optimisation de theta (array de taille d)
        - corr     : type de corrélation ("squared_exponential", "matern32", "matern52", etc.)
        - print_global : bool, si True affiche les logs SMT durant l'entraînement
        - n_start  : nombre de ré-optimisations du maximum de vraisemblance (pour éviter un optimum local)
        """
        self.theta0 = theta0
        self.corr = corr
        self.print_global = print_global
        self.n_start = n_start

        # On crée l’instance SMT KRG mais on l'entraînera dans fit()
        self.model_ = KRG()

    def fit(self, X, y):
        """
        Entraîne le Kriging SMT sur (X, y).
        X : array de forme (n_samples, n_features)
        y : array de forme (n_samples,) ou (n_samples, 1)
        """
        # SMT s'attend à y de forme (n_samples, 1), on reshape si nécessaire
        y = np.atleast_2d(y).reshape(-1, 1)

        # Instanciation du modèle KRG avec les hyperparamètres passés
        self.model_ = KRG(theta0=self.theta0,
                          corr=self.corr,
                          print_global=self.print_global,
                          n_start=self.n_start)
        # On ajuste KRG sur X, y
        self.model_.set_training_values(X, y)
        self.model_.train()

        return self

    def predict(self, X):
        """
        Prédit la moyenne du Kriging pour chaque point de X.
        X : array de forme (n_samples, n_features)
        Retourne un array 1D de taille n_samples (scikit-learn attend un vecteur 1D).
        """
        # SMT KRG renvoie shape (n_samples, n_output) ; ici n_output = 1
        y_pred = self.model_.predict_values(X)  # shape = (n_samples, 1)
        y_pred = y_pred*[y_pred>0]
        return y_pred.flatten()                 # on renvoie un vecteur 1D

    def get_params(self, deep=True):
        """
        Nécessaire pour être totalement compatible avec BaseEstimator.
        """
        return {
            "theta0": self.theta0,
            "corr": self.corr,
            "print_global": self.print_global,
            "n_start": self.n_start,
        }

    def set_params(self, **params):
        """
        Permet de modifier à la volée les hyperparamètres (compatible GridSearchCV, etc.).
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    
# 2) Génération d’un nuage de points 2D pour l’exemple
np.random.seed(0)
n = 120
mu1, cov1 = [0,0], [[1, 0.3],[0.3,1]]
mu2, cov2 = [3,3], [[0.8, -0.2],[-0.2, 0.8]]
X1 = np.random.multivariate_normal(mu1, cov1, n//2)
X2 = np.random.multivariate_normal(mu2, cov2, n//2)
X = np.vstack([X1, X2])  # taille (300, 2)

# 3) Calcul préliminaire de la KDE « véritable » sur X pour en extraire la densité de chaque point
kde_true = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(X)
log_dens_train = kde_true.score_samples(X)
dens_train = np.exp(log_dens_train)  # y_train = densité KDE(X)

# 4) Instanciation du régressseur KDE + MAPIE
regressor = KRGRegressor()
mapie = MapieRegressor(regressor, cv="split", method="plus")

# 5) On ajuste MAPIE sur (X, dens_train)
mapie.fit(X, dens_train)

# 6) Pour visualiser : on crée une grille 2D et on appelle mapie.predict
import matplotlib.pyplot as plt

xg = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
yg = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
xx, yy = np.meshgrid(xg, yg)
grid = np.vstack([xx.ravel(), yy.ravel()]).T  # shape (10000, 2)

#  α = 0.05 pour l’intervalle de confiance 95%
y_pred, y_pis = mapie.predict(grid, alpha=0.1)
lower, upper = y_pis[:, 0], y_pis[:, 1]

# 7) Reformater pour l’affichage
zz_true = np.exp(kde_true.score_samples(grid)).reshape(xx.shape)  # densité « véritable »
zz_pred = y_pred.reshape(xx.shape)                                # densité prédite par KDERegressor
zz_lower = lower.reshape(xx.shape)
zz_upper = upper.reshape(xx.shape)
ci_width = zz_upper - zz_lower

# 8) Tracer : vraie KDE vs prédiction vs largeur de CI
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# 8.1) Vraie densité KDE
cf0 = axes[0].contourf(xx, yy, zz_true, cmap='viridis')
axes[0].scatter(X[:,0], X[:,1], s=5, c='white')
axes[0].set_title("Densité KDE")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(cf0, ax=axes[0])

# 8.2) Densité prédite par KDERegressor + MAPIE
cf1 = axes[1].contourf(xx, yy, zz_pred, cmap='viridis')
axes[1].scatter(X[:,0], X[:,1], s=5, c='white')
axes[1].set_title("Prédiction GP SMT + MAPIE")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
plt.colorbar(cf1, ax=axes[1])

plt.tight_layout()
plt.show()

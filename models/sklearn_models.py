# models/sklearn_models.py
from sklearn.ensemble import (
    AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
    AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge 
from sklearn.linear_model import (
    Ridge, LogisticRegression, ElasticNet, Lasso, BayesianRidge,
    SGDRegressor, SGDClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Aliases for consistency
XGBoostRegressor = xgb.XGBRegressor
XGBoostClassifier = xgb.XGBClassifier
GBDTRegressor = GradientBoostingRegressor  # GBDT alias
GBDTClassifier = GradientBoostingClassifier
GPRegressor = GaussianProcessRegressor  # GPR alias
GPClassifier = GaussianProcessClassifier
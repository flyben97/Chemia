# train_valid_test/models/sklearn_models.py
from sklearn.ensemble import (
    AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
    AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge 
from sklearn.linear_model import Ridge, LogisticRegression 
from sklearn.svm import SVR, SVC
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

XGBoostRegressor = xgb.XGBRegressor
XGBoostClassifier = xgb.XGBClassifier
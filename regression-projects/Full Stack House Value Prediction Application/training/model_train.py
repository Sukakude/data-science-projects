import xgboost as xgb
import optuna
import numpy as np
import sklearn
import time
from sklearn.metrics import mean_absolute_percentage_error

import os
os.environ['LC_ALL'] = 'en_US.UTF-8'

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    This function is responsible for determining the best parameters for the XGBoost model

    Args:
        X_train (np.ndarray): These are the features that will be used as part of the training dataset.
        X_test (np.ndarray): These are the features that will be used to make predictions using the trained model.
        y_train (np.array): These are the labels that will be used as part of the training dataset.
        y_test (np.array): These are the features that will be used to make predictions using the trained model.

    Returns:
        dict: Returns the best parameters for the XGBoost model
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    def objective(trial):

        param = {
            "objective": "reg:squarederror",
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "tree_method": "hist",
            "n_jobs": 1,
            "verbosity": 0,
        }

        n_estimators = trial.suggest_int("n_estimators", 100, 500)

        model = xgb.train(
            param,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtest, "eval")],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        preds = model.predict(dtest)
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        return mape

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150, timeout=600)

    print()
    print(f'Best RSME: {study.best_value}')
    print(f'Best params: {study.best_params}\n')

    return study.best_params

def model_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ):

    try:
        print('************ Starting Model Train Step ************\n')
        start_time = time.time()

        best_params = hyperparameter_tuning(X_train, X_test, y_train, y_test)
        model = xgb.XGBRegressor(params=best_params)
        model.fit(X_train, y_train)

        # DISPLAY PERFORMANCE METRICS
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Total time taken: {round(total_time, 4)}s')
        print('------------------------------------\n')

        return model
    except Exception as e:
        print(f'Error in model_training: {e}')
        raise e

import logging 
from zenml import step
import pandas as pd

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow # type: ignore
from zenml.client import Client # type: ignore

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker='mlflow_tracker')
def train_model(
       X_train: pd.DataFrame,
       X_test: pd.DataFrame,
       y_train: pd.Series,
       y_test: pd.Series
    ) -> RegressorMixin:
    """
    Description:
        - Trains the model
        
    Parameters:
        -
    
    Returns:
        - 
    """
    try:
        model = None
        model_config = ModelNameConfig('LinearRegression')
        
        if model_config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog() # logs model scores
            model = LinearRegressionModel()
            trained_model = model.train(X_train=X_train, y_train=y_train)
            return trained_model
        else:
            raise ValueError('Model {} not supported'.format(model_config.model_name))
    except Exception as e:
        logging.info('Error in training model: {}'.format(e))
        raise e

    
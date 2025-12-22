import logging
import pandas as pd

from zenml import step
from zenml.client import Client
import mlflow

from sklearn.base import RegressorMixin
from src.evaluation import MSE, RSME, R2
from typing_extensions import Annotated
from typing import Tuple

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker='mlflow_tracker')
def evaluate_model(
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame        
) -> Tuple[
    Annotated[float, 'r2_score'],
    Annotated[float, 'rsme'],
    Annotated[float, 'mse']
]:
    try:
        y_pred = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.evaluate(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric('mse', mse)
        
        rsme_class = RSME()
        rsme = rsme_class.evaluate(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric('rsme', rsme)
        
        r2_class = R2()
        r2_score = r2_class.evaluate(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric('r2', r2_score)
        
        logging.info('Evaluation complete!')
        return r2_score, rsme, mse
    
    except Exception as e:
        logging.info('Error while evaluation: {}'.format(e))
        raise e        
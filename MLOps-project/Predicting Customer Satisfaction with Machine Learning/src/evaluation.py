import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class AEvaluation(ABC):
    """ ABSTRACT CLASS FOR ALL OTHER EVALUATION CLASSES """
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(AEvaluation):
    """ THIS IS AN EVALUATION STRATEGY THAT USES THE MEAN SQUARED ERROR. """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE...')
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info('MSE: {}'.format(mse))
            return mse
        except Exception as e:
            logging.info('Error in calculating MSE: {}'.format(e))
            raise e

class R2(AEvaluation):
    """ THIS IS AN EVALUATION STRATEGY THAT USES THE R2 SCORE. """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating r2_score...')
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info('R2 score: {}'.format(r2))
            return r2
        except Exception as e:
            logging.info('Error in calculating R2 Score: {}'.format(e))
            raise e

class RSME(AEvaluation):
    """ THIS IS AN EVALUATION STRATEGY THAT USES THE ROOT SQUARED MEAN ERROR (RSME). """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating RSME...')
            rsme = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info('RSME: {}'.format(rsme))
            return rsme
        except Exception as e:
            logging.info('Error in calculating RSME: {}'.format(e))
            raise e        

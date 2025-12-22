from abc import ABC, abstractmethod
import logging

from sklearn.linear_model import LinearRegression

class AModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Description:
            - Trains the model
        
        Parameters:
            - X_train: Training data
            - y_train: Training labels
        
        Returns:
            - None
        
        """
        pass
        
class LinearRegressionModel(AModel):
    def train(self, X_train, y_train, **kwargs):
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            logging.info('Model training completed')
            return lr
        except Exception as e:
            logging.info('Error while training model: {}'.format(e))
            raise e
        
# IMPLEMENT MORE MODELS
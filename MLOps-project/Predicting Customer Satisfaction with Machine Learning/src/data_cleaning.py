import logging
from zenml import step
import pandas as pd
from typing import Union
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """ ABSTRACT CLASS THAT DEFINES THE STRATEGIES FOR HANDLING DATA """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass 
    
class DataPreprocessStrategy(DataStrategy):
    """ PREPROCESSING CLASS """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # REMOVE UNNECESSARY COLUMNS
            df = df.drop([
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date',
                'order_purchase_timestamp'
            ], axis=1)
            
            # FILL MISSING VALUES
            df['product_weight_g'].fillna(df['product_weight_g'].median(), inplace=True)
            df['product_length_cm'].fillna(df['product_length_cm'].median(), inplace=True)
            df['product_height_cm'].fillna(df['product_height_cm'].median(), inplace=True)
            df['product_width_cm'].fillna(df['product_width_cm'].median(), inplace=True)
            df['review_comment_message'].fillna('No review', inplace=True)
            
            # REMOVE CATEGORICAL COLUMNS FOR SIMPLICITY
            df = df.select_dtypes(include=[np.number])
            
            cols_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            df.drop(cols_to_drop, axis=1)
            
            return df
        except Exception as e:
            logging.info('Error in preprocessing data: {}'.format(e))
            raise e
        

class DataPartitionStrategy(DataStrategy):
    """ STRATEGY FOR PARTITIONING THE DATA INTO 80% TRAIN AND 20% TEST SETS. """
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop(['review_score'], axis=1) # REMOVE THE TARGET VARIABLE
            y = df['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.info('Error while partioning the data: {}'.format(e))
            raise e
        
class DataCleaning:
    """ CLASS FOR PREPROCESSING THE DATA AND PARTITIONING THE DATA. """
    def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
        self.df = df
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.info('Error in handling data: {}'.format(e))
            raise e
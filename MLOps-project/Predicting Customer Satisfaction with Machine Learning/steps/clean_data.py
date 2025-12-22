import logging
from zenml import step
import pandas as pd

from src.data_cleaning import DataCleaning, DataPartitionStrategy, DataPreprocessStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """ 
    Description
        - This function is responsible for preprocessing the data then splitting into train/test sets.
    
    Parameters:
        - df: This is the raw data
    
    Returns:
        - X_train: Training features.
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels 
    """
    try:
        # INSTANTIATE THE DATA PREPROCESS OBJECT
        process_strategy = DataPreprocessStrategy()
        
        # INSTANTIATE THE DATA CLEANING OBJECT
        data_cleaning = DataCleaning(df, process_strategy)
        
        # INVOKE THE METHOD TO PREPROCESS THE DATA
        processed_df = data_cleaning.handle_data()
        
        # INSTANTIATE THE DATA PARTITIONING OBJECT
        partition_strategy = DataPartitionStrategy()
        
        # INSTANTIATE THE DATA CLEANING OBJECT
        data_cleaning = DataCleaning(processed_df, partition_strategy)
        
        # INVOKE THE METHOD TO PARTITION THE DATA
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        # SUCCESS MESSAGE
        logging.info('Data cleaning completed')
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.info('Error in data cleaning: {}'.format(e))
        raise e

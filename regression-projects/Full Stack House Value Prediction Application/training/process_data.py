import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import joblib
import time

def encode_data(
    data: Union[pd.DataFrame | list],
    ordinal_cols: Optional[list]
    ) -> Union[np.ndarray | pd.DataFrame]:
    """
    This function is responsible for converting categorical data into numeric data.

    Parameters:
        - series: This is the array of data to be converted into numerical data.

    Returns:
        - 
    """
    categories_order = [['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']]
    # INITIALIZE THE ONEHOT ENCODER OBJECT
    encoder = OrdinalEncoder(categories=categories_order, handle_unknown="use_encoded_value", unknown_value=-1)

    if (isinstance(data, pd.DataFrame)) & (ordinal_cols is not None):
        for ordinal_col in ordinal_cols:
            data[ordinal_col] = encoder.fit_transform(data[[ordinal_col]])
        
        joblib.dump(encoder, "encoder.pkl")
        return data
    else:
        return encoder.fit_transform(data)

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    This function is responsible for normalizing the data into a common range (usually between 0 and 1).

    Parameters:
        - data
    """
    # INITIALIZE THE MINMAX SCALER OBJECT
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # SAVE THE SCALER OBJECT TO USE LATER
    joblib.dump(scaler, "scaler.pkl")

    return X_train, X_test

def preprocess_data(df: pd.DataFrame):
    """
    This function is responsible for perform preprocessing tasks such as handling missing values.

    Parameters:
        - df: This is the data to be processed.

    Returns:
        - processed_df: This is the processed data.
    """
    if df is not None:
        try:
            print('************ Starting Preprocessing Step ************\n')
            start_time = time.time()

            # REMOVE THE EMPTY ROWS FROM OUR DATASET
            processed_df = df.dropna()

            # CONVERT CATEGORICAL DATA INTO NUMERICAL DATA
            processed_df = encode_data(processed_df, ordinal_cols=['ocean_proximity'])

            # DISPLAY PERFORMANCE METRICS
            end_time = time.time()
            total_time = end_time - start_time
            print(f'Total time taken: {round(total_time, 4)}s')
            print('------------------------------------\n')

            return processed_df
        except Exception as e:
            print(f'Error in process_data: {e}')
            raise e
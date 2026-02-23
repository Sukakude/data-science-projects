import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import time

def partition_data(
    df: pd.DataFrame,
    target: object, 
    test_size: float = 0.2
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray
    ]:
    try:
        print('************ Starting Data Partitioning Step ************\n')
        start_time = time.time()

        X = df.drop([target], axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # DISPLAY PERFORMANCE METRICS
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Total time taken: {round(total_time, 4)}s')
        print('------------------------------------\n')
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
            print(f'Error in split_data: {e}')
            raise e
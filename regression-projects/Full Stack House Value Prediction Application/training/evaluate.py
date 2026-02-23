import numpy as np
from sklearn.metrics import mean_squared_error, mean_percentage_error


def evaluate_model(y_test: np.ndarray, y_pred: ndarray) -> None:
    """
    This function is responsible for evaluating the predictions of the model.

    Parameters:
        - y_test
        - y_pred
    
    Returns:
        - None
    """
    try:
        mape = mean_percentage_error(y_test=y_test, y_pred=y_pred) * 100
        rsme = np.sqrt(mean_squared_error(y_test=y_test, y_pred=y_pred))

        print(f'MAPE: {round(mape, 2)}%')
        print(f'RSME: {round(rsme, 2)}')
    except Exception as e:
        print(f'Error in process_data: {e}')
        raise e
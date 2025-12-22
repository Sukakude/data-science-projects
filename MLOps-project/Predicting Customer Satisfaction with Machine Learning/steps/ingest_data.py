import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Description: 
        - This class is responsible for defining all the operations for ingesting the data. 
    """
    def __init__(self, path: str):
        """
        Description:
            - This method serves as the constructor for the class
        
        Parameters:
            - self: Current instance of an object
            - path: Path for the dataset 
        """
        self.path = path
    
    def get_data(self):
        """
        Description: 
            - This function is responsible for ingesting the data from a specific file path. 
        """
        logging.info(f'Ingesting data from {self.path}')
        return pd.read_csv(self.path)
    
@step
def ingest_df(path: str) -> pd.DataFrame:
    """ 
    Description: 
        - Ingesting data from a specific path
    
    Parameters:
        - path: path to the dataset
        
    Returns:
        - pd.DataFrame
    
    """
    try:
        ingest_data = IngestData(path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting the data: {e}')
        raise e
    
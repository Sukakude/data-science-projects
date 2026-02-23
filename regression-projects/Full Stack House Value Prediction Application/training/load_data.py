import pandas as pd
import matplotlib.pyplot as plt

def import_data(filepath: object):
    """
    This function is responsible for loading the data from a specific filepath.

    Parameters:
        - filepath
    
    Returns:
        - Dataframe containing the data
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f'Error in import_data: {e}')
        raise e

# *********************** UNCOMMENT LINE BELOW ONLY WHEN DEBUGGING ***********************
"""
if __name__ == "__main__":
    df = import_data('/home/tshep/repos/house-prediction-app/data/housing.csv')
    grp_df = df.groupby('ocean_proximity')['median_house_value'].sum()
    grp_df.plot(kind='bar')
    plt.show()
"""
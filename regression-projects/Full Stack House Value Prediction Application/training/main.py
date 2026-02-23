from load_data import import_data
from process_data import preprocess_data, scale_data
from split_data import partition_data
from model_train import model_training
import joblib
import pathlib
from datetime import datetime

if __name__ == "__main__":
    # CHECK IF THE MODEL HAS BEEN TRAINED BEFORE
    if not pathlib.Path('model.pkl').exists():
        # FILE PATH FOR OUR THE DATA
        path = './data/housing.csv'

        # IMPORT THE DATA
        df = import_data(path)

        # PREPROCESSING
        processed_df = preprocess_data(df)
        
        # SPLIT INTO TRAIN AND TEST SETS
        X_train, X_test, y_train, y_test = partition_data(processed_df, target='median_house_value')
        
        # NORMALISE THE DATA
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test) 
        
        # TRAIN THE MODEL
        fitted_model = model_training(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # SAVE THE MODEL
        joblib.dump(fitted_model, 'model.pkl')
        print(f'[LOGS] {datetime.now()}: Model saved!')
    else:
        print(f'[LOGS] {datetime.now()}: Loading Model')

        fitted_model = joblib.load('model.pkl') # load the trained model
        
        print(f'[LOGS] {datetime.now()}: Model Loaded!')
        print(f'[LOGS] {datetime.now()}: Ready to start making predictions')

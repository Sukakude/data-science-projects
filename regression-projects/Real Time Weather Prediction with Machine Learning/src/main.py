import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# API KEY
API_KEY = "<YOUR_API_KEY>" # REPLACE WITH YOUR OWN API KEY 

# BASE_URL
BASE_URL = "https://api.openweathermap.org/data/2.5/"

def fetch_weather_data(city):
    """
    This function is responsible for fetching the weather data for a particular city.
    
    Parameters:
        - city: City name
    
    Returns:
        - Dictionary object containing the required information
    """
    
    try:
        url = f"{BASE_URL}weather?q={city}&appId={API_KEY}&units=metric"
    
        # GET REQUEST
        response = requests.get(url, verify=False)
        data = response.json()
        return {
            'city': data['name'],
            'current_temperature': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'wind_gust_speed': data['wind']['speed'], 
            'clouds': data['clouds']['all'],
            'visibility': data['visibility'],
        }
    except Exception as e:
        print('Error in fetch_weather_data: ', e)
    
def read_historical_data(filename):
    """
    This function is responsible for loading the data.
    
    Parameters:
        - filename: Name of the historical weather data file
    
    Returns:
        -  DataFrame that contains the historical weather data
    """
    
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
         print('Error in read_historical_data: ', e)

def data_preprocessing():
    """
    This function is responsible for performing basic data preprocessing steps, and transforming the data into a usable format.
    
    Parameters:
        - df: DataFrame that contains the historical weather data
        
    Returns:
        - X: Feature variable
        - y: Target variable
        - encoder: LabelEncoder object
    """
    try:
        filename = r'..\dataset\weather.csv'
        df = read_historical_data(filename=filename)
        
        # REMOVE EMPTY ROWS, DUPLICATES
        df = df.dropna()
        df.drop_duplicates()
        
        # ENCODE CATEGORICAL DATA
        encoder = LabelEncoder()
        df['WindGustDir'] = encoder.fit_transform(df['WindGustDir'])
        df['RainTomorrow'] = encoder.fit_transform(df['RainTomorrow'])
        
        # SEPARATE THE FEATURE AND TARGET VARIABLE
        X = df.drop(['RainTomorrow'], axis=1)
        y = df['RainTomorrow']
        
        return df, X, y, encoder
    except Exception as e:
        print('Error in data_processing: ', e)
        
def classifier_model_development(X, y):
    """
    This function is responsible for training and evaluating a RandomForestClassifier.
    
    Parameters:
        - X: Feature variables
        - y: Target variables
        
    Returns:
        - rf: Fitted RandomForestClassifier model
    """
    
    try:
        # TRAIN TEST SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # MODEL TRAINING
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # MODEL EVALUATION
        # print('Model Accuracy: {}'.format(
        #         round(accuracy_score(y_test, y_pred), 2)
        #     )
        # )  
        
        return rf
    except Exception as e:
        print('Error in classifier_model_development: ', e)
        
def prepare_regression_data(df, feature):
    """
    This function is responsible for preparing the data for training a RandomForestRegressor.
    
    Parameters:
        - df: Preprocessed DataFrame bject
        - feature: Column name to be used for regression
        
    Returns:
        - X: 2D array containing the features
        - y: 1D array containing the target
    """
    try:
        X, y = [], []
        for i in range(len(df) - 1):
            X.append(df[feature].iloc[i])
            y.append(df[feature].iloc[i+1])
        
        X = np.array(X).reshape(-1,1)
        y = np.array(y)
        return X, y
    except Exception as e:
        print('Error in prepare_regression_data: ', e)
        
def train_regression_model(X, y):
    """
    This function is responsible for training a RandomForestRegressor.
    
    Parameters:
        - X: 2D array containing the features
        - y: 1D array containing the target
        
    Returns:
        - rf: Trained RandomForestRegressor model
    """
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        return rf
    except Exception as e:
        print('Error in train_regression_model: ', e)

def predict(rf, current_value):
    """
    This function is responsible for predicting future values
    
    Parameters:
        - rf: Trained RandomForestRegressor model
        - current_value: Most recent data point for a feature (e.g humidity, temperature etc)
    
    Returns:
        - predictions: 1D array containing the predictions
    """
    try:
        predictions = [current_value] # initial value for making predictions
    
        for i in range(5):
            next_value = rf.predict(np.array([[predictions[-1]]]))[0]
            
            predictions.append(next_value)
        
        return predictions[1:]
    except Exception as e:
        print('Error in predict: ', e)


def weather_view():
    """
    This function is responsible for invoking the above functions.
    
    Parameters:
        - city: Name of the city
    
    Returns:
        - None
    """
    is_valid_city = False  
    
    # PROMPT THE USER FOR INPUT
    city = input('Enter city name: ')
    while not is_valid_city:
        if city == "":
            city = input('Please Enter A Valid City Name: ')
            is_valid_city = False
        else:
            is_valid_city = True
    
    # GET THE CURRENT LIVE DATA
    current_weather = fetch_weather_data(city=city)
    
    # DATA PREPROCESSING
    historical_data, X, y, encoder = data_preprocessing()
    rain_model = classifier_model_development(X=X, y=y) # trains a classifier
    
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    compass_direction = next(point for point, start, end in compass_points if start < wind_deg < end)
    
    compass_direction_encoded = encoder.transform([compass_direction])[0] if compass_direction in encoder.classes_ else -1
    
    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temperature']
    }
    
    current_df = pd.DataFrame([current_data])
    
    # PREDICT WHETHER IT WILL RAIN BASED ON CURRENT DATA
    rain_prediction = rain_model.predict(current_df)[0]
    
    # FORECAST TEMPERATURE AND HUMIDITY
    X_temp, y_temp = prepare_regression_data(historical_data, feature='Temp')
    X_humidity, y_humidity = prepare_regression_data(historical_data, feature='Humidity')
    
    temp_model = train_regression_model(X_temp, y_temp)
    humidity_model = train_regression_model(X_humidity, y_humidity)
    
    # PREDICT FUTURE HUMIDITY AND TEMPERATURE
    temp_pred = predict(temp_model, current_weather['temp_min'])
    humidity_pred = predict(humidity_model, current_weather['humidity'])
    
    # PREPARE TIMESTAMPS FOR PREDICTIONS
    timezone = pytz.timezone('Africa/Johannesburg')
    time_now = datetime.now(timezone)
    next_hour = time_now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
        
    # SHOW RESULTS
    print(f"City: {city}, {current_weather['country']}")
    print(f"Current Temperature: {current_weather['current_temperature']}°C")
    print(f"Minimum Temperature : {current_weather['temp_min']}°C")
    print(f"Maximum Temperature : {current_weather['temp_max']}°C")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Weather Prediction: {current_weather['description']}")
    print(f"Possible Rain? : {'Yes' if rain_prediction else 'No'}")
    
    print("\nFuture Temperature Predictions")
    for time, temp in zip(future_times, temp_pred):
        print(f"{time}: {round(temp, 0)}°C")
        
    print("\nFuture Humidity Predictions")
    for time, humidity in zip(future_times, humidity_pred):
        print(f"{time}: {round(humidity, 0)}%")
    
    return temp_pred, humidity_pred, current_weather, future_times
    
if __name__ == "__main__":
    weather_view()
    

    
    
    

    

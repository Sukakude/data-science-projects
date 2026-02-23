import streamlit as st
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time


load_dotenv()
NOMINATIM_URL = os.getenv("NOMINATIM_URL")

def get_location_coordinates(location: str, max_retries=3):
    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "house-price-predictor (gracedemus@outlook.com)"
    }

    try:
        for attempt in range(max_retries):
            response = requests.get(
                url=f"{NOMINATIM_URL}",
                params=params,
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            if data is not None:
                return data
            time.sleep(5)
        return None
    
    except requests.exceptions.Timeout:
        st.warning("The request timed out. Try again later.")
        return None

    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to the geocoding service. Check your internet.")
        return None

    except requests.exceptions.HTTPError as e:
        st.warning(f"HTTP error: {e}")
        return None

    except Exception as e:
        st.warning(f"An unexpected error occurred: {e}")
        return None

def load_cities():
    df = pd.read_excel('california_demographics_by_city.xlsx')
    df = df.loc[df['name'].str.contains('city')]
    df['name'] = df['name'].str.replace('city', '')
    df['name'] = df['name'].str.strip()
    df.drop(columns=['state', 'city'], axis=1, inplace=True)

    return df

def get_city_names():
    df = load_cities()

    return df.iloc[1:]['name']

def get_city_population(city_name: str):
    df = load_cities()

    return df.loc[df['name']==city_name]['population'].values[0]

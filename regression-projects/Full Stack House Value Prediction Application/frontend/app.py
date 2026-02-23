import streamlit as st
import requests
import os
from dotenv import load_dotenv
import time
from datetime import datetime

from utils import get_location_coordinates, get_city_population, get_city_names

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")
NOMINATIM_URL = os.getenv("NOMINATIM_URL")

def model_info():
    try:
        response = requests.get(f'{BACKEND_URL}/')
        data = response.json()
        return data
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

def predict(
        latitude, longitude, 
        housing_median_age, 
        total_rooms, total_bedrooms, 
        population, households, 
        median_income, ocean_proximity,
        max_retries=3
    ):
    payload = {
        "latitude":float(latitude),
        "longitude":float(longitude),
        "housing_median_age":float(housing_median_age),
        "total_rooms":int(total_rooms),
        "total_bedrooms":int(total_bedrooms),
        "population":int(population),
        "households":int(households),
        "median_income":float(median_income),
        "ocean_proximity":str(ocean_proximity)
    }
    try:
        for attempt in range(max_retries):
            response = requests.post(
                url=f"{BACKEND_URL}/predict/",
                json=payload,
                timeout=10
            )

            response.raise_for_status()

            data = response.json()['house_value']

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

def main():
    st.set_page_config(
        page_title=' California House Value Predictor 2026',
        page_icon=':chart_with_upwards_trend:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # ADD A HEADING FOR THE SIDEBAR
    st.sidebar.header("California House Value Predictor")

    # ADD A DIVIDER
    st.sidebar.divider()
    
    city_names = get_city_names()

    # ADD A SEARCH BAR FOR THE LOCATION
    txt_location = st.sidebar.selectbox('Search for a city..', city_names)
    
    # GET THE COORDINATES
    data = get_location_coordinates(txt_location)

    lat, lon = None, None
    if data:
        lat = data[0]['lat']
        lon = data[0]['lon']

    if (lat is None) and (lon is None):
        st.error("City not found or invalid input.")

    # ADD AN INPUT FIELD FOR HOUSE AGE
    median_age = st.sidebar.number_input(label='House Median Age', placeholder='e.g 37', step=1, min_value=37)

    # ADD AN INPUT FIELD FOR THE TOTAL ROOMS
    total_rooms = st.sidebar.number_input(label='Total Rooms', placeholder='e.g 3', step=1, min_value=1)

    # ADD AN INPUT FIELD FOR THE TOTAL BEDROOMS
    total_bedrooms = st.sidebar.number_input(label='Total Bedrooms', placeholder='e.g 2', step=1, min_value=1)

    population = get_city_population(city_name=txt_location)

    # ADD AN INPUT FIELD FOR THE POPULATION
    st.sidebar.number_input(label='Population (in millions)', value=population, disabled=True)

    # ADD AN INPUT FIELD FOR THE TOTAL HOUSEHOLDS
    total_households = st.sidebar.number_input(label='Total Households (in thousands)', placeholder='e.g 100', step=1, min_value=1)

    # ADD AN INPUT FIELD FOR THE MEDIAN INCOME
    median_income = st.sidebar.number_input(label='Median Income (per annum)', placeholder='e.g 100,100', step=100, min_value=100100)

    ocean_proximity = st.sidebar.selectbox(
        "Choose ocean proximity",
        ["Near Bay", "<1 OCEAN", "Near Ocean", "Inland", "Island"]
    )

    st.sidebar.button(label="Reset", type='primary', use_container_width=True)

    
    with st.container():
        st.title('Curious about your home’s value?') # h1 tag
        st.write("Use our AI-powered predictor to get a quick and reliable price estimate.") # p tag

        if model_info()['status'] == "Healthy":
            if st.button(
                label="Predict", 
                on_click=predict, 
                args=(lat, lon, median_age, total_rooms, total_bedrooms, population, total_households, median_income, ocean_proximity),
                type='secondary'
            ):
                
                with st.spinner(text="Calculating your house value..."):
                    house_value = predict(
                        latitude=lat, longitude=lon,
                        housing_median_age=median_age,
                        total_rooms=total_rooms,
                        total_bedrooms=total_bedrooms,
                        population=population,
                        households=total_households,
                        median_income=median_income,
                        ocean_proximity=ocean_proximity
                    )
                    st.success(f"Your house is likely to be worth ${house_value}.", icon='🔥')

        else:
            st.warning("Sorry. Our AI predictor can't make predictions right now.")

    st.divider()

    # DISPLAY MODEL INFORMATION
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label='Status', value=f"{model_info()['status']}", delta_color="red")
        
    with col2:
        st.metric(label='Version', value=f"{model_info()['version']}", delta_color="green")

    with col3:
        st.metric(label="Accuracy", value=f"{ model_info()['accuracy'] }%", delta_color="red")

    with col4:
        st.metric(label="Last Train Date", value=f"{datetime.now().strftime("%Y-%m-%d")}", delta_color="green")

if __name__ == "__main__":
    main()
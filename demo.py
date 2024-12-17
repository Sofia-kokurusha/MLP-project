import os
from pathlib import Path
import pandas as pd
import numpy as np
import dill
from PIL import Image
import streamlit as st

from ecolyon.transforms import process_df


path_to_repo = Path(__file__).parent.resolve()
path_to_data = path_to_repo / 'data' / 'cleaned.csv'


def init_session_state():
    # session state
    if 'loaded' not in st.session_state:

        # import raw data
        df_clean = pd.read_csv(path_to_data)

         # Store the min and max of scaled coordinates
        st.session_state.lat_min = df_clean["LATITUDE_SCALED"].min()
        st.session_state.lat_max = df_clean["LATITUDE_SCALED"].max()
        st.session_state.lon_min = df_clean["LONGITUDE_SCALED"].min()
        st.session_state.lon_max = df_clean["LONGITUDE_SCALED"].max()
        
        # Actual latitude and longitude ranges for New York City
        st.session_state.actual_lat_min = 40.477399  # Approximate southernmost point of NYC
        st.session_state.actual_lat_max = 40.917577  # Approximate northernmost point of NYC
        st.session_state.actual_lon_min = -74.259090 # Approximate westernmost point of NYC
        st.session_state.actual_lon_max = -73.700272 # Approximate easternmost point of NYC

        # preprocess data
        df_train = df_clean[df_clean["Split"]== 0].reset_index(drop=True)
        df_test = df_clean[df_clean["Split"]== 1].reset_index(drop=True)
        df_train= df_train.drop(columns="Split")
        df_test = df_test.drop(columns="Split")

        X_test = df_test[["LATITUDE_SCALED","LONGITUDE_SCALED","CRASH DATEDayofweek"]]
        Y_test = df_test["y"]

        # load regression model
        path_to_model = os.path.join(path_to_repo, 'saves', 'CatB_regressor.pk')
        with open(path_to_model, 'rb') as file:
            model = dill.load(file)

        # store in cache
        st.session_state.loaded = True
        st.session_state.X_test = X_test
        st.session_state.Y_test = Y_test
        st.session_state.model = model


def rescale_coordinates(lat_scaled,lon_scaled):
    lat_rescaled = np.interp(lat_scaled, 
                             [st.session_state.lat_min, st.session_state.lat_max], 
                             [st.session_state.actual_lat_min, st.session_state.actual_lat_max])
    lon_rescaled = np.interp(lon_scaled, 
                             [st.session_state.lon_min, st.session_state.lon_max], 
                             [st.session_state.actual_lon_min, st.session_state.actual_lon_max])
    return lat_rescaled,lon_rescaled

def display_map(latitude_scaled,longitude_scaled):
    st.subheader('Crash Location')
    latitude, longitude = rescale_coordinates(latitude_scaled, longitude_scaled)
    df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(df)
    return

def display_crash_prediction(prediction):
    st.subheader('Predicted Number of Crashes')
    st.write(f"{prediction:.2f}")
    return

def display_input_features(latitude_scaled, longitude_scaled, day_of_week):
    st.subheader('Car Crash Description')
    feat0, val0, feat1, val1, feat2, val2 = st.columns(6)
    
    with feat0:
        st.info("Latitude (Scaled)")
    with val0:
        st.success(f"{latitude_scaled:.4f}")
    with feat1:
        st.info("Longitude (Scaled)")
    with val1:
        st.success(f"{longitude_scaled:.4f}")
    with feat2:
        st.info("Day of Week")
    with val2:
        st.success(str(day_of_week))
    return

def app():
    init_session_state()
    st.title('New York Number of Car Crashes Predictor')

    latitude_scaled = st.slider("Latitude (Scaled)", 0.0, 100.0, 50.0)
    longitude_scaled = st.slider("Longitude (Scaled)", 0.0, 100.0, 50.0)
    day_of_week = st.selectbox("Day of the Week", range(7))

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "LATITUDE_SCALED": [latitude_scaled],
            "LONGITUDE_SCALED": [longitude_scaled],
            "CRASH DATEDayofweek": [day_of_week]
        })
        prediction = st.session_state.model.predict(input_data)[0]
        
        display_map(latitude_scaled, longitude_scaled)
        display_crash_prediction(prediction)
        display_input_features(latitude_scaled, longitude_scaled, day_of_week)

    return


if __name__ == '__main__':
    app()
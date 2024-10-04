import streamlit as st
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

# DB Configuration
DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

# Column names as provided
COLUMN_NAMES = {
    'VESSEL_NAME': 'VESSEL_NAME',
    'REPORT_DATE': 'REPORT_DATE',
    'ME_CONSUMPTION': 'ME_CONSUMPTION',
    'OBSERVERD_DISTANCE': 'OBSERVERD_DISTANCE',
    'SPEED': 'SPEED',
    'DISPLACEMENT': 'DISPLACEMENT',
    'STEAMING_TIME_HRS': 'STEAMING_TIME_HRS',
    'WINDFORCE': 'WINDFORCE',
    'VESSEL_ACTIVITY': 'VESSEL_ACTIVITY',
    'LOAD_TYPE': 'LOAD_TYPE',
    'DRAFTFWD': 'DRAFTFWD',
    'DRAFTAFT': 'DRAFTAFT'
}

# Function to create DB engine
def get_db_engine():
    encoded_password = urllib.parse.quote(DB_CONFIG['password'])
    db_url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(db_url)
    return engine

# Query the database for the vessel's last 1 year of data with wind force <= 4
def get_vessel_data(vessel_name, engine):
    vessel_name = vessel_name.strip().upper().replace("'", "''")  # Handling special characters and uppercasing

    query = f"""
    SELECT * FROM sf_consumption_logs 
    WHERE "VESSEL_NAME" = upper('{vessel_name}') 
    AND "REPORT_DATE" >= '{datetime.now() - timedelta(days=365)}'
    AND "WINDFORCE" <= 4
    """
    
    try:
        data = pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error executing query: {e}")
        data = pd.DataFrame()  # Return an empty DataFrame if the query fails
    return data

# Function to evaluate the model and cache it to avoid retraining
def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Function to optimize trim by finding the best forward and aft drafts
def optimize_trim(model, speed, displacement, wind_force):
    def objective(drafts):
        forward_draft, aft_draft = drafts
        trim = aft_draft - forward_draft
        input_data = np.array([[speed, wind_force, trim, displacement]])
        predicted_fuel_consumption = model.predict(input_data)
        return predicted_fuel_consumption[0]  # Minimize this value

    initial_guess = [7.0, 9.0]  # Example starting points for fwd and aft draft
    bounds = [(5.0, 12.0), (5.0, 12.0)]  # Fwd and aft draft bounds
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun  # Optimal drafts and minimum fuel consumption

# Streamlit App for Vessel Data Input and Model Training
st.title('Trim Optimization: Vessel Data-Based')

# Input vessel name
vessel_name = st.text_input('Enter Vessel Name')

# Use session_state to store data and model to avoid resetting
if 'vessel_data' not in st.session_state:
    st.session_state.vessel_data = None

if 'model' not in st.session_state:
    st.session_state.model = None

if st.button('Fetch Vessel Data'):
    if not vessel_name:
        st.error("Please enter a valid vessel name.")
    else:
        engine = get_db_engine()
        st.session_state.vessel_data = get_vessel_data(vessel_name, engine)

if st.session_state.vessel_data is not None:
    st.write(f"Data fetched for vessel: {vessel_name}")
    st.dataframe(st.session_state.vessel_data)

    # Preprocess the data, calculating trim as DRAFTAFT - DRAFTFWD
    vessel_data = st.session_state.vessel_data
    vessel_data['trim'] = vessel_data[COLUMN_NAMES['DRAFTAFT']] - vessel_data[COLUMN_NAMES['DRAFTFWD']]
    features = [COLUMN_NAMES['SPEED'], COLUMN_NAMES['WINDFORCE'], 'trim', COLUMN_NAMES['DISPLACEMENT']]
    target = COLUMN_NAMES['ME_CONSUMPTION']

    # Filter the data for the selected vessel
    X = vessel_data[features]
    y = vessel_data[target]

    # Train the model and store it in session state
    if st.session_state.model is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        st.session_state.model = train_model(X_train, y_train)

        # Display model performance
        y_pred = st.session_state.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = st.session_state.model.score(X_test, y_test)

        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAE: {mae:.4f}')
        st.write(f'R² Score: {r2:.4f}')

    # Trim Optimization Based on User Input
    st.header('Trim Optimization')
    speed = st.slider('Speed (knots)', min_value=5, max_value=20, value=10)
    displacement = st.slider('Displacement (tonnes)', min_value=1000, max_value=20000, value=10000)
    wind_force = st.slider('Wind Force (Beaufort Scale)', min_value=0, max_value=12, value=5)

    # Run optimization to find the best forward and aft drafts
    if st.button('Optimize Trim'):
        optimal_drafts, min_fuel_consumption = optimize_trim(st.session_state.model, speed, displacement, wind_force)
        st.write(f"Optimal Forward Draft: {optimal_drafts[0]:.2f} meters")
        st.write(f"Optimal Aft Draft: {optimal_drafts[1]:.2f} meters")
        st.write(f"Minimum Fuel Consumption: {min_fuel_consumption:.2f} tons per hour")

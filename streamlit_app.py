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

# Query the database for the vessel's last 6 months of data with wind force <= 4
def get_vessel_data(vessel_name, engine):
    vessel_name = vessel_name.strip().upper().replace("'", "''")  # Handling special characters and uppercasing

    query = f"""
    SELECT * FROM sf_consumption_logs
    WHERE "VESSEL_NAME" = upper('{vessel_name}')
    AND "REPORT_DATE" >= '{datetime.now() - timedelta(days=180)}'
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
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adding more estimators and a fixed random state for stability
    model.fit(X_train, y_train)
    return model

# Function to optimize trim by finding the best forward and aft drafts
def optimize_trim(model, speed, displacement):
    def objective(drafts):
        forward_draft, aft_draft = drafts
        trim = aft_draft - forward_draft
        input_data = pd.DataFrame([[speed, trim, displacement, forward_draft, aft_draft]], columns=['SPEED', 'trim', 'DISPLACEMENT', 'DRAFTFWD', 'DRAFTAFT'])
        predicted_fuel_consumption = model.predict(input_data)
        return predicted_fuel_consumption[0]  # Minimize this value

    initial_guess = [6.0, 8.0]  # Adjusted starting points for fwd and aft draft
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

if 'model_laden' not in st.session_state:
    st.session_state.model_laden = None

if 'model_ballast' not in st.session_state:
    st.session_state.model_ballast = None

if st.button('Fetch Vessel Data'):
    if not vessel_name:
        st.error("Please enter a valid vessel name.")
    else:
        engine = get_db_engine()
        st.session_state.vessel_data = get_vessel_data(vessel_name, engine)

if st.session_state.vessel_data is not None:
    st.write(f"Data fetched for vessel: {vessel_name}")
    st.dataframe(st.session_state.vessel_data[list(COLUMN_NAMES.values())])

    # Preprocess the data, calculating trim as DRAFTAFT - DRAFTFWD
    vessel_data = st.session_state.vessel_data.dropna(subset=[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTAFT'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['STEAMING_TIME_HRS'], COLUMN_NAMES['LOAD_TYPE']])
    vessel_data['trim'] = vessel_data[COLUMN_NAMES['DRAFTAFT']] - vessel_data[COLUMN_NAMES['DRAFTFWD']]
    vessel_data['ME_CONSUMPTION_HR'] = vessel_data[COLUMN_NAMES['ME_CONSUMPTION']] / vessel_data[COLUMN_NAMES['STEAMING_TIME_HRS']]
    
    # Separate data into laden and ballast conditions
    laden_data = vessel_data[vessel_data[COLUMN_NAMES['LOAD_TYPE']].str.lower() == 'laden']
    ballast_data = vessel_data[vessel_data[COLUMN_NAMES['LOAD_TYPE']].str.lower() == 'ballast']

    # Features and target for training
    features = [COLUMN_NAMES['SPEED'], 'trim', COLUMN_NAMES['DISPLACEMENT'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]
    target = 'ME_CONSUMPTION_HR'

    # Train models for laden and ballast conditions
    if not laden_data.empty:
        X_laden = laden_data[features]
        y_laden = laden_data[target]
        if not X_laden.isnull().values.any() and not y_laden.isnull().values.any():
            X_train, X_test, y_train, y_test = train_test_split(X_laden, y_laden, test_size=0.2, random_state=42)
            st.session_state.model_laden = train_model(X_train, y_train)
            st.write("Laden model trained successfully.")
    else:
        st.warning("No data available for laden condition.")

    if not ballast_data.empty:
        X_ballast = ballast_data[features]
        y_ballast = ballast_data[target]
        if not X_ballast.isnull().values.any() and not y_ballast.isnull().values.any():
            X_train, X_test, y_train, y_test = train_test_split(X_ballast, y_ballast, test_size=0.2, random_state=42)
            st.session_state.model_ballast = train_model(X_train, y_train)
            st.write("Ballast model trained successfully.")
    else:
        st.warning("No data available for ballast condition.")

    # Trim Optimization for Speeds between 9-14 knots for laden and ballast
    st.header('Trim Optimization Results')
    results_laden = []
    results_ballast = []

    for speed in range(9, 15):
        if st.session_state.model_laden is not None:
            optimal_drafts, min_fuel_consumption = optimize_trim(st.session_state.model_laden, speed, displacement=10000)  # Example displacement
            results_laden.append({
                'Speed (knots)': speed,
                'Loading Condition': 'Laden',
                'Optimal Forward Draft (m)': optimal_drafts[0],
                'Optimal Aft Draft (m)': optimal_drafts[1],
                'Minimum Fuel Consumption (tons/hr)': min_fuel_consumption
            })

        if st.session_state.model_ballast is not None:
            optimal_drafts, min_fuel_consumption = optimize_trim(st.session_state.model_ballast, speed, displacement=10000)  # Example displacement
            results_ballast.append({
                'Speed (knots)': speed,
                'Loading Condition': 'Ballast',
                'Optimal Forward Draft (m)': optimal_drafts[0],
                'Optimal Aft Draft (m)': optimal_drafts[1],
                'Minimum Fuel Consumption (tons/hr)': min_fuel_consumption
            })

    # Display the results as tables
    if results_laden:
        st.write("Optimization results for laden condition:")
        results_laden_df = pd.DataFrame(results_laden)
        st.dataframe(results_laden_df)

    if results_ballast:
        st.write("Optimization results for ballast condition:")
        results_ballast_df = pd.DataFrame(results_ballast)
        st.dataframe(results_ballast_df)

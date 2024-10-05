import streamlit as st
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf')
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}
    
    return results

# Function to optimize drafts
def optimize_drafts(model, scaler, speed):
    best_consumption = float('inf')
    best_drafts = None
    
    for fwd in np.arange(4, 15, 0.1):
        for aft in np.arange(4, 15, 0.1):
            X = scaler.transform([[speed, fwd, aft]])
            consumption = model.predict(X)[0]
            
            if consumption < best_consumption:
                best_consumption = consumption
                best_drafts = (fwd, aft)
    
    return best_drafts, best_consumption

# Streamlit App for Vessel Data Input and Model Training
st.title('Trim Optimization: Vessel Data-Based')

# Input vessel name
vessel_name = st.text_input('Enter Vessel Name')

# Use session_state to store data and model to avoid resetting
if 'vessel_data' not in st.session_state:
    st.session_state.vessel_data = None

if 'results_ballast' not in st.session_state:
    st.session_state.results_ballast = None

if 'results_laden' not in st.session_state:
    st.session_state.results_laden = None

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
    vessel_data['ME_CONSUMPTION_HR'] = vessel_data[COLUMN_NAMES['ME_CONSUMPTION']] / vessel_data[COLUMN_NAMES['STEAMING_TIME_HRS']]
    
    # Separate data into laden and ballast conditions
    ballast_data = vessel_data[vessel_data[COLUMN_NAMES['LOAD_TYPE']].str.lower() == 'ballast']
    laden_data = vessel_data[vessel_data[COLUMN_NAMES['LOAD_TYPE']].str.lower() == 'laden']

    # Features and target for training
    features = [COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]
    target = 'ME_CONSUMPTION_HR'

    # Train models for ballast condition
    if not ballast_data.empty:
        X_ballast = ballast_data[features]
        y_ballast = ballast_data[target]
        if not X_ballast.isnull().values.any() and not y_ballast.isnull().values.any():
            st.session_state.results_ballast = train_and_evaluate_models(X_ballast, y_ballast)
            st.write("Ballast condition models trained successfully.")
    else:
        st.warning("No data available for ballast condition.")

    # Train models for laden condition
    if not laden_data.empty:
        X_laden = laden_data[features]
        y_laden = laden_data[target]
        if not X_laden.isnull().values.any() and not y_laden.isnull().values.any():
            st.session_state.results_laden = train_and_evaluate_models(X_laden, y_laden)
            st.write("Laden condition models trained successfully.")
    else:
        st.warning("No data available for laden condition.")

    # Display model performance
    if st.session_state.results_ballast:
        st.write("Ballast Condition Results:")
        for name, result in st.session_state.results_ballast.items():
            st.write(f"{name}: MSE = {result['MSE']:.4f}, R² = {result['R2']:.4f}")

    if st.session_state.results_laden:
        st.write("Laden Condition Results:")
        for name, result in st.session_state.results_laden.items():
            st.write(f"{name}: MSE = {result['MSE']:.4f}, R² = {result['R2']:.4f}")

    # Optimize drafts for different speeds
    speeds_to_test = [9, 10, 11, 12, 13, 14]

    if st.session_state.results_ballast:
        best_model_ballast = min(st.session_state.results_ballast.items(), key=lambda x: x[1]['MSE'])[1]
        st.write("\nOptimized Drafts for Ballast Condition:")
        for speed in speeds_to_test:
            best_drafts, best_consumption = optimize_drafts(best_model_ballast['Model'], best_model_ballast['Scaler'], speed)
            st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")

    if st.session_state.results_laden:
        best_model_laden = min(st.session_state.results_laden.items(), key=lambda x: x[1]['MSE'])[1]
        st.write("\nOptimized Drafts for Laden Condition:")
        for speed in speeds_to_test:
            best_drafts, best_consumption = optimize_drafts(best_model_laden['Model'], best_model_laden['Scaler'], speed)
            st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")

import streamlit as st
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize  # Import optimization method

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

# Function to evaluate the model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    return rmse, mae, r2

# Function to optimize trim by finding the best forward and aft drafts
def optimize_trim(model, speed, displacement, wind_force):
    # Define the objective function: fuel consumption based on forward and aft drafts
    def objective(drafts):
        forward_draft, aft_draft = drafts
        trim = aft_draft - forward_draft
        input_data = np.array([[speed, wind_force, trim, displacement]])
        predicted_fuel_consumption = model.predict(input_data)
        return predicted_fuel_consumption[0]  # Minimize this value

    # Initial guess for the drafts (mid-range values)
    initial_guess = [7.0, 9.0]  # Example starting points for fwd and aft draft

    # Boundaries for forward and aft drafts (min and max values)
    bounds = [(5.0, 12.0), (5.0, 12.0)]  # Fwd and aft draft bounds

    # Run the optimization
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

    # Return the optimal drafts and the corresponding fuel consumption
    return result.x, result.fun  # Optimal drafts and minimum fuel consumption

# Streamlit App for Vessel Data Input and Model Training
st.title('Trim Optimization: Vessel Data-Based')

# Input vessel name
vessel_name = st.text_input('Enter Vessel Name')

if st.button('Fetch Vessel Data'):
    if not vessel_name:
        st.error("Please enter a valid vessel name.")
    else:
        engine = get_db_engine()
        vessel_data = get_vessel_data(vessel_name, engine)

        if vessel_data.empty:
            st.write(f'No data found for vessel: {vessel_name} with wind force <= 4 in the past year.')
        else:
            st.write(f"Data fetched for vessel: {vessel_name}")
            st.dataframe(vessel_data)

            # Preprocess the data, calculating trim as DRAFTAFT - DRAFTFWD
            vessel_data['trim'] = vessel_data[COLUMN_NAMES['DRAFTAFT']] - vessel_data[COLUMN_NAMES['DRAFTFWD']]
            features = [COLUMN_NAMES['SPEED'], COLUMN_NAMES['WINDFORCE'], 'trim', COLUMN_NAMES['DISPLACEMENT']]
            target = COLUMN_NAMES['ME_CONSUMPTION']

            X = vessel_data[features]
            y = vessel_data[target]

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train and evaluate the model
            model = RandomForestRegressor()
            rmse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)

            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'MAE: {mae:.4f}')
            st.write(f'R² Score: {r2:.4f}')

            # Trim Optimization Based on User Input
            st.header('Trim Optimization')
            speed = st.slider('Speed (knots)', min_value=5, max_value=20, value=10)
            displacement = st.slider('Displacement (tonnes)', min_value=1000, max_value=20000, value=10000)
            wind_force = st.slider('Wind Force (Beaufort Scale)', min_value=0, max_value=12, value=5)

            # Run optimization to find the best forward and aft drafts
            optimal_drafts, min_fuel_consumption = optimize_trim(model, speed, displacement, wind_force)

            st.write(f"Optimal Forward Draft: {optimal_drafts[0]:.2f} meters")
            st.write(f"Optimal Aft Draft: {optimal_drafts[1]:.2f} meters")
            st.write(f"Minimum Fuel Consumption: {min_fuel_consumption:.2f} tons per hour")

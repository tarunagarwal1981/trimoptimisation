import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# DB Configuration
DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

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

@st.cache_data
def fetch_data(vessel_name):
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"""
    SELECT * FROM sf_consumption_logs
    WHERE {COLUMN_NAMES['VESSEL_NAME']} = %s
    AND {COLUMN_NAMES['WINDFORCE']} <= 4
    AND {COLUMN_NAMES['STEAMING_TIME_HRS']} >= 16
    """
    df = pd.read_sql_query(query, conn, params=(vessel_name,))
    conn.close()
    return df

def preprocess_data(df):
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']] > 0) &
            (df[COLUMN_NAMES['SPEED']] > 0) &
            (df[COLUMN_NAMES['DRAFTFWD']] > 0) &
            (df[COLUMN_NAMES['DRAFTAFT']] > 0)]
    return df

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

st.title("Vessel Draft Optimization")

vessel_name = st.text_input("Enter Vessel Name:")

if vessel_name:
    df = fetch_data(vessel_name)
    
    if df.empty:
        st.warning("No data found for the specified vessel.")
    else:
        df = preprocess_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast']
        
        # Train models for ballast condition
        if not df_ballast.empty:
            X_ballast = df_ballast[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]]
            y_ballast = df_ballast[COLUMN_NAMES['ME_CONSUMPTION']]
            ballast_results = train_and_evaluate_models(X_ballast, y_ballast)
            st.write("Ballast Condition Results:")
            for name, result in ballast_results.items():
                st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
        else:
            st.warning("No data available for ballast condition.")
        
        # Train models for laden condition
        if not df_laden.empty:
            X_laden = df_laden[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]]
            y_laden = df_laden[COLUMN_NAMES['ME_CONSUMPTION']]
            laden_results = train_and_evaluate_models(X_laden, y_laden)
            st.write("Laden Condition Results:")
            for name, result in laden_results.items():
                st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
        else:
            st.warning("No data available for laden condition.")
        
        # Optimize drafts for different speeds
        speeds_to_test = [10, 11, 12]
        
        if df_ballast.empty == False:
            st.subheader("Optimized Drafts for Ballast Condition:")
            best_model_ballast = min(ballast_results.items(), key=lambda x: x[1]['MSE'])[1]
            for speed in speeds_to_test:
                best_drafts, best_consumption = optimize_drafts(best_model_ballast['Model'], best_model_ballast['Scaler'], speed)
                st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")
        
        if df_laden.empty == False:
            st.subheader("Optimized Drafts for Laden Condition:")
            best_model_laden = min(laden_results.items(), key=lambda x: x[1]['MSE'])[1]
            for speed in speeds_to_test:
                best_drafts, best_consumption = optimize_drafts(best_model_laden['Model'], best_model_laden['Scaler'], speed)
                st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")

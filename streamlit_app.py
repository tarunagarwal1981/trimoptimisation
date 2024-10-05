import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import multiprocessing

# DB Configuration
DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

# Database connection pooling
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
    if connection_pool:
        st.info("Connection pool created successfully")
except Exception as e:
    st.error(f"Error creating connection pool: {e}")

COLUMN_NAMES = {
    'VESSEL_NAME': 'vessel_name',
    'REPORT_DATE': 'report_date',
    'ME_CONSUMPTION': 'me_consumption',
    'OBSERVERD_DISTANCE': 'observerd_distance',
    'SPEED': 'speed',
    'DISPLACEMENT': 'displacement',
    'STEAMING_TIME_HRS': 'steaming_time_hrs',
    'WINDFORCE': 'windforce',
    'VESSEL_ACTIVITY': 'vessel_activity',
    'LOAD_TYPE': 'load_type',
    'DRAFTFWD': 'draftfwd',
    'DRAFTAFT': 'draftaft'
}

@st.cache_data
def fetch_data(vessel_name):
    try:
        conn = connection_pool.getconn()
        query = f"""
        SELECT {', '.join(COLUMN_NAMES.values())} FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
        AND "{COLUMN_NAMES['WINDFORCE']}"::float <= 4
        AND "{COLUMN_NAMES['STEAMING_TIME_HRS']}"::float >= 16
        LIMIT 10000
        """
        df = pd.read_sql_query(query, conn, params=(vessel_name,))
        connection_pool.putconn(conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']])
    return df

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),  # Reduced n_estimators for faster training
        'Linear Regression': LinearRegression(n_jobs=-1),
        'SVR': SVR(kernel='rbf')
    }
    
    results = {}
    
    def train_model(name, model):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}
    
    processes = []
    for name, model in models.items():
        p = multiprocessing.Process(target=train_model, args=(name, model))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    return results

def optimize_drafts(model, scaler, speed):
    def objective(drafts):
        fwd, aft = drafts
        X = scaler.transform([[speed, fwd, aft]])
        return model.predict(X)[0]
    
    bounds = [(4, 15), (4, 15)]  # Forward and aft draft bounds
    initial_guess = [7, 7]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

st.title("Vessel Draft Optimization")

vessel_name = st.text_input("Enter Vessel Name:")

if vessel_name:
    df = fetch_data(vessel_name)
    
    if df.empty:
        st.warning("No data retrieved. Please check the vessel name and try again.")
    else:
        df = preprocess_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'laden']
        
        # Train models for ballast condition
        if not df_ballast.empty:
            X_ballast = df_ballast[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]].astype(float)
            y_ballast = df_ballast[COLUMN_NAMES['ME_CONSUMPTION']].astype(float)
            ballast_results = train_and_evaluate_models(X_ballast, y_ballast)
            
            st.subheader("Ballast Condition Results:")
            for name, result in ballast_results.items():
                st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
            
            st.subheader("Optimized Drafts for Ballast Condition:")
            best_model_ballast = min(ballast_results.items(), key=lambda x: x[1]['MSE'])[1]
            for speed in [10, 11, 12]:
                best_drafts, best_consumption = optimize_drafts(best_model_ballast['Model'], best_model_ballast['Scaler'], speed)
                st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")
        else:
            st.warning("No data available for ballast condition.")
        
        # Train models for laden condition
        if not df_laden.empty:
            X_laden = df_laden[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]].astype(float)
            y_laden = df_laden[COLUMN_NAMES['ME_CONSUMPTION']].astype(float)
            laden_results = train_and_evaluate_models(X_laden, y_laden)
            
            st.subheader("Laden Condition Results:")
            for name, result in laden_results.items():
                st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
            
            st.subheader("Optimized Drafts for Laden Condition:")
            best_model_laden = min(laden_results.items(), key=lambda x: x[1]['MSE'])[1]
            for speed in [10, 11, 12]:
                best_drafts, best_consumption = optimize_drafts(best_model_laden['Model'], best_model_laden['Scaler'], speed)
                st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")
        else:
            st.warning("No data available for laden condition.")

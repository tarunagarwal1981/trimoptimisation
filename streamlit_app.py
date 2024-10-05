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
    'vessel_name': 'vessel_name',
    'report_date': 'report_date',
    'me_consumption': 'me_consumption',
    'observerd_distance': 'observerd_distance',
    'speed': 'speed',
    'displacement': 'displacement',
    'steaming_time_hrs': 'steaming_time_hrs',
    'windforce': 'windforce',
    'vessel_activity': 'vessel_activity',
    'load_type': 'load_type',
    'draftfwd': 'draftfwd',
    'draftaft': 'draftaft'
}

@st.cache_data
def fetch_data(vessel_name):
    try:
        conn = connection_pool.getconn()
        query = f"""
        SELECT {', '.join(COLUMN_NAMES.values())} FROM sf_consumption_logs
        WHERE {COLUMN_NAMES['vessel_name']} = %s
        AND {COLUMN_NAMES['windforce']}::float <= 4
        AND {COLUMN_NAMES['steaming_time_hrs']}::float >= 16
        LIMIT 10000
        """
        df = pd.read_sql_query(query, conn, params=(vessel_name,))
        connection_pool.putconn(conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    df[COLUMN_NAMES['report_date']] = pd.to_datetime(df[COLUMN_NAMES['report_date']])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[COLUMN_NAMES['me_consumption'], COLUMN_NAMES['speed'], COLUMN_NAMES['draftfwd'], COLUMN_NAMES['draftaft']])
    return df

# The rest of your code remains the same

st.title("Vessel Draft Optimization")

vessel_name = st.text_input("Enter Vessel Name:")

if vessel_name:
    df = fetch_data(vessel_name)
    
    if df.empty:
        st.warning("No data retrieved. Please check the vessel name and try again.")
    else:
        df = preprocess_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['load_type']] == 'ballast']
        df_laden = df[df[COLUMN_NAMES['load_type']] == 'laden']
        
        # Train models for ballast condition
        if not df_ballast.empty:
            X_ballast = df_ballast[[COLUMN_NAMES['speed'], COLUMN_NAMES['draftfwd'], COLUMN_NAMES['draftaft']]].astype(float)
            y_ballast = df_ballast[COLUMN_NAMES['me_consumption']].astype(float)
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
            X_laden = df_laden[[COLUMN_NAMES['speed'], COLUMN_NAMES['draftfwd'], COLUMN_NAMES['draftaft']]].astype(float)
            y_laden = df_laden[COLUMN_NAMES['me_consumption']].astype(float)
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

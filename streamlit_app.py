import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
    try:
        conn = psycopg2.connect(**DB_CONFIG, connect_timeout=10)
        query = f"""
        SELECT * FROM sf_consumption_logs
        WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
        AND "{COLUMN_NAMES['WINDFORCE']}"::float <= 4
        AND "{COLUMN_NAMES['STEAMING_TIME_HRS']}"::float >= 16
        """
        df = pd.read_sql_query(query, conn, params=(vessel_name,))
        conn.close()
        return df
    except OperationalError as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()
    except PSQLError as e:
        st.error(f"Database query error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']].astype(float) > 0) &
            (df[COLUMN_NAMES['SPEED']].astype(float) > 0) &
            (df[COLUMN_NAMES['DRAFTFWD']].astype(float) > 0) &
            (df[COLUMN_NAMES['DRAFTAFT']].astype(float) > 0)]
    return df

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
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
        st.warning("No data retrieved. Please check the vessel name and try again.")
    else:
        df = preprocess_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast']
        
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

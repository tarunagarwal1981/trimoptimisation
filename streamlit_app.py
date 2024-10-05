import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

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
    numeric_columns = [COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_columns)
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']] > 0) & 
            (df[COLUMN_NAMES['SPEED']] > 0) & 
            (df[COLUMN_NAMES['DRAFTFWD']] > 0) & 
            (df[COLUMN_NAMES['DRAFTAFT']] > 0)]
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
    
    return df

@st.cache_resource
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
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
    def objective(drafts):
        fwd, aft = drafts
        X = scaler.transform([[speed, fwd, aft]])
        return model.predict(X)[0]
    
    bounds = [(4, 15), (4, 15)]
    result = minimize(objective, [7, 7], bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

st.title("Vessel Draft Optimization")

vessel_name = st.text_input("Enter Vessel Name:")

if vessel_name:
    df = fetch_data(vessel_name)
    
    if df.empty:
        st.warning("No data retrieved. Please check the vessel name and try again.")
    else:
        df = preprocess_data(df)
        
        if df.empty:
            st.warning("After preprocessing, no valid data remains. Please check the data quality.")
        else:
            # Separate ballast and laden conditions
            df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']
            df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast']
            
            for condition, data in [("Ballast", df_ballast), ("Laden", df_laden)]:
                if not data.empty:
                    st.subheader(f"{condition} Condition Results:")
                    X = data[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']]].astype(float)
                    y = data[COLUMN_NAMES['ME_CONSUMPTION']].astype(float)
                    
                    try:
                        results = train_and_evaluate_models(X, y)
                        
                        for name, result in results.items():
                            st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                        
                        st.subheader(f"Optimized Drafts for {condition} Condition:")
                        best_model = min(results.items(), key=lambda x: x[1]['MSE'])[1]
                        for speed in [10, 11, 12]:
                            best_drafts, best_consumption = optimize_drafts(best_model['Model'], best_model['Scaler'], speed)
                            st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")
                    except Exception as e:
                        st.error(f"An error occurred during model training for {condition} condition: {e}")
                else:
                    st.warning(f"No data available for {condition.lower()} condition.")

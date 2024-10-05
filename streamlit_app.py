import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# DB Configuration and COLUMN_NAMES remain the same

@st.cache_data
def fetch_data(vessel_name):
    # fetch_data function remains the same

@st.cache_data
def preprocess_data(df):
    # preprocess_data function remains the same

def plot_data(df):
    # plot_data function remains the same

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}

def optimize_drafts(model, scaler, speed, displacement):
    best_consumption = float('inf')
    best_drafts = None
    
    for mean_draft in np.arange(5, 15, 0.1):
        for trim in np.arange(-2, 2, 0.1):
            fwd = mean_draft - trim/2
            aft = mean_draft + trim/2
            if 4 <= fwd <= 15 and 4 <= aft <= 15:
                X = scaler.transform([[speed, fwd, aft, displacement, trim, mean_draft]])
                consumption = model.predict(X)[0]
                
                if consumption < best_consumption:
                    best_consumption = consumption
                    best_drafts = (fwd, aft)
    
    return best_drafts, best_consumption

st.title("Vessel Draft Optimization")

vessel_name = st.text_input("Enter Vessel Name:")

if vessel_name:
    with st.spinner("Fetching and processing data..."):
        df = fetch_data(vessel_name)
    
    if df.empty:
        st.warning("No data retrieved. Please check the vessel name and try again.")
    else:
        df = preprocess_data(df)
        
        st.subheader("Data Overview")
        st.dataframe(df[list(COLUMN_NAMES.values()) + ['TRIM', 'MEAN_DRAFT']])
        
        plot_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast']
        
        for condition, data in [("Ballast", df_ballast), ("Laden", df_laden)]:
            if not data.empty:
                st.subheader(f"{condition} Condition Analysis")
                X = data[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT'], 
                          COLUMN_NAMES['DISPLACEMENT'], 'TRIM', 'MEAN_DRAFT']]
                y = data[COLUMN_NAMES['ME_CONSUMPTION']]
                
                with st.spinner(f"Training model for {condition} condition..."):
                    result = train_model(X, y)
                
                st.write(f"Random Forest: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                
                st.subheader(f"Optimized Drafts for {condition} Condition:")
                avg_displacement = data[COLUMN_NAMES['DISPLACEMENT']].mean()
                
                optimized_drafts = []
                for speed in range(9, 15):
                    best_drafts, best_consumption = optimize_drafts(result['Model'], result['Scaler'], 
                                                                    speed, avg_displacement)
                    optimized_drafts.append({
                        'Speed': speed,
                        'FWD Draft': round(best_drafts[0], 2),
                        'AFT Draft': round(best_drafts[1], 2),
                        'Estimated Consumption': round(best_consumption, 2)
                    })
                
                st.table(pd.DataFrame(optimized_drafts))
            else:
                st.warning(f"No data available for {condition.lower()} condition.")

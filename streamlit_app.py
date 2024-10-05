import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    df = df.dropna(subset=[COLUMN_NAMES['LOAD_TYPE']])
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[COLUMN_NAMES['ME_CONSUMPTION'], COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT']])
    df['MEAN_DRAFT'] = (df[COLUMN_NAMES['DRAFTAFT']] + df[COLUMN_NAMES['DRAFTFWD']]) / 2
    return df

@st.cache_resource(ttl=3600)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}

def optimize_drafts(model, scaler, speed, displacement):
    best_consumption = float('inf')
    best_drafts = None
    
    for mean_draft in np.arange(5, 15, 1.0):  # Increased step size for performance
        for trim in np.arange(-1, 1, 1.0):
            fwd = mean_draft - trim / 2
            aft = mean_draft + trim / 2
            if 4 <= fwd <= 15 and 4 <= aft <= 15:
                X = scaler.transform([[speed, fwd, aft, displacement, mean_draft]])
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
        
        st.subheader("Data Overview")
        st.dataframe(df[list(COLUMN_NAMES.values()) + ['MEAN_DRAFT']])
        
        # 3D Scatter Plot
        st.subheader("3D Data Visualization")
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df[COLUMN_NAMES['SPEED']],
            y=df['MEAN_DRAFT'],
            z=df[COLUMN_NAMES['ME_CONSUMPTION']],
            mode='markers',
            marker=dict(
                size=5,
                color=df[COLUMN_NAMES['DISPLACEMENT']],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig_3d.update_layout(scene=dict(
            xaxis_title='Speed',
            yaxis_title='Mean Draft',
            zaxis_title='ME Consumption'),
            width=800, height=800,
            title="3D Plot: Speed, Mean Draft, ME Consumption (Color: Displacement)"
        )
        
        st.plotly_chart(fig_3d)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'laden']
        
        for condition, data in [("Ballast", df_ballast), ("Laden", df_laden)]:
            if not data.empty:
                st.subheader(f"{condition} Condition Analysis")
                X = data[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT'], COLUMN_NAMES['DISPLACEMENT'], 'MEAN_DRAFT']]
                y = data[COLUMN_NAMES['ME_CONSUMPTION']]
                
                with st.spinner(f"Training model for {condition} condition..."):
                    result = train_model(X, y)
                
                st.write(f"Random Forest: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                
                st.subheader(f"Optimized Drafts for {condition} Condition:")
                avg_displacement = data[COLUMN_NAMES['DISPLACEMENT']].mean()
                optimization_results = []
                for speed in range(9, 15):  # Speeds from 9 to 14 knots
                    best_drafts, best_consumption = optimize_drafts(result['Model'], result['Scaler'], speed, avg_displacement)
                    optimization_results.append((speed, best_drafts[0], best_drafts[1], best_consumption))
                
                # Display the results in a table
                optimization_df = pd.DataFrame(optimization_results, columns=['Speed (knots)', 'Best FWD Draft (m)', 'Best AFT Draft (m)', 'Estimated Consumption (tons/hr)'])
                st.table(optimization_df)
            else:
                st.warning(f"No data available for {condition.lower()} condition.")

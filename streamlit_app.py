import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

@st.cache_data
def preprocess_data(df):
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    for col in ['ME_CONSUMPTION', 'SPEED', 'DRAFTFWD', 'DRAFTAFT', 'DISPLACEMENT', 'STEAMING_TIME_HRS', 'WINDFORCE']:
        df[COLUMN_NAMES[col]] = pd.to_numeric(df[COLUMN_NAMES[col]], errors='coerce')
    
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']] > 0) &
            (df[COLUMN_NAMES['SPEED']] > 0) &
            (df[COLUMN_NAMES['DRAFTFWD']] > 0) &
            (df[COLUMN_NAMES['DRAFTAFT']] > 0)]
    
    df['TRIM'] = df[COLUMN_NAMES['DRAFTAFT']] - df[COLUMN_NAMES['DRAFTFWD']]
    df['MEAN_DRAFT'] = (df[COLUMN_NAMES['DRAFTAFT']] + df[COLUMN_NAMES['DRAFTFWD']]) / 2
    df['DRAFT_RATIO'] = df[COLUMN_NAMES['DRAFTFWD']] / df[COLUMN_NAMES['DRAFTAFT']]
    
    return df

def plot_data(df):
    st.subheader("Data Visualization")
    
    # 2D Scatter Plots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Speed vs ME Consumption", "Mean Draft vs ME Consumption",
                                                        "Trim vs ME Consumption", "Displacement vs ME Consumption"))
    
    fig.add_trace(go.Scatter(x=df[COLUMN_NAMES['SPEED']], y=df[COLUMN_NAMES['ME_CONSUMPTION']], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['MEAN_DRAFT'], y=df[COLUMN_NAMES['ME_CONSUMPTION']], mode='markers'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['TRIM'], y=df[COLUMN_NAMES['ME_CONSUMPTION']], mode='markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df[COLUMN_NAMES['DISPLACEMENT']], y=df[COLUMN_NAMES['ME_CONSUMPTION']], mode='markers'), row=2, col=2)
    
    fig.update_layout(height=800, width=800, title_text="2D Scatter Plots")
    st.plotly_chart(fig)
    
    # 3D Scatter Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df[COLUMN_NAMES['SPEED']],
        y=df['TRIM'],
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
        yaxis_title='Trim',
        zaxis_title='ME Consumption'),
        width=800, height=800,
        title="3D Plot: Speed, Trim, ME Consumption (Color: Displacement)"
    )
    
    st.plotly_chart(fig_3d)

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}
    
    return results

def optimize_drafts(model, scaler, speed, displacement):
    def objective(x):
        fwd, aft = x
        trim = aft - fwd
        mean_draft = (fwd + aft) / 2
        X = scaler.transform([[speed, fwd, aft, displacement, trim, mean_draft, fwd/aft]])
        return model.predict(X)[0]
    
    bounds = [(4, 15), (4, 15)]
    result = minimize(objective, [9, 9], method='L-BFGS-B', bounds=bounds)
    
    return result.x, result.fun

def generate_optimization_table(model, scaler, displacement, condition):
    speeds = range(9, 15)
    data = []
    
    for speed in speeds:
        best_drafts, best_consumption = optimize_drafts(model, scaler, speed, displacement)
        data.append({
            'Speed': speed,
            'Draft FWD': best_drafts[0],
            'Draft AFT': best_drafts[1],
            'Estimated Consumption': best_consumption
        })
    
    df = pd.DataFrame(data)
    st.subheader(f"Optimized Drafts for {condition} Condition:")
    st.table(df.style.format({
        'Draft FWD': '{:.2f}',
        'Draft AFT': '{:.2f}',
        'Estimated Consumption': '{:.2f}'
    }))

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
        st.dataframe(df[list(COLUMN_NAMES.values()) + ['TRIM', 'MEAN_DRAFT', 'DRAFT_RATIO']])
        
        plot_data(df)
        
        # Separate ballast and laden conditions
        df_ballast = df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']
        df_laden = df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast']
        
        for condition, data in [("Ballast", df_ballast), ("Laden", df_laden)]:
            if not data.empty:
                st.subheader(f"{condition} Condition Analysis")
                X = data[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT'], 
                          COLUMN_NAMES['DISPLACEMENT'], 'TRIM', 'MEAN_DRAFT', 'DRAFT_RATIO']]
                y = data[COLUMN_NAMES['ME_CONSUMPTION']]
                
                with st.spinner(f"Training models for {condition} condition..."):
                    results = train_models(X, y)
                
                for name, result in results.items():
                    st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                
                selected_model = st.selectbox(f"Select model for {condition} condition optimization:", list(results.keys()))
                
                avg_displacement = data[COLUMN_NAMES['DISPLACEMENT']].mean()
                generate_optimization_table(results[selected_model]['Model'], results[selected_model]['Scaler'], avg_displacement, condition)
            else:
                st.warning(f"No data available for {condition.lower()} condition.")

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-ap-south-1.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.conrxbcvuogbzfysomov'),
    'password': os.getenv('DB_PASSWORD', 'wXAryCC8@iwNvj#'),
    'port': os.getenv('DB_PORT', '6543')
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
def fetch_data(vessel_name: str) -> pd.DataFrame:
    """
    Fetch data from the database for a given vessel name.
    
    Args:
        vessel_name (str): Name of the vessel to fetch data for.
    
    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    logger.debug(f"Fetching data for vessel: {vessel_name}")
    try:
        with psycopg2.connect(**DB_CONFIG, connect_timeout=10) as conn:
            query = f"""
            SELECT * FROM sf_consumption_logs
            WHERE "{COLUMN_NAMES['VESSEL_NAME']}" = %s
            AND "{COLUMN_NAMES['WINDFORCE']}"::float <= 4
            AND "{COLUMN_NAMES['STEAMING_TIME_HRS']}"::float >= 16
            """
            df = pd.read_sql_query(query, conn, params=(vessel_name,))
        if df.empty:
            logger.warning(f"No data found for vessel: {vessel_name}")
            st.warning("No data found for the specified vessel name.")
        return df
    except OperationalError as e:
        logger.error(f"Database connection error: {e}")
        st.error(f"Unable to connect to the database. Please check your connection and try again.")
    except PSQLError as e:
        logger.error(f"Database query error: {e}")
        st.error(f"An error occurred while querying the database. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred. Please try again later.")
    return pd.DataFrame()

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the fetched data.
    
    Args:
        df (pd.DataFrame): Raw data fetched from the database.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    logger.debug("Preprocessing data")
    df[COLUMN_NAMES['REPORT_DATE']] = pd.to_datetime(df[COLUMN_NAMES['REPORT_DATE']])
    numeric_columns = ['ME_CONSUMPTION', 'SPEED', 'DRAFTFWD', 'DRAFTAFT', 'DISPLACEMENT', 'STEAMING_TIME_HRS', 'WINDFORCE']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']] > 0) &
            (df[COLUMN_NAMES['SPEED']] > 0) &
            (df[COLUMN_NAMES['DRAFTFWD']] > 0) &
            (df[COLUMN_NAMES['DRAFTAFT']] > 0)]
    
    df['TRIM'] = df[COLUMN_NAMES['DRAFTAFT']] - df[COLUMN_NAMES['DRAFTFWD']]
    df['MEAN_DRAFT'] = (df[COLUMN_NAMES['DRAFTAFT']] + df[COLUMN_NAMES['DRAFTFWD']]) / 2
    df['DRAFT_RATIO'] = df[COLUMN_NAMES['DRAFTFWD']] / df[COLUMN_NAMES['DRAFTAFT']]
    
    return df

def plot_data(df: pd.DataFrame) -> None:
    """
    Create and display a 3D scatter plot of the data.
    
    Args:
        df (pd.DataFrame): Preprocessed data to plot.
    """
    logger.debug("Plotting data")
    st.subheader("Data Visualization")
    
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

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str) -> Optional[Dict]:
    """
    Train a machine learning model on the given data.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_type (str): Type of model to train.
    
    Returns:
        Optional[Dict]: Dictionary containing model performance metrics and the trained model, or None if an error occurs.
    """
    logger.debug(f"Training {model_type} model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Linear Regression with Polynomial Features': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42),
        'Stacking Regressor': StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
                ('dt', DecisionTreeRegressor(random_state=42))
            ],
            final_estimator=LinearRegression()
        ),
        'Decision Tree with AdaBoost': AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=50,
            random_state=42
        )
    }
    
    if model_type not in models:
        logger.error(f"Invalid model type: {model_type}")
        st.error("Invalid model type selected.")
        return None
    
    model = models[model_type]
    model.fit(X_train_scaled, y_train_sample)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MSE': mse, 'R2': r2, 'Model': model, 'Scaler': scaler}

def optimize_drafts(model, scaler, speed: float, displacement: float) -> Tuple[Tuple[float, float], float]:
    """
    Optimize drafts for given speed and displacement.
    
    Args:
        model: Trained machine learning model.
        scaler: Fitted StandardScaler.
        speed (float): Vessel speed.
        displacement (float): Vessel displacement.
    
    Returns:
        Tuple[Tuple[float, float], float]: Optimized drafts (forward, aft) and estimated consumption.
    """
    logger.debug(f"Optimizing drafts for speed: {speed}, displacement: {displacement}")
    best_consumption = float('inf')
    best_drafts = None
    
    for mean_draft in np.arange(5, 15, 0.1):
        for trim in np.arange(-2, 2, 0.1):
            fwd = mean_draft - trim/2
            aft = mean_draft + trim/2
            if 4 <= fwd <= 15 and 4 <= aft <= 15:
                X = scaler.transform([[speed, fwd, aft, displacement, trim, mean_draft, fwd/aft]])
                try:
                    consumption = model.predict(X)[0]
                except Exception as e:
                    logger.warning(f"Prediction failed for speed {speed}, fwd {fwd}, aft {aft}: {e}")
                    continue
                
                if consumption < best_consumption:
                    best_consumption = consumption
                    best_drafts = (fwd, aft)
    
    return best_drafts, best_consumption

def main():
    st.title("Vessel Draft Optimization")
    logger.debug("Application started")

    vessel_name = st.text_input("Enter Vessel Name:")
    logger.debug(f"Vessel name entered: {vessel_name}")

    model_type = st.sidebar.selectbox("Select Model Type:", [
        'Random Forest',
        'Linear Regression with Polynomial Features',
        'MLP Regressor',
        'Stacking Regressor',
        'Decision Tree with AdaBoost'
    ])
    logger.debug(f"Model type selected: {model_type}")

    if vessel_name:
        logger.debug("Fetching data")
        with st.spinner("Fetching and processing data..."):
            df = fetch_data(vessel_name)
        
        if df.empty:
            logger.warning("No data retrieved")
            st.warning("No data retrieved. Please check the vessel name and try again.")
        else:
            logger.debug("Data retrieved successfully")
            df = preprocess_data(df)
            logger.debug("Data preprocessed")
            
            st.subheader("Data Overview")
            st.dataframe(df[list(COLUMN_NAMES.values()) + ['TRIM', 'MEAN_DRAFT', 'DRAFT_RATIO']])
            
            plot_data(df)
            logger.debug("Data plotted")
            
            for condition, data in [("Ballast", df[df[COLUMN_NAMES['LOAD_TYPE']] == 'Ballast']),
                                    ("Laden", df[df[COLUMN_NAMES['LOAD_TYPE']] != 'Ballast'])]:
                if not data.empty:
                    st.subheader(f"{condition} Condition Analysis")
                    X = data[[COLUMN_NAMES['SPEED'], COLUMN_NAMES['DRAFTFWD'], COLUMN_NAMES['DRAFTAFT'],
                              COLUMN_NAMES['DISPLACEMENT'], 'TRIM', 'MEAN_DRAFT', 'DRAFT_RATIO']]
                    y = data[COLUMN_NAMES['ME_CONSUMPTION']]
                    
                    logger.debug(f"Training model for {condition} condition")
                    with st.spinner(f"Training model for {condition} condition..."):
                        result = train_model(X, y, model_type)
                    
                    if result is not None:
                        st.write(f"{model_type}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                        
                        st.subheader(f"Optimized Drafts for {condition} Condition:")
                        avg_displacement = data[COLUMN_NAMES['DISPLACEMENT']].mean()
                        
                        optimized_drafts = []
                        logger.debug(f"Optimizing drafts for {condition} condition")
                        with st.spinner(f"Optimizing drafts for {condition} condition..."):
                            for speed in range(9, 15):
                                best_drafts, best_consumption = optimize_drafts(result['Model'], result['Scaler'], speed, avg_displacement)
                                optimized_drafts.append({
                                    'Speed': speed,
                                    'FWD Draft': round(best_drafts[0], 2),
                                    'AFT Draft': round(best_drafts[1], 2),
                                    'Estimated Consumption': round(best_consumption, 2)
                                })
                        
                        st.table(pd.DataFrame(optimized_drafts))
                else:
                    logger.warning(f"No data available for {condition.lower()} condition")
                    st.warning(f"No data available for {condition.lower()} condition.")

    logger.debug("Application finished")

if __name__ == "__main__":
    main()

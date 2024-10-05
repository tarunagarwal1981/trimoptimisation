import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import OperationalError, Error as PSQLError
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
    for col in ['ME_CONSUMPTION', 'SPEED', 'DRAFTFWD', 'DRAFTAFT', 'DISPLACEMENT', 'STEAMING_TIME_HRS', 'WINDFORCE']:
        df[COLUMN_NAMES[col]] = pd.to_numeric(df[COLUMN_NAMES[col]], errors='coerce')
    
    df = df[(df[COLUMN_NAMES['ME_CONSUMPTION']] > 0) &
            (df[COLUMN_NAMES['SPEED']] > 0) &
            (df[COLUMN_NAMES['DRAFTFWD']] > 0) &
            (df[COLUMN_NAMES['DRAFTAFT']] > 0)]
    
    # Feature engineering
    df['TRIM'] = df[COLUMN_NAMES['DRAFTAFT']] - df[COLUMN_NAMES['DRAFTFWD']]
    df['MEAN_DRAFT'] = (df[COLUMN_NAMES['DRAFTAFT']] + df[COLUMN_NAMES['DRAFTFWD']]) / 2
    df['DRAFT_RATIO'] = df[COLUMN_NAMES['DRAFTFWD']] / df[COLUMN_NAMES['DRAFTAFT']]
    
    return df

def plot_data(df):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].scatter(df[COLUMN_NAMES['SPEED']], df[COLUMN_NAMES['ME_CONSUMPTION']])
    axs[0, 0].set_xlabel('Speed')
    axs[0, 0].set_ylabel('ME Consumption')
    axs[0, 1].scatter(df['MEAN_DRAFT'], df[COLUMN_NAMES['ME_CONSUMPTION']])
    axs[0, 1].set_xlabel('Mean Draft')
    axs[0, 1].set_ylabel('ME Consumption')
    axs[1, 0].scatter(df['TRIM'], df[COLUMN_NAMES['ME_CONSUMPTION']])
    axs[1, 0].set_xlabel('Trim')
    axs[1, 0].set_ylabel('ME Consumption')
    axs[1, 1].scatter(df[COLUMN_NAMES['DISPLACEMENT']], df[COLUMN_NAMES['ME_CONSUMPTION']])
    axs[1, 1].set_xlabel('Displacement')
    axs[1, 1].set_ylabel('ME Consumption')
    st.pyplot(fig)

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf')
    }
    
    param_grids = {
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'Linear Regression': {},
        'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    }
    
    results = {}
    
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2, 'Model': best_model, 'Scaler': scaler}
    
    return results

def optimize_drafts(model, scaler, speed, displacement):
    best_consumption = float('inf')
    best_drafts = None
    
    for mean_draft in np.arange(5, 15, 0.1):
        for trim in np.arange(-2, 2, 0.1):
            fwd = mean_draft - trim/2
            aft = mean_draft + trim/2
            if 4 <= fwd <= 15 and 4 <= aft <= 15:
                X = scaler.transform([[speed, fwd, aft, displacement, trim, mean_draft, fwd/aft]])
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
        st.dataframe(df[list(COLUMN_NAMES.values()) + ['TRIM', 'MEAN_DRAFT', 'DRAFT_RATIO']])
        
        st.subheader("Data Visualization")
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
                results = train_and_evaluate_models(X, y)
                
                for name, result in results.items():
                    st.write(f"{name}: MSE = {result['MSE']:.4f}, R2 = {result['R2']:.4f}")
                
                best_model = min(results.items(), key=lambda x: x[1]['MSE'])[1]
                
                st.subheader(f"Optimized Drafts for {condition} Condition:")
                avg_displacement = data[COLUMN_NAMES['DISPLACEMENT']].mean()
                for speed in [10, 11, 12]:
                    best_drafts, best_consumption = optimize_drafts(best_model['Model'], best_model['Scaler'], 
                                                                    speed, avg_displacement)
                    st.write(f"Speed: {speed} knots, Best Drafts: FWD = {best_drafts[0]:.2f}, "
                             f"AFT = {best_drafts[1]:.2f}, Estimated Consumption: {best_consumption:.2f}")
            else:
                st.warning(f"No data available for {condition.lower()} condition.")

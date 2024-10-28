import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import psycopg2
import os
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-ap-south-1.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.conrxbcvuogbzfysomov'),
    'password': os.getenv('DB_PASSWORD', 'wXAryCC8@iwNvj#'),
    'port': os.getenv('DB_PORT', '6543')
}

def fetch_data(vessel_name):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        one_year_ago = datetime.now() - timedelta(days=365)
        query = """
            SELECT "VESSEL_NAME", "ME_CONSUMPTION", "SPEED", "DRAFTFWD", "DRAFTAFT", 
                   "DISPLACEMENT", "LOAD_TYPE", "REPORT_DATE"
            FROM sf_consumption_logs
            WHERE "VESSEL_NAME" = %s 
            AND "REPORT_DATE" >= %s
            AND "WINDFORCE"::float <= 4 
            AND "STEAMING_TIME_HRS"::float >= 16
        """
        df = pd.read_sql_query(query, conn, params=(vessel_name, one_year_ago))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    df['TRIM'] = df['DRAFTAFT'] - df['DRAFTFWD']
    df['MEAN_DRAFT'] = (df['DRAFTAFT'] + df['DRAFTFWD']) / 2
    return df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def optimize_drafts(model, scaler, speed, displacement, min_fwd, max_fwd, min_aft, max_aft):
    def objective_function(drafts):
        fwd, aft = drafts
        mean_draft = (fwd + aft) / 2
        trim = aft - fwd
        X = np.array([[speed, fwd, aft, displacement, trim, mean_draft]])
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)[0]

    bounds = [(min_fwd, max_fwd), (min_aft, max_aft)]
    
    result = differential_evolution(
        objective_function,
        bounds,
        mutation=(0.5, 1),
        recombination=0.7,
        popsize=20,
        tol=0.01,
        polish=True
    )
    
    return result.x, result.fun

def main():
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
            st.dataframe(df)
            
            X = df[['SPEED', 'DRAFTFWD', 'DRAFTAFT', 'DISPLACEMENT', 'TRIM', 'MEAN_DRAFT']]
            y = df['ME_CONSUMPTION']
            
            model, scaler = train_model(X, y)
            
            st.subheader("Optimized Trim:")
            avg_displacement = df['DISPLACEMENT'].mean()
            min_fwd, max_fwd = df['DRAFTFWD'].min(), df['DRAFTFWD'].max()
            min_aft, max_aft = df['DRAFTAFT'].min(), df['DRAFTAFT'].max()
            
            optimized_trims = []
            for speed in range(9, 14):
                best_drafts, best_consumption = optimize_drafts(model, scaler, speed, avg_displacement, min_fwd, max_fwd, min_aft, max_aft)
                trim = best_drafts[1] - best_drafts[0]
                optimized_trims.append({
                    'Speed': round(speed, 1),
                    'Trim': round(trim, 1),
                    'Estimated Consumption': round(best_consumption, 1)
                })
            
            st.table(pd.DataFrame(optimized_trims))

            # Validation checks
            if any(trim['Trim'] < 0 for trim in optimized_trims):
                st.warning("Warning: Some optimized trims are negative. This may indicate an issue with the optimization process.")

if __name__ == "__main__":
    main()

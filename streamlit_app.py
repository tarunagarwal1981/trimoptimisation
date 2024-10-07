import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import psycopg2
import os
from datetime import datetime, timedelta
from scipy.optimize import minimize

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
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

def objective_function(drafts, model, scaler, speed, displacement):
    fwd, aft = drafts
    mean_draft = (fwd + aft) / 2
    trim = aft - fwd
    X = np.array([[speed, fwd, aft, displacement, trim, mean_draft]])
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]

def optimize_drafts(model, scaler, speed, displacement):
    bounds = ((4, 15), (4, 15))  # Assuming draft limits between 4 and 15 meters
    initial_guess = [9, 9]  # Start with equal drafts
    
    result = minimize(
        objective_function,
        initial_guess,
        args=(model, scaler, speed, displacement),
        method='L-BFGS-B',
        bounds=bounds
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
            
            for condition in ['Ballast', 'Laden']:
                st.subheader(f"{condition} Condition Analysis")
                condition_df = df[df['LOAD_TYPE'] == condition]
                
                if condition_df.empty:
                    st.warning(f"No data available for {condition.lower()} condition.")
                    continue
                
                X = condition_df[['SPEED', 'DRAFTFWD', 'DRAFTAFT', 'DISPLACEMENT', 'TRIM', 'MEAN_DRAFT']]
                y = condition_df['ME_CONSUMPTION']
                
                model, scaler, mse, r2 = train_model(X, y)
                st.write(f"Model performance: MSE = {mse:.4f}, R2 = {r2:.4f}")
                
                st.subheader(f"Optimized Drafts for {condition} Condition:")
                avg_displacement = condition_df['DISPLACEMENT'].mean()
                
                optimized_drafts = []
                for speed in range(9, 14):
                    best_drafts, best_consumption = optimize_drafts(model, scaler, speed, avg_displacement)
                    optimized_drafts.append({
                        'Speed': speed,
                        'FWD Draft': round(best_drafts[0], 2),
                        'AFT Draft': round(best_drafts[1], 2),
                        'Estimated Consumption': round(best_consumption, 2)
                    })
                
                st.table(pd.DataFrame(optimized_drafts))

if __name__ == "__main__":
    main()

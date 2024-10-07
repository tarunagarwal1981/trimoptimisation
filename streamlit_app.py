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

# ... [previous code remains unchanged] ...

def optimize_drafts(model, scaler, speed, displacement, min_fwd, max_fwd, min_aft, max_aft):
    def objective_function(drafts):
        fwd, aft = drafts
        mean_draft = (fwd + aft) / 2
        trim = aft - fwd
        X = np.array([[speed, fwd, aft, displacement, trim, mean_draft]])
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)[0]

    bounds = ((min_fwd, max_fwd), (min_aft, max_aft))
    initial_guess = [(min_fwd + max_fwd) / 2, (min_aft + max_aft) / 2]
    
    result = minimize(
        objective_function,
        initial_guess,
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
                min_fwd, max_fwd = condition_df['DRAFTFWD'].min(), condition_df['DRAFTFWD'].max()
                min_aft, max_aft = condition_df['DRAFTAFT'].min(), condition_df['DRAFTAFT'].max()
                
                optimized_drafts = []
                for speed in range(9, 14):
                    best_drafts, best_consumption = optimize_drafts(model, scaler, speed, avg_displacement, min_fwd, max_fwd, min_aft, max_aft)
                    optimized_drafts.append({
                        'Speed': speed,
                        'FWD Draft': round(best_drafts[0], 2),
                        'AFT Draft': round(best_drafts[1], 2),
                        'Estimated Consumption': round(best_consumption, 2)
                    })
                
                st.table(pd.DataFrame(optimized_drafts))

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Set the page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Energy Price Prediction Dashboard")

st.title('Energy Price Prediction Dashboard')

# --- Load Data from Files ---
def load_data():
    try:
        # Load y_val and y_pred_val
        y_data = pd.read_csv('y_data.csv')
        y_val_loaded = y_data['y_val']
        y_pred_val_loaded = y_data['y_pred_val']

        # Load permutation importance values
        feature_importances_values_loaded = joblib.load('permutation_importance_values.pkl')

        # Load X_val column names
        X_val_columns_loaded = pd.read_csv('X_val_columns.csv')['0'].tolist()

        # Reconstruct feature_importances Series
        feature_importances_loaded = pd.Series(feature_importances_values_loaded, index=X_val_columns_loaded)
        feature_importances_loaded = feature_importances_loaded.sort_values(ascending=True)

        # Load metrics
        metrics = {}
        with open('metrics.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                metrics[key] = float(value)
        mae_val_loaded = metrics.get('mae_val')
        rmse_val_loaded = metrics.get('rmse_val')

        return y_val_loaded, y_pred_val_loaded, feature_importances_loaded, mae_val_loaded, rmse_val_loaded
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please ensure 'y_data.csv', 'permutation_importance_values.pkl', 'X_val_columns.csv', and 'metrics.txt' are in the same directory as app.py.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

y_val_series, y_pred_val, feature_importances, mae_val, rmse_val = load_data()

if y_val_series is not None:

    st.header('Model Performance Overview')

    if mae_val is not None and rmse_val is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation MAE", f"{mae_val:.4f}")
        with col2:
            st.metric("Validation RMSE", f"{rmse_val:.4f}")

    # Create a DataFrame for Actual vs Predicted values
    results_df = pd.DataFrame({
        'Actual Price': y_val_series,
        'Predicted Price': y_pred_val
    }).reset_index(drop=True)

    # Plot Actual vs Predicted Prices
    st.subheader('Actual vs. Predicted Prices')
    fig_actual_vs_pred = px.scatter(
        results_df,
        x='Actual Price',
        y='Predicted Price',
        title='Actual vs. Predicted Prices',
        labels={'Actual Price': 'Actual Price', 'Predicted Price': 'Predicted Price'},
        hover_data=['Actual Price', 'Predicted Price']
    )
    fig_actual_vs_pred.add_shape(
        type="line", line=dict(dash='dash'),
        x0=results_df['Actual Price'].min(), y0=results_df['Actual Price'].min(),
        x1=results_df['Actual Price'].max(), y1=results_df['Actual Price'].max()
    )
    st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

    # Plot Feature Importance
    if feature_importances is not None:
        st.subheader('Feature Importance')
        fig_feature_importance = px.bar(
            x=feature_importances.values,
            y=feature_importances.index,
            orientation='h',
            title='Feature Importance (Permutation Importance)',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig_feature_importance.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_feature_importance, use_container_width=True)

    # Plot Target Variable Distribution
    st.subheader('Distribution of Actual Prices')
    fig_target_distribution = px.histogram(
        y_val_series,
        nbins=50,
        title='Distribution of Actual Prices (Validation Set)',
        labels={'value': 'Price Actual'},
        marginal='box'
    )
    st.plotly_chart(fig_target_distribution, use_container_width=True)

    st.markdown("--- \n _This dashboard provides an interactive overview of the energy price prediction model._")
else:
    st.warning("Dashboard data not fully available. Please ensure all data files are correctly placed and loaded.")


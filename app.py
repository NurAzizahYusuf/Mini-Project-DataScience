import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Load daily and weekly Brent Oil data
insight_tambahan = pd.read('brentcrudeoil - dailybrentoil.csv')
df_daily = pd.read_csv('Hasil-Prediksi.csv')  
df_weekly_ma = pd.read_csv('hasil_prediksi_ma.csv')  
df_weekly_eks = pd.read_csv('hasil_prediksi_eks.csv')  

st.title('Grafik Hasil Prediksi')

prediction_method = st.sidebar.selectbox('Pilih Metode Prediksi', ['Moving Average', 'Exponential Smoothing'])

background_color = '#f4f4f4'  
st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
        }}
    </style>
""", unsafe_allow_html=True)

# Visualisasi grafik untuk Daily Brent Oil
fig_daily, ax_daily = plt.subplots(figsize=(12, 6))
ax_daily.plot(df_daily['Close'], label='Actual Close Price (Daily)', color='blue')

if prediction_method == 'Moving Average':
    ax_daily.plot(df_daily['MA'], label='Predicted Close Price (MA)', color='green', linestyle='dashed')
    r2_score_daily = r2_score(df_daily['Close'], df_daily['MA'])
    mse_score_daily = mean_squared_error(df_daily['Close'], df_daily['MA'])
    st.write(f'R2 Score (MA - Daily): {r2_score_daily:.4f}')
    st.write(f'Mean Squared Error (MA - Daily): {mse_score_daily:.4f}')
elif prediction_method == 'Exponential Smoothing':
    ax_daily.plot(df_daily['Exponential_Smoothing'], label='Predicted Close Price (Exponential Smoothing)', color='red', linestyle='dashed')
    r2_score_daily = r2_score(df_daily['Close'], df_daily['Exponential_Smoothing'])
    mse_score_daily = mean_squared_error(df_daily['Close'], df_daily['Exponential_Smoothing'])
    st.write(f'R2 Score (Exponential Smoothing - Daily): {r2_score_daily:.4f}')
    st.write(f'Mean Squared Error (Exponential Smoothing - Daily): {mse_score_daily:.4f}')

ax_daily.set_title('Actual vs. Predicted Close Price (Daily)')
ax_daily.set_xlabel('Date')
ax_daily.set_ylabel('Close Price')
ax_daily.legend()
st.pyplot(fig_daily)

# Visualisasi grafik untuk Weekly Brent Oil (Moving Average)
fig_weekly_ma, ax_weekly_ma = plt.subplots(figsize=(12, 6))
ax_weekly_ma.plot(df_weekly_ma['Actual_Close'], label='Actual Close Price (Weekly - Moving Average)', color='blue')

if prediction_method == 'Moving Average':
    ax_weekly_ma.plot(df_weekly_ma['Predicted_MA'], label='Predicted Close Price (MA - Weekly)', color='green', linestyle='dashed')
    r2_score_weekly_ma = r2_score(df_weekly_ma['Actual_Close'], df_weekly_ma['Predicted_MA'])
    mse_score_weekly_ma = mean_squared_error(df_weekly_ma['Actual_Close'], df_weekly_ma['Predicted_MA'])
    st.write(f'R2 Score (MA - Weekly): {r2_score_weekly_ma:.4f}')
    st.write(f'Mean Squared Error (MA - Weekly): {mse_score_weekly_ma:.4f}')

ax_weekly_ma.set_title('Actual vs. Predicted Close Price (Weekly - Moving Average)')
ax_weekly_ma.set_xlabel('Date')
ax_weekly_ma.set_ylabel('Close Price')
ax_weekly_ma.legend()

# Visualisasi grafik untuk Weekly Brent Oil (Exponential Smoothing)
fig_weekly_eks, ax_weekly_eks = plt.subplots(figsize=(12, 6))
ax_weekly_eks.plot(df_weekly_eks['Actual_Close'], label='Actual Close Price (Weekly - Exponential Smoothing)', color='blue')

if prediction_method == 'Exponential Smoothing':
    ax_weekly_eks.plot(df_weekly_eks['Predicted_Exponential_Smoothing'], label='Predicted Close Price (Exponential Smoothing - Weekly)', color='red', linestyle='dashed')
    r2_score_weekly_eks = r2_score(df_weekly_eks['Actual_Close'], df_weekly_eks['Predicted_Exponential_Smoothing'])
    mse_score_weekly_eks = mean_squared_error(df_weekly_eks['Actual_Close'], df_weekly_eks['Predicted_Exponential_Smoothing'])
    st.write(f'R2 Score (Exponential Smoothing - Weekly): {r2_score_weekly_eks:.4f}')
    st.write(f'Mean Squared Error (Exponential Smoothing - Weekly): {mse_score_weekly_eks:.4f}')

ax_weekly_eks.set_title('Actual vs. Predicted Close Price (Weekly - Exponential Smoothing)')
ax_weekly_eks.set_xlabel('Date')
ax_weekly_eks.set_ylabel('Close Price')
ax_weekly_eks.legend()

st.pyplot(fig_weekly_ma)
st.pyplot(fig_weekly_eks)

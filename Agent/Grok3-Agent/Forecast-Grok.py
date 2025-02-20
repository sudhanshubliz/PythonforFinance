import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Setup
st.title("📊 AI-Powered FP&A Agent")
st.write("Upload revenue data for advanced forecasting and strategic insights!")

# --- Data Handling & User Input ---
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
if uploaded_file:
    # Handle file formats
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.selectbox("Select sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Dynamic column selection
    columns = df.columns.tolist()
    date_column = st.selectbox("Select date column", columns)
    revenue_column = st.selectbox("Select revenue column", columns)

    # Validate columns
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except:
        st.error("Selected date column cannot be parsed to datetime.")
        st.stop()
    
    if not pd.api.types.is_numeric_dtype(df[revenue_column]):
        st.error("Selected revenue column must be numerical.")
        st.stop()

    # Standardize dataframe
    df = df[[date_column, revenue_column]].rename(columns={date_column: 'date', revenue_column: 'revenue'})
    df.set_index('date', inplace=True)
    st.write("Data successfully loaded and columns selected.")

    # --- Multi-Method Forecasting ---
    st.subheader("Forecasting Options")
    models = st.multiselect("Select forecasting models", ["Prophet", "ARIMA", "Exponential Smoothing"])
    frequency = st.selectbox("Select frequency", ["Daily", "Monthly", "Quarterly"])
    horizon = st.number_input("Forecast horizon (number of periods)", min_value=1, value=12)

    if st.button("Generate Forecasts"):
        with st.spinner("Fitting models and generating forecasts..."):
            forecasts = {}
            for model in models:
                if model == "Prophet":
                    prophet_df = df.reset_index().rename(columns={'date': 'ds', 'revenue': 'y'})
                    m = Prophet()
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=horizon, freq=frequency[0])
                    forecast = m.predict(future)
                    forecast_tail = forecast.tail(horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    forecasts['Prophet'] = forecast_tail

                elif model == "ARIMA":
                    # Basic ARIMA implementation using statsmodels (no auto_arima)
                    arima_model = ARIMA(df['revenue'], order=(1, 1, 1), 
                                       seasonal_order=(1, 1, 1, 12 if frequency == "Monthly" else 4 if frequency == "Quarterly" else 1))
                    arima_fit = arima_model.fit()
                    forecast_arima = arima_fit.forecast(steps=horizon)
                    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=horizon, freq=frequency[0])
                    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_arima})
                    forecasts['ARIMA'] = forecast_df

                elif model == "Exponential Smoothing":
                    es_model = ExponentialSmoothing(df['revenue'], trend='add', seasonal='add', 
                                                  seasonal_periods=12 if frequency == "Monthly" else 4 if frequency == "Quarterly" else None)
                    es_fit = es_model.fit()
                    forecast_es = es_fit.forecast(steps=horizon)
                    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=horizon, freq=frequency[0])
                    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_es})
                    forecasts['Exponential Smoothing'] = forecast_df

            # --- Rich & Interactive Visualizations ---
            st.subheader("Forecast Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['revenue'], mode='lines', name='Historical', line=dict(color='blue')))
            for model_name, forecast_df in forecasts.items():
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name=f'{model_name} Forecast'))
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', 
                                            fillcolor='rgba(0,100,80,0.2)', showlegend=False))
            fig.update_layout(title="Historical Data and Forecasts", xaxis_title="Date", yaxis_title="Revenue")
            st.plotly_chart(fig)

            # --- AI-Powered Insights ---
            historical_summary = f"Historical revenue from {df.index.min()} to {df.index.max()} with {len(df)} data points."
            forecast_summary = "Forecasts generated using " + ", ".join(models) + f" for the next {horizon} {frequency} periods."
            ai_prompt = historical_summary + " " + forecast_summary

            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an FP&A AI expert analyzing revenue data."},
                    {"role": "user", "content": ai_prompt}
                ],
                model="llama3-8b-8192",
            )
            ai_commentary = response.choices[0].message.content
            st.subheader("💡 AI-Generated Insights")
            st.markdown(ai_commentary)

            # --- Q&A Feature ---
            st.subheader("🤖 Ask the AI")
            user_question = st.text_input("Ask a question about the forecast (e.g., 'What factors might impact revenue growth?')")
            if st.button("Get AI Response"):
                with st.spinner("Generating AI response..."):
                    question_prompt = ai_prompt + " " + user_question
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an FP&A AI expert analyzing revenue data."},
                            {"role": "user", "content": question_prompt}
                        ],
                        model="llama3-8b-8192",
                    )
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)

            # --- Export Options ---
            st.subheader("📤 Export Results")
            export_df = pd.concat([df.reset_index()] + [f.reset_index() for f in forecasts.values()], axis=0)
            csv = export_df.to_csv(index=False)
            st.download_button("Download Forecast Data (CSV)", csv, "forecast_data.csv", "text/csv")
            st.write("To export visualizations, right-click the plot and save as an image.")

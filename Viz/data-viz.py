import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
st.title("📊 AI-Powered FP&A Visualization Agent")
st.write("Upload an Excel file, and let the AI recommend a visualization and provide financial insights!")

# --- Data Upload ---
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    # Handle Excel file with multiple sheets
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names
    selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --- AI Visualization Recommendation ---
    st.subheader("🤖 AI Visualization Recommendation")
    with st.spinner("Analyzing your data..."):
        # Prepare data summary for AI
        data_summary = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).tolist(),
            "rows": len(df),
            "numeric_cols": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "datetime_cols": df.select_dtypes(include=['datetime64[ns]']).columns.tolist(),
            "sample_data": df.head(5).to_dict()
        }

        # AI prompt to recommend a visualization
        prompt = f"""
        You are an FP&A AI expert. Based on the following data summary, recommend an appropriate data visualization (e.g., line chart, bar chart, scatter plot, pie chart) and explain why it suits the data. Keep it concise.

        Data Summary:
        - Columns: {data_summary['columns']}
        - Data Types: {data_summary['dtypes']}
        - Number of Rows: {data_summary['rows']}
        - Numeric Columns: {data_summary['numeric_cols']}
        - Datetime Columns: {data_summary['datetime_cols']}
        - Sample Data (first 5 rows): {data_summary['sample_data']}
        """
        
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an FP&A AI expert specializing in data visualization."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        ai_recommendation = response.choices[0].message.content
        st.markdown(ai_recommendation)

        # Extract recommended visualization type (assuming AI returns a recognizable type)
        viz_type = "line"  # Default fallback
        if "bar chart" in ai_recommendation.lower():
            viz_type = "bar"
        elif "scatter plot" in ai_recommendation.lower():
            viz_type = "scatter"
        elif "pie chart" in ai_recommendation.lower():
            viz_type = "pie"
        elif "line chart" in ai_recommendation.lower():
            viz_type = "line"

    # --- Visualization Generation ---
    st.subheader("📈 Generated Visualization")
    
    # Let user select columns for X and Y axes (if applicable)
    columns = df.columns.tolist()
    if viz_type in ["line", "bar", "scatter"]:
        x_col = st.selectbox("Select X-axis column", columns)
        y_col = st.selectbox("Select Y-axis column", [col for col in columns if col != x_col and pd.api.types.is_numeric_dtype(df[col])])

        # Handle datetime conversion if applicable
        if pd.api.types.is_datetime64_any_dtype(df[x_col]) or pd.api.types.is_string_dtype(df[x_col]):
            try:
                df[x_col] = pd.to_datetime(df[x_col])
            except:
                pass  # If conversion fails, proceed with original data

        # Create the visualization
        fig = go.Figure()
        if viz_type == "line":
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=y_col))
        elif viz_type == "bar":
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col))
        elif viz_type == "scatter":
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name=y_col))
        
        fig.update_layout(
            title=f"{viz_type.capitalize()} Chart: {y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_white"
        )
        st.plotly_chart(fig)

    elif viz_type == "pie":
        pie_col = st.selectbox("Select column for pie chart (categorical or numeric)", columns)
        if pd.api.types.is_numeric_dtype(df[pie_col]):
            values = df[pie_col].value_counts().values
            labels = df[pie_col].value_counts().index
        else:
            values = df[pie_col].value_counts().values
            labels = df[pie_col].value_counts().index
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title=f"Pie Chart: Distribution of {pie_col}")
        st.plotly_chart(fig)

    # --- FP&A Commentary ---
    st.subheader("💡 FP&A Commentary")
    with st.spinner("Generating financial insights..."):
        commentary_prompt = f"""
        You are an FP&A AI expert. Based on the following data and visualization, provide concise financial planning and analysis commentary. Highlight trends, anomalies, or insights relevant to the data.

        Data Summary:
        - Columns: {data_summary['columns']}
        - Selected Visualization: {viz_type.capitalize()} Chart
        - X-axis: {x_col if viz_type != 'pie' else 'N/A'}
        - Y-axis or Values: {y_col if viz_type != 'pie' else pie_col}
        - Sample Data (first 5 rows): {data_summary['sample_data']}
        """
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an FP&A AI expert providing financial commentary."},
                {"role": "user", "content": commentary_prompt}
            ],
            model="llama3-8b-8192",
        )
        ai_commentary = response.choices[0].message.content
        st.markdown(ai_commentary)

    # --- Export Option ---
    st.subheader("📤 Export Visualization")
    st.write("Right-click the chart and save as an image.")

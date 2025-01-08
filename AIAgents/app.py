# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
from io import BytesIO

# -------------------------
# Global Config/Settings
# -------------------------
st.set_page_config(page_title="FP&A Data Analyzer", layout="wide")

# -------------------------
# Helper Functions
# -------------------------

def compute_variances(df, actual_col="Actual", budget_col="Budget"):
    """
    Compute variance and variance percentage for given columns.
    """
    df["Variance"] = df[actual_col] - df[budget_col]
    df["Variance %"] = np.where(df[budget_col] != 0, 
                                (df["Variance"] / df[budget_col]) * 100, 
                                np.nan)
    return df

def generate_variance_table(df):
    """
    Returns a summarized variance table for display.
    """
    # This could be grouped or pivoted as needed. For demo, we show as-is.
    return df

def generate_bar_chart(df, category_col, value_col):
    """
    Create a bar chart by category or metrics to show variance.
    """
    fig = px.bar(
        df,
        x=category_col,
        y=value_col,
        title=f"Variance by {category_col}",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title=category_col, yaxis_title=value_col)
    return fig

def generate_line_chart(df, x_col, y_col, color_col=None):
    """
    Create a line chart to track trends over time or across dimensions.
    """
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=f"Trend of {y_col} over {x_col}",
        template="plotly_white"
    )
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

def perform_pareto_analysis(df, category_col="Category", value_col="Variance"):
    """
    Pareto Analysis:
    Sort data by 'value_col' descending, compute cumulative % to identify 
    the most significant contributors.
    """
    pareto_df = df[[category_col, value_col]].copy()
    pareto_df = pareto_df.groupby(category_col, as_index=False).sum()
    pareto_df = pareto_df.sort_values(by=value_col, ascending=False)
    pareto_df["Cumulative"] = pareto_df[value_col].cumsum()
    total = pareto_df[value_col].sum()
    pareto_df["Cumulative %"] = round(pareto_df["Cumulative"] / total * 100, 2)
    return pareto_df

def generate_sankey(df, source_col, target_col, value_col):
    """
    Generate a Sankey diagram to visualize data flows/relationships.
    """
    # Label encoding for Sankey
    all_nodes = list(pd.unique(df[[source_col, target_col]].values.ravel('K')))
    mapping = {k: v for v, k in enumerate(all_nodes)}

    sankey_data = {
        'source': df[source_col].map(mapping),
        'target': df[target_col].map(mapping),
        'value': df[value_col]
    }

    link = dict(
        source=sankey_data['source'],
        target=sankey_data['target'],
        value=sankey_data['value']
    )
    node = dict(
        label=all_nodes,
        pad=20,
        thickness=30,
        line=dict(color="black", width=0.5)
    )
    fig = go.Figure(data=[go.Sankey(link=link, node=node)])
    fig.update_layout(title_text="Sankey Diagram", font_size=10)
    return fig

def detect_outliers(df, value_col, threshold=1.5):
    """
    Simple outlier detection using IQR (Interquartile Range).
    Returns a DataFrame of potential outliers.
    """
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
    return outliers

def generate_cfo_commentary(df, openai_api_key, prompt_template=None):
    """
    Generate CFO-level commentary via OpenAI API.
    """
    openai.api_key = openai_api_key

    if prompt_template is None:
        prompt_template = (
            "You are a CFO. Provide a succinct commentary on the following data, "
            "highlighting key insights, major variances, outliers, and recommended actions.\n\n"
            "Data:\n{data}\n"
        )

    # Convert DataFrame to CSV-like text or condensed text
    data_snippet = df.to_string(index=False)
    
    prompt = prompt_template.format(data=data_snippet)
    
    # Make OpenAI API call (GPT-3.5/4 recommended)
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    commentary = response.choices[0].text.strip()
    return commentary

# -------------------------
# Streamlit App Layout
# -------------------------
def main():
    st.title("FP&A Data Analyzer")

    # -- Sidebar: Pro version
    st.sidebar.title("Upgrade to Pro")
    use_pro_version = st.sidebar.checkbox("Use Pro Version (OpenAI API key required)")
    openai_api_key = None
    if use_pro_version:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    # -- File upload
    st.sidebar.title("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        st.write("## Uploaded Data Preview")
        st.dataframe(df.head())

        # -- Variance Analysis
        st.write("---")
        st.write("### Variance Analysis")

        # For demonstration, assume columns are "Category", "Actual", "Budget"
        # Adjust as per your real dataset
        if "Actual" in df.columns and "Budget" in df.columns:
            df_var = compute_variances(df, actual_col="Actual", budget_col="Budget")
            variance_table = generate_variance_table(df_var)
            st.dataframe(variance_table)
        else:
            st.warning("Data must have 'Actual' and 'Budget' columns to perform variance analysis.")

        # -- Charts Section
        st.write("---")
        st.write("### Visualizations")

        # Bar Chart
        if "Category" in df.columns and "Variance" in df_var.columns:
            bar_fig = generate_bar_chart(df_var, "Category", "Variance")
            st.plotly_chart(bar_fig, use_container_width=True)

        # Line Chart
        # For demonstration, assume we have a "Time" column
        if "Time" in df.columns and "Actual" in df.columns:
            line_fig = generate_line_chart(df, "Time", "Actual", color_col="Category" if "Category" in df.columns else None)
            st.plotly_chart(line_fig, use_container_width=True)
        else:
            st.info("Add a 'Time' column in your dataset to see line charts of Actual data over Time.")

        # -- Advanced Analysis
        st.write("---")
        st.write("### Advanced Analysis")
        analysis_options = ["None", "Pareto Analysis", "Sankey Diagram", "Outliers Analysis"]
        selected_analysis = st.selectbox("Choose an advanced analysis method:", analysis_options)

        if selected_analysis == "Pareto Analysis":
            if "Category" in df.columns and "Variance" in df_var.columns:
                pareto_df = perform_pareto_analysis(df_var, "Category", "Variance")
                st.subheader("Pareto Analysis Results")
                st.dataframe(pareto_df)

                # Quick Pareto chart
                fig = px.bar(
                    pareto_df,
                    x="Category",
                    y="Variance",
                    title="Pareto Chart of Variance",
                    template="plotly_white"
                )
                fig.update_layout(xaxis_title="Category", yaxis_title="Variance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Your data must have 'Category' and 'Variance' columns for Pareto Analysis.")

        elif selected_analysis == "Sankey Diagram":
            # For demonstration, we assume the dataset has "Source", "Target", and "Value" columns
            if set(["Source", "Target", "Value"]).issubset(df.columns):
                sankey_fig = generate_sankey(df, "Source", "Target", "Value")
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.warning("Your data must have 'Source', 'Target', and 'Value' columns for Sankey Diagram.")

        elif selected_analysis == "Outliers Analysis":
            # For demonstration, use "Variance" as the column to detect outliers
            if "Variance" in df_var.columns:
                outliers_df = detect_outliers(df_var, "Variance", threshold=1.5)
                if not outliers_df.empty:
                    st.subheader("Potential Outliers")
                    st.dataframe(outliers_df)
                else:
                    st.info("No outliers detected based on the chosen threshold.")
            else:
                st.warning("Your data must have a 'Variance' column for outlier detection.")

        # -- Pro Version: Generate CFO Commentary
        if use_pro_version and openai_api_key:
            st.write("---")
            st.write("### CFO-Level Commentary")
            if st.button("Generate Commentary"):
                try:
                    commentary = generate_cfo_commentary(df_var, openai_api_key)
                    st.success("CFO Commentary Generated Successfully!")
                    st.text_area("CFO Commentary", value=commentary, height=200)
                except Exception as e:
                    st.error(f"Error generating commentary: {e}")
        elif use_pro_version and not openai_api_key:
            st.warning("Please enter your OpenAI API key to generate CFO commentary.")

    else:
        st.info("Please upload an Excel file in the sidebar to begin analysis.")

if __name__ == "__main__":
    main()

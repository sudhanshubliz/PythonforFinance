import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Function to generate synthetic FP&A-style data
# with multiple dimensions: date, region, product, revenue, cost, profit, etc.

@st.cache_data
def generate_synthetic_data(num_records=500):
    np.random.seed(42)

    dates = pd.date_range(start='2022-01-01', periods=num_records // 5, freq='D')
    dates = np.random.choice(dates, size=num_records)
    regions = np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Africa'], size=num_records)
    products = np.random.choice(['Product A', 'Product B', 'Product C', 'Product D', 'Product E'], size=num_records)

    revenue = np.random.normal(loc=100000, scale=20000, size=num_records)
    cost = revenue * np.random.uniform(0.5, 0.8, size=num_records)
    profit = revenue - cost
    yoy_growth = np.random.normal(loc=0.05, scale=0.03, size=num_records)
    yoy_growth_percent = yoy_growth * 100

    df = pd.DataFrame({
        'Date': dates,
        'Region': regions,
        'Product': products,
        'Revenue': revenue,
        'Cost': cost,
        'Profit': profit,
        'YoY Growth': yoy_growth,
        'YoY Growth %': yoy_growth_percent
    })
    # Make sure Date is sorted
    df = df.sort_values(by='Date').reset_index(drop=True)

    return df


def create_waterfall_chart(df):
    # Summarize data for waterfall
    summary_df = df[['Revenue', 'Cost', 'Profit']].sum()
    # We'll do a bridging from Revenue to Profit
    base = summary_df['Revenue']
    steps = [
        dict(label='Revenue', value=summary_df['Revenue']),
        dict(label='Cost', value=-summary_df['Cost']),
        dict(label='Profit', value=summary_df['Profit'])
    ]
    fig = go.Figure(go.Waterfall(
        measure=['relative', 'relative', 'total'],
        x=[step['label'] for step in steps],
        y=[step['value'] for step in steps],
        connector=dict(line=dict(color='rgb(63, 63, 63)'))
    ))
    fig.update_layout(title='Waterfall Chart (Revenue -> Profit)', waterfallgap=0.2)
    st.plotly_chart(fig)


def create_tornado_chart(df):
    # Tornado charts are often used for sensitivity analysis.
    # We'll simulate sensitivity of Profit to changes in Revenue and Cost.
    # For demonstration, let's just pick a sample of changes.

    # We can define a baseline profit and some changes in revenue/cost.
    baseline_profit = df['Profit'].mean()
    revenue_changes = [0.1, 0.05, -0.05, -0.1]
    cost_changes = [0.1, 0.05, -0.05, -0.1]

    scenarios = []
    for r in revenue_changes:
        for c in cost_changes:
            new_rev = df['Revenue'].mean() * (1 + r)
            new_cost = df['Cost'].mean() * (1 + c)
            new_profit = new_rev - new_cost
            scenarios.append((r, c, new_profit))

    scenario_df = pd.DataFrame(scenarios, columns=['RevenueChange', 'CostChange', 'NewProfit'])

    # We'll pivot to visualize in a bar chart that looks like a tornado
    scenario_df['Label'] = scenario_df.apply(
        lambda row: f"Rev: {row['RevenueChange']*100:.0f}%, Cost: {row['CostChange']*100:.0f}%", axis=1
    )
    scenario_df.sort_values('NewProfit', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=scenario_df['Label'],
        x=scenario_df['NewProfit'] - baseline_profit,
        orientation='h',
        marker_color='blue'
    ))
    fig.update_layout(
        title='Tornado Chart: Profit Sensitivity to Revenue/Cost Changes',
        xaxis_title='Profit Difference from Baseline',
        yaxis_title='Scenario'
    )
    st.plotly_chart(fig)


def create_marimekko_chart(df):
    # Marimekko charts typically show 2D categorical data in blocks sized by measure.
    # We'll do region vs product, sized by total revenue.

    group_df = df.groupby(['Region', 'Product'])['Revenue'].sum().reset_index()
    total_revenue = group_df['Revenue'].sum()
    group_df['RevenuePct'] = group_df['Revenue'] / total_revenue

    # For a marimekko, we need cumulative distribution along x for region, y for product.
    # This requires a bit of manual arrangement.

    # We'll pivot to get region totals and product totals
    region_totals = df.groupby('Region')['Revenue'].sum()
    product_totals = df.groupby('Product')['Revenue'].sum()

    # Sort regions/products by total for a more structured look
    region_sorted = region_totals.sort_values(ascending=False).index.tolist()
    product_sorted = product_totals.sort_values(ascending=False).index.tolist()

    # We'll create a figure manually using shapes or try to replicate marimekko style.
    # For simplicity, let's do a treemap, which is a close alternative.

    fig = px.treemap(group_df, path=['Region', 'Product'], values='Revenue', title='Marimekko-Style Treemap')
    st.plotly_chart(fig)


def create_sankey_chart(df):
    # Sankey for money flow across Region -> Product or similar.

    region_totals = df.groupby('Region')['Revenue'].sum().reset_index()
    product_totals = df.groupby('Product')['Revenue'].sum().reset_index()

    # We'll connect region -> product with revenue as flow.
    region_product = df.groupby(['Region', 'Product'])['Revenue'].sum().reset_index()

    # Construct sankey node and link data
    regions_list = region_totals['Region'].unique().tolist()
    products_list = product_totals['Product'].unique().tolist()

    all_nodes = regions_list + products_list
    node_dict = {name: idx for idx, name in enumerate(all_nodes)}

    # Build links
    links = []
    for _, row in region_product.iterrows():
        source = node_dict[row['Region']]
        target = node_dict[row['Product']]
        value = row['Revenue']
        links.append(dict(source=source, target=target, value=value))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes
        ),
        link=dict(
            source=[l['source'] for l in links],
            target=[l['target'] for l in links],
            value=[l['value'] for l in links]
        )
    )])

    fig.update_layout(title_text="Sankey Chart: Region to Product Revenue Flow", font_size=10)
    st.plotly_chart(fig)


def create_bullet_chart(df):
    # We'll do a bullet chart comparing actual vs target revenue for each region.
    region_df = df.groupby('Region')['Revenue'].sum().reset_index()
    region_df['Target'] = region_df['Revenue'] * np.random.uniform(1.1, 1.3, size=len(region_df))  # hypothetical targets

    fig = go.Figure()

    for _, row in region_df.iterrows():
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=row['Revenue'],
                delta={'reference': row['Target']},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [None, row['Target']*1.5]},
                    'bar': {'color': "blue"}
                },
                domain={'x': [0.0, 1.0], 'y': [0.0, 0.2]},
                title={'text': row['Region']}
            )
        )
    fig.update_layout(height=3000, margin=dict(l=100, r=100, t=100, b=100), title_text="Bullet Charts by Region")
    st.plotly_chart(fig)


def create_correlation_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
    st.plotly_chart(fig)


def create_forecast_variance_chart(df):
    # We'll simulate actual vs forecast.
    # Let's assume the forecast is slightly different.
    actual = df.groupby('Date')['Revenue'].sum().reset_index(name='Actual')
    forecast = actual.copy()
    forecast['Forecast'] = forecast['Actual'] * np.random.uniform(0.9, 1.1, size=len(forecast))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['Date'], y=actual['Actual'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Forecast'], mode='lines+markers', name='Forecast'))

    fig.update_layout(title='Actual vs Forecast Revenue Over Time')
    st.plotly_chart(fig)


def create_time_series_decomposition(df):
    # We can do a rolling average or decomposition style chart.
    # We'll just do a rolling average demonstration for the time being.
    ts = df.groupby('Date')['Revenue'].sum().sort_index()
    ts = ts.asfreq('D').fillna(method='ffill')
    window_size = 7  # weekly
    ts_rolling = ts.rolling(window_size).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name='Daily Revenue', mode='lines'))
    fig.add_trace(go.Scatter(x=ts_rolling.index, y=ts_rolling.values, name=f'{window_size}-Day Rolling Avg', mode='lines'))
    fig.update_layout(title='Time Series with Rolling Average')
    st.plotly_chart(fig)


def create_sunburst_chart(df):
    # A sunburst chart can show hierarchical data, similar to treemap.
    fig = px.sunburst(
        df,
        path=['Region', 'Product'],
        values='Revenue',
        title='Sunburst Chart by Region and Product'
    )
    st.plotly_chart(fig)


def main():
    st.title('Top 10 Advanced Data Visualizations for FP&A')
    df = generate_synthetic_data()

    viz_options = [
        'Waterfall Chart',
        'Tornado Chart',
        'Marimekko (Treemap)',
        'Sankey Chart',
        'Bullet Chart',
        'Correlation Heatmap',
        'Actual vs Forecast Variance Chart',
        'Time Series Rolling Avg (Decomposition)',
        'Sunburst Chart',
        'All in One'
    ]

    choice = st.sidebar.selectbox('Select a Visualization', viz_options)

    if choice == 'Waterfall Chart':
        create_waterfall_chart(df)
    elif choice == 'Tornado Chart':
        create_tornado_chart(df)
    elif choice == 'Marimekko (Treemap)':
        create_marimekko_chart(df)
    elif choice == 'Sankey Chart':
        create_sankey_chart(df)
    elif choice == 'Bullet Chart':
        create_bullet_chart(df)
    elif choice == 'Correlation Heatmap':
        create_correlation_heatmap(df)
    elif choice == 'Actual vs Forecast Variance Chart':
        create_forecast_variance_chart(df)
    elif choice == 'Time Series Rolling Avg (Decomposition)':
        create_time_series_decomposition(df)
    elif choice == 'Sunburst Chart':
        create_sunburst_chart(df)
    else:
        st.subheader('Waterfall Chart')
        create_waterfall_chart(df)
        st.subheader('Tornado Chart')
        create_tornado_chart(df)
        st.subheader('Marimekko (Treemap)')
        create_marimekko_chart(df)
        st.subheader('Sankey Chart')
        create_sankey_chart(df)
        st.subheader('Bullet Chart')
        create_bullet_chart(df)
        st.subheader('Correlation Heatmap')
        create_correlation_heatmap(df)
        st.subheader('Actual vs Forecast Variance Chart')
        create_forecast_variance_chart(df)
        st.subheader('Time Series Rolling Avg (Decomposition)')
        create_time_series_decomposition(df)
        st.subheader('Sunburst Chart')
        create_sunburst_chart(df)

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px

def monte_carlo_simulation(
    initial_investment=100000,
    expected_return=0.10,
    return_std=0.15,
    time_horizon=10,
    num_simulations=10000,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random returns
    random_returns = np.random.normal(loc=expected_return, scale=return_std, 
                                      size=(time_horizon, num_simulations))
    
    # Calculate investment paths
    investment_paths = initial_investment * np.cumprod(1 + random_returns, axis=0)
    
    results_df = pd.DataFrame(investment_paths, 
                              index=range(1, time_horizon+1),
                              columns=[f"Simulation_{i+1}" for i in range(num_simulations)])
    final_values = results_df.iloc[-1].values
    return results_df, final_values

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Monte Carlo Investment Simulator"),
    html.Div([
        html.Div([
            html.Label("Initial Investment:"),
            dcc.Input(id='initial_investment', type='number', value=50000, step=1000),
        ], style={'padding': '10px', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Expected Return (decimal):"),
            dcc.Input(id='expected_return', type='number', value=0.12, step=0.01),
        ], style={'padding': '10px', 'display': 'inline-block'}),

        html.Div([
            html.Label("Return Std. Dev. (decimal):"),
            dcc.Input(id='return_std', type='number', value=0.20, step=0.01),
        ], style={'padding': '10px', 'display': 'inline-block'}),

        html.Div([
            html.Label("Time Horizon (periods):"),
            dcc.Input(id='time_horizon', type='number', value=5, step=1),
        ], style={'padding': '10px', 'display': 'inline-block'}),

        html.Div([
            html.Label("Number of Simulations:"),
            dcc.Input(id='num_simulations', type='number', value=2000, step=1000),
        ], style={'padding': '10px', 'display': 'inline-block'}),

        html.Div([
            html.Label("Random Seed (optional):"),
            dcc.Input(id='seed', type='number', value=42, step=1),
        ], style={'padding': '10px', 'display': 'inline-block'})
    ]),
    html.Hr(),
    
    # Statistics
    html.Div(id='stats-output', style={'margin': '20px 0'}),
    
    # Graphs
    html.Div([
        html.H3("Distribution of Final Investment Values"),
        dcc.Graph(id='final-distribution-graph'),
    ]),
    
    html.Div([
        html.H3("Sample of Simulation Paths"),
        dcc.Graph(id='simulation-paths-graph'),
    ]),
    
    html.Div([
        html.H3("Distribution of Values at Each Period"),
        dcc.Graph(id='boxplot-graph'),
    ])
], style={'width': '80%', 'margin': 'auto'})

@app.callback(
    [Output('stats-output', 'children'),
     Output('final-distribution-graph', 'figure'),
     Output('simulation-paths-graph', 'figure'),
     Output('boxplot-graph', 'figure')],
    [Input('initial_investment', 'value'),
     Input('expected_return', 'value'),
     Input('return_std', 'value'),
     Input('time_horizon', 'value'),
     Input('num_simulations', 'value'),
     Input('seed', 'value')]
)
def update_simulation(initial_investment, expected_return, return_std, time_horizon, num_simulations, seed):
    if None in (initial_investment, expected_return, return_std, time_horizon, num_simulations):
        return ["Please provide all inputs.", {}, {}, {}]
    
    results_df, final_values = monte_carlo_simulation(
        initial_investment=initial_investment,
        expected_return=expected_return,
        return_std=return_std,
        time_horizon=time_horizon,
        num_simulations=num_simulations,
        seed=seed
    )
    
    mean_final = np.mean(final_values)
    median_final = np.median(final_values)
    std_final = np.std(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)
    
    # Stats display
    stats = html.Div([
        html.H3("Final Value Statistics"),
        html.P(f"Mean Final Value: ${mean_final:,.2f}"),
        html.P(f"Median Final Value: ${median_final:,.2f}"),
        html.P(f"Standard Deviation: ${std_final:,.2f}"),
        html.P(f"5th Percentile: ${percentile_5:,.2f}"),
        html.P(f"95th Percentile: ${percentile_95:,.2f}")
    ])
    
    # Distribution of final values - using a histogram
    fig_dist = px.histogram(final_values, nbins=50, marginal="rug", opacity=0.75)
    fig_dist.update_layout(
        title="Distribution of Final Investment Values",
        xaxis_title="Final Value",
        yaxis_title="Count"
    )
    fig_dist.add_vline(x=mean_final, line_dash='dash', line_color='red', 
                       annotation_text=f"Mean: ${mean_final:,.0f}", annotation_position="top right")
    fig_dist.add_vline(x=median_final, line_dash='dash', line_color='green', 
                       annotation_text=f"Median: ${median_final:,.0f}", annotation_position="bottom left")
    
    # Simulation paths (just a subset to avoid clutter)
    subset_size = min(50, num_simulations)
    fig_paths = go.Figure()
    for col in results_df.columns[:subset_size]:
        fig_paths.add_trace(go.Scatter(x=results_df.index, y=results_df[col],
                                       mode='lines', line=dict(width=1), opacity=0.4,
                                       showlegend=False))
    fig_paths.update_layout(
        title="Sample of Monte Carlo Simulation Paths Over Time",
        xaxis_title="Period",
        yaxis_title="Portfolio Value"
    )
    
    # Boxplot by period
    # Transpose results so each period is a column in a long format
    long_df = results_df.stack().reset_index()
    long_df.columns = ["Period", "Simulation", "Value"]
    fig_box = px.box(long_df, x="Value", y="Period", orientation="h")
    fig_box.update_layout(
        title="Distribution of Values at Each Period",
        xaxis_title="Portfolio Value",
        yaxis_title="Period"
    )
    
    return stats, fig_dist, fig_paths, fig_box

if __name__ == '__main__':
    app.run_server(debug=True)

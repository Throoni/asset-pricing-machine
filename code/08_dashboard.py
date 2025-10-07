#!/usr/bin/env python3
"""
Interactive Dashboard for Asset Pricing Machine

A Dash app that visualizes pipeline outputs and validation results.
Provides live data reloading and comprehensive visualization of:
- Summary metrics and validation results
- Security Market Line (SML) plots
- Efficient frontier and portfolio analysis
- Beta distributions and statistics
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Asset Pricing Machine Dashboard"

def load_all():
    """Load all data files safely, returning None for missing files."""
    data = {}
    
    files = {
        'fmb_results': 'output/tables/fmb_results.csv',
        'zero_beta_portfolio': 'output/tables/zero_beta_portfolio.csv',
        'optimizer_weights': 'output/tables/optimizer_weights.csv',
        'betas': 'output/tables/betas.csv',
        'vw_beta_summary': 'output/tables/vw_beta_summary.csv',
        'returns': 'data/processed/returns.parquet'
    }
    
    for key, filepath in files.items():
        path = Path(filepath)
        if path.exists():
            try:
                if filepath.endswith('.parquet'):
                    data[key] = pd.read_parquet(path)
                else:
                    data[key] = pd.read_csv(path)
                logger.info(f"Loaded {key}: {len(data[key])} rows")
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
                data[key] = None
        else:
            logger.info(f"File not found: {filepath}")
            data[key] = None
    
    return data

def create_placeholder_figure(message="Data not available"):
    """Create a placeholder figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        plot_bgcolor='white'
    )
    return fig

def create_summary_cards(data):
    """Create summary metric cards."""
    cards = []
    
    # Sharpe ratio from optimizer weights
    if data.get('optimizer_weights') is not None:
        ow_df = data['optimizer_weights']
        tangency_row = ow_df[ow_df['portfolio'].str.contains('tangency', case=False, na=False)]
        if not tangency_row.empty:
            sharpe = tangency_row.iloc[0]['sharpe_or_slope']
            cards.append(html.Div([
                html.H3(f"{sharpe:.3f}", style={'color': '#2E8B57', 'margin': 0}),
                html.P("Tangency Sharpe", style={'margin': 0, 'fontSize': 14})
            ], className="metric-card"))
    
    # R_Z from zero-beta portfolio
    if data.get('zero_beta_portfolio') is not None:
        zb_df = data['zero_beta_portfolio']
        if 'R_Z' in zb_df.columns:
            rz = zb_df['R_Z'].iloc[0]
            cards.append(html.Div([
                html.H3(f"{rz:.3f}", style={'color': '#4169E1', 'margin': 0}),
                html.P("Zero-Beta Rate", style={'margin': 0, 'fontSize': 14})
            ], className="metric-card"))
    
    # Number of tickers
    if data.get('betas') is not None:
        n_tickers = data['betas']['ticker'].nunique()
        cards.append(html.Div([
            html.H3(f"{n_tickers}", style={'color': '#FF6347', 'margin': 0}),
            html.P("Tickers", style={'margin': 0, 'fontSize': 14})
        ], className="metric-card"))
    
    # Sample period
    if data.get('returns') is not None:
        returns_df = data['returns']
        if 'date' in returns_df.columns:
            start_date = returns_df['date'].min().strftime('%Y-%m')
            end_date = returns_df['date'].max().strftime('%Y-%m')
            cards.append(html.Div([
                html.H3(f"{start_date} to {end_date}", style={'color': '#9370DB', 'margin': 0}),
                html.P("Sample Period", style={'margin': 0, 'fontSize': 14})
            ], className="metric-card"))
    
    return cards

def create_sml_plot(data):
    """Create Security Market Line plot."""
    if data.get('fmb_results') is None or data.get('betas') is None:
        return create_placeholder_figure("SML data not available")
    
    fmb_df = data['fmb_results']
    betas_df = data['betas']
    
    # Get static results for plotting
    static_excess = fmb_df[fmb_df['method'] == 'static_excess']
    static_raw = fmb_df[fmb_df['method'] == 'static_raw']
    
    if static_excess.empty and static_raw.empty:
        return create_placeholder_figure("No static regression results available")
    
    # Use excess returns if available, otherwise raw
    use_excess = not static_excess.empty
    static_results = static_excess if use_excess else static_raw
    
    # Get average returns and betas
    if use_excess and data.get('returns') is not None:
        returns_df = data['returns']
        if 'ret_excess' in returns_df.columns and 'ticker' in returns_df.columns:
            avg_returns = returns_df.groupby('ticker')['ret_excess'].mean()
            y_label = "Average Excess Return"
        else:
            avg_returns = returns_df.groupby('ticker')['ret'].mean()
            y_label = "Average Return"
    else:
        avg_returns = returns_df.groupby('ticker')['ret'].mean()
        y_label = "Average Return"
    
    # Merge with betas
    plot_data = []
    for model_type in betas_df['model_type'].unique():
        model_betas = betas_df[betas_df['model_type'] == model_type]
        for _, row in model_betas.iterrows():
            ticker = row['ticker']
            beta = row['beta']
            if ticker in avg_returns.index:
                plot_data.append({
                    'ticker': ticker,
                    'beta': beta,
                    'avg_return': avg_returns[ticker],
                    'model_type': model_type
                })
    
    if not plot_data:
        return create_placeholder_figure("No data available for SML plot")
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create scatter plot
    fig = px.scatter(plot_df, x='beta', y='avg_return', 
                     color='model_type', 
                     hover_data=['ticker'],
                     title="Security Market Line",
                     labels={'beta': 'Beta', 'avg_return': y_label})
    
    # Add fitted line
    gamma0 = static_results.iloc[0]['gamma0']
    gamma_m = static_results.iloc[0]['gamma_m']
    
    beta_range = np.linspace(plot_df['beta'].min(), plot_df['beta'].max(), 100)
    fitted_line = gamma0 + gamma_m * beta_range
    
    fig.add_trace(go.Scatter(
        x=beta_range, y=fitted_line,
        mode='lines',
        name=f'Fitted Line (γ₀={gamma0:.3f}, γₘ={gamma_m:.3f})',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title_x=0.5,
        showlegend=True,
        height=500
    )
    
    return fig

def create_frontier_plot(data):
    """Create efficient frontier plot."""
    if data.get('optimizer_weights') is None:
        return create_placeholder_figure("Portfolio data not available")
    
    ow_df = data['optimizer_weights']
    
    # Create scatter plot of portfolios
    fig = go.Figure()
    
    # Add portfolios
    for _, row in ow_df.iterrows():
        portfolio = row['portfolio']
        expected_return = row['expected_return']
        volatility = row['volatility']
        sharpe = row.get('sharpe_or_slope', 0)
        
        # Color by Sharpe ratio
        color = 'red' if sharpe < 0.5 else 'orange' if sharpe < 0.8 else 'green'
        
        fig.add_trace(go.Scatter(
            x=[volatility], y=[expected_return],
            mode='markers+text',
            text=[portfolio],
            textposition="top center",
            marker=dict(size=15, color=color),
            name=portfolio,
            hovertemplate=f"<b>{portfolio}</b><br>" +
                         f"Return: {expected_return:.3f}<br>" +
                         f"Volatility: {volatility:.3f}<br>" +
                         f"Sharpe: {sharpe:.3f}<extra></extra>"
        ))
    
    # Add efficient frontier line if we have enough points
    if len(ow_df) >= 2:
        # Sort by volatility for frontier line
        sorted_df = ow_df.sort_values('volatility')
        fig.add_trace(go.Scatter(
            x=sorted_df['volatility'],
            y=sorted_df['expected_return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        title_x=0.5,
        height=500,
        showlegend=True
    )
    
    return fig

def create_beta_histogram(data):
    """Create beta distribution histogram."""
    if data.get('betas') is None:
        return create_placeholder_figure("Beta data not available")
    
    betas_df = data['betas']
    
    # Create histogram for each model type
    fig = go.Figure()
    
    for model_type in betas_df['model_type'].unique():
        model_betas = betas_df[betas_df['model_type'] == model_type]['beta'].dropna()
        
        fig.add_trace(go.Histogram(
            x=model_betas,
            name=model_type,
            opacity=0.7,
            nbinsx=20
        ))
    
    fig.update_layout(
        title="Beta Distribution by Model",
        xaxis_title="Beta",
        yaxis_title="Frequency",
        title_x=0.5,
        height=500,
        barmode='overlay'
    )
    
    return fig

def create_validation_plot(data):
    """Create validation comparison plot."""
    if data.get('fmb_results') is None:
        return create_placeholder_figure("Validation data not available")
    
    fmb_df = data['fmb_results']
    
    # Create comparison plot for gamma_m
    methods = fmb_df['method'].tolist()
    gamma_m_values = fmb_df['gamma_m'].tolist()
    t_values = fmb_df['t_gamma_m'].tolist()
    
    # Color by significance
    colors = ['red' if abs(t) < 1.96 else 'green' for t in t_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=gamma_m_values,
        marker_color=colors,
        text=[f"t={t:.2f}" for t in t_values],
        textposition='auto',
        name="γₘ (Market Risk Premium)"
    ))
    
    # Add significance line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Fama-MacBeth Consistency Check",
        xaxis_title="Method",
        yaxis_title="γₘ (Market Risk Premium)",
        title_x=0.5,
        height=500
    )
    
    return fig

# Define app layout
app.layout = html.Div([
    html.H1("Asset Pricing Machine Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Summary cards
    html.Div(id='summary-cards', className="summary-cards"),
    
    # Reload button
    html.Div([
        html.Button('🔄 Reload Data', id='reload-button', n_clicks=0,
                   style={'margin': '20px', 'padding': '10px 20px', 'fontSize': 16})
    ], style={'textAlign': 'center'}),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='sml', children=[
        dcc.Tab(label='SML', value='sml'),
        dcc.Tab(label='Frontier', value='frontier'),
        dcc.Tab(label='Betas', value='betas'),
        dcc.Tab(label='Validation', value='validation')
    ]),
    
    # Tab content
    html.Div(id='tab-content'),
    
    # Store for data
    dcc.Store(id='data-store')
])

# CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .summary-cards {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 20px 0;
                flex-wrap: wrap;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .metric-card h3 {
                font-size: 2em;
                margin: 0;
                font-weight: bold;
            }
            .metric-card p {
                margin: 5px 0 0 0;
                font-size: 0.9em;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callbacks
@app.callback(
    Output('data-store', 'data'),
    Input('reload-button', 'n_clicks')
)
def load_data(n_clicks):
    """Load data when reload button is clicked."""
    return load_all()

@app.callback(
    Output('summary-cards', 'children'),
    Input('data-store', 'data')
)
def update_summary_cards(data):
    """Update summary cards when data changes."""
    if data is None:
        return [html.Div("Click 'Reload Data' to load metrics", 
                        style={'textAlign': 'center', 'color': 'gray'})]
    return create_summary_cards(data)

@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    Input('data-store', 'data')
)
def update_tab_content(active_tab, data):
    """Update tab content based on selected tab and data."""
    if data is None:
        return html.Div("Click 'Reload Data' to load visualizations", 
                       style={'textAlign': 'center', 'color': 'gray', 'marginTop': 50})
    
    if active_tab == 'sml':
        return dcc.Graph(figure=create_sml_plot(data))
    elif active_tab == 'frontier':
        return dcc.Graph(figure=create_frontier_plot(data))
    elif active_tab == 'betas':
        return dcc.Graph(figure=create_beta_histogram(data))
    elif active_tab == 'validation':
        return dcc.Graph(figure=create_validation_plot(data))
    
    return html.Div("Select a tab to view visualizations")

def main():
    """Main function to run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Asset Pricing Machine Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    args = parser.parse_args()
    
    print("🚀 Starting Asset Pricing Machine Dashboard...")
    print(f"📊 Dashboard will be available at: http://127.0.0.1:{args.port}")
    print("🔄 Use the 'Reload Data' button to refresh visualizations")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    app.run(debug=False, host='127.0.0.1', port=args.port)

if __name__ == '__main__':
    main()

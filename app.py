# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Initialize the app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div(children=[
    html.H1('Financial Risk Simulator'),

    html.Div([
        html.Label('Investment Amount ($):'),
        dcc.Input(id='investment-amount', value=10000, type='number')
    ]),

    html.Div([
        html.Label('Expected Annual Interest Rate (%):'),
        dcc.Input(id='interest-rate', value=5, type='number')
    ]),

    html.Div([
        html.Label('Expected Annual Inflation Rate (%):'),
        dcc.Input(id='inflation-rate', value=2, type='number')
    ]),

    html.Div([
        html.Label('Investment Duration (Years):'),
        dcc.Input(id='investment-years', value=10, type='number')
    ]),

    html.Div([
        html.Label('Investment Type:'),
        dcc.Dropdown(
            id='investment-type',
            options=[
                {'label': 'Low Risk (Bonds)', 'value': 'low'},
                {'label': 'Medium Risk (Index Funds)', 'value': 'medium'},
                {'label': 'High Risk (Stocks)', 'value': 'high'}
            ],
            value='medium'
        )
    ]),

    html.Button('Run Simulation', id='run-simulation', n_clicks=0),

    html.Hr(),

    html.Div(id='simulation-results'),

    dcc.Graph(id='histogram-chart'),

    dcc.Graph(id='line-chart')
])

# Callback to update the simulation results
@app.callback(
    [Output('simulation-results', 'children'),
     Output('histogram-chart', 'figure'),
     Output('line-chart', 'figure')],
    [Input('run-simulation', 'n_clicks')],
    [dash.dependencies.State('investment-amount', 'value'),
     dash.dependencies.State('interest-rate', 'value'),
     dash.dependencies.State('inflation-rate', 'value'),
     dash.dependencies.State('investment-years', 'value'),
     dash.dependencies.State('investment-type', 'value')]
)
def update_simulation(n_clicks, investment_amount, interest_rate, inflation_rate, investment_years, investment_type):
    if n_clicks > 0:
        # Adjust interest variation based on investment type
        if investment_type == 'low':
            interest_variation = 0.01  # +/- 1%
        elif investment_type == 'medium':
            interest_variation = 0.03  # +/- 3%
        elif investment_type == 'high':
            interest_variation = 0.05  # +/- 5%
        else:
            interest_variation = 0.03  # Default to medium

        # Number of simulations
        num_simulations = 1000

        # Initialize array to store simulation results
        simulation_results = []

        # Perform Monte Carlo Simulations
        for _ in range(num_simulations):
            simulated_value = investment_amount
            annual_values = [simulated_value]
            for _ in range(int(investment_years)):
                # Simulate annual interest rate variation
                annual_interest = np.random.uniform(
                    (interest_rate / 100) - interest_variation,
                    (interest_rate / 100) + interest_variation
                )
                simulated_value *= (1 + annual_interest)
                annual_values.append(simulated_value)
            simulation_results.append({
                'final_value': simulated_value,
                'annual_values': annual_values
            })

        # Create a DataFrame from the simulation results
        df_results = pd.DataFrame({
            'Final Value': [result['final_value'] for result in simulation_results]
        })

        # Calculate statistics
        average_result = df_results['Final Value'].mean()
        min_result = df_results['Final Value'].min()
        max_result = df_results['Final Value'].max()

        # Prepare the annual values for line chart
        df_annual = pd.DataFrame(
            [result['annual_values'] for result in simulation_results]
        ).transpose()
        df_annual['Year'] = df_annual.index

        # Build the output components
        result_text = html.Div([
            html.H4(f"After {investment_years} years:"),
            html.P(f"Average Future Value: ${average_result:,.2f}"),
            html.P(f"Minimum Future Value: ${min_result:,.2f}"),
            html.P(f"Maximum Future Value: ${max_result:,.2f}")
        ])

        # Create histogram chart
        hist_fig = px.histogram(
            df_results,
            x='Final Value',
            nbins=50,
            title='Distribution of Final Investment Values',
            labels={'Final Value': 'Final Value ($)'}
        )

        # Enhance histogram with tooltips
        hist_fig.update_traces(hovertemplate='Final Value: $%{x:,.2f}<br>Frequency: %{y}')

        # Create line chart of annual values
        line_fig = px.line(
            df_annual,
            x='Year',
            y=df_annual.columns[:-1],  # Exclude the 'Year' column
            title='Investment Growth Over Time',
            labels={'value': 'Investment Value ($)', 'Year': 'Year'}
        )

        # Limit the number of lines to avoid clutter
        line_fig.for_each_trace(
            lambda trace: trace.update(visible='legendonly') if int(trace.name) > 10 else ()
        )

        # Enhance line chart with tooltips
        line_fig.update_traces(hovertemplate='Year: %{x}<br>Value: $%{y:,.2f}')

        return result_text, hist_fig, line_fig

    # Return empty components if the button hasn't been clicked
    return '', {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)

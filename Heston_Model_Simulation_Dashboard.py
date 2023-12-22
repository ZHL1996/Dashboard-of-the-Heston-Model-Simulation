# -*- coding: utf-8 -*-
"""
Created on 10.07.2023

@author: Zhuohang Li
"""
# import all needed libraries

import numpy as np
from dash import html, callback, Input, Output, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash

length_of_W = 20  # how detailed is each plot  20
t = np.arange(length_of_W)  # for the y-axes in the graph
calc_paths = 501
# how many paths are calculated, is exogen, could be made endogen

hist_paths = 200  # 200
hist_length_of_W = 252
"""
Two functions, to make an external random process, and to simulate Heston.
"""

W1 = np.random.normal(0, 1, size=(calc_paths, length_of_W))
W2 = np.random.normal(0, 1, size=(calc_paths, length_of_W))
W2h = np.random.normal(0, 1, size=(hist_paths, hist_length_of_W))
W1h = np.random.normal(0, 1, size=(hist_paths, hist_length_of_W))

"""
Simulation for the big histogram 
"""

def Heston_Simulation2(rho, r, sigma_0, theta, kappa, volvol, T, S0, num_paths):
    """
    Simulate stock price paths under the Heston model.

    Parameters:
        S0 (float): Initial stock price.
        r (float): Risk-free rate.
        T (float): Time to maturity.
        sigma_0 (float): Initial volatility (square root of variance).
        theta (float): Long-term average variance.
        kappa (float): Mean reversion speed of variance.
        volvol (float): Volatility of volatility.
        rho (float): Correlation between stock returns and variance returns.
        num_steps (int): Number of time steps in the simulation.
        num_paths (int): Number of paths to simulate.

    Returns:
        numpy.ndarray: Array of stock price paths with shape (num_paths, num_steps + 1).
    """
    num_steps = hist_length_of_W
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)

    # Initialize the arrays to store the stock price paths
    stock_paths = np.zeros((num_paths, num_steps + 1))

    # Set the initial stock prices
    stock_paths[:, 0] = S0

    # Generate the random normal variables for the stock returns and variance returns
    z1 = np.random.randn(num_paths, num_steps)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.randn(num_paths, num_steps)

    # Simulate the stock price paths using Euler's method
    for i in range(1, num_steps + 1):
        # Calculate the instantaneous variance and volatility at each step
        sigma_t = np.maximum(sigma_0 + kappa * (theta - sigma_0) * dt + volvol * np.sqrt(sigma_0) * sqrt_dt * z1[:, i - 1], 0)
        volatility_t = np.sqrt(sigma_t)

        # Calculate the stock returns and update the stock prices
        stock_returns = r * dt + volatility_t * sqrt_dt * z2[:, i - 1]
        stock_paths[:, i] = stock_paths[:, i - 1] * np.exp(stock_returns)

        # Update sigma_0 for the next step
        sigma_0 = sigma_t

    return stock_paths, sigma_0


# the filter doesnt catch all mistakes, it gets harder the shorter the Maturity
# and if the Strike is out of money

"""
Starting the layout of the page 
"""
all_options = {'no SAA ': [], 'SAA ': ['Underlying ', 'Bond ', 'Cash '] }

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout =  html.Div([
    #html.Img(src='assets/logo.jpg', width='400px', height='auto', alt='image', style={'display': 'block', 'margin': '0 auto', 'text-align': 'center'}),
    dbc.Container([
        dcc.Store(id='price-data'),
        html.Br(),
        html.Hr(),
        dbc.Row([
            html.Center(html.H4("Heston-Model Simulation")),
            html.Hr(),
            dbc.Col([
                dcc.Graph(
                    id='graph_output'
                ),
                html.Div(
                    id='call_output'
                ),
                html.Div(
                    id='put_output'
                ),
                html.Div(
                    id='idhowmanyoptions'
                ),
                html.Div(
                    id='heston_call_output'
                ),
                html.Div(
                    id='heston_put_output'
                ),
                html.Hr(),
                html.P('Shown Paths'),
                html.Div(
                    dcc.Input(
                        id='paths_input',
                        value=50,
                        type='number'
                    )
                ),
            ]),
            dbc.Col([
                html.P('Correlation between price and volatility'),
                html.Div(
                    dcc.Input(
                        id='rho_output',
                        value=-0.65
                    )
                ),
                dcc.Slider(
                    -0.9, 0.9, value=-0.65,
                    id='rho_input'
                ),
                html.P('risk-free rate'),
                html.Div(
                    dcc.Input(
                        id='my_output',
                        value=0.02
                    )
                ),
                dcc.Slider(
                    -0.02, 0.2, value=0.05,
                    id='my_input'
                ),
                html.P('initial volatility'),
                html.Div(
                    dcc.Input(
                        id='sigma_0_output',
                        value=0.2
                    )
                ),
                dcc.Slider(
                    0.05, 0.6, value=0.2,
                    id='sigma_0_input'
                ),
                html.P('long-term volatility'),
                html.Div(
                    dcc.Input(
                        id='theta_vol_output',
                        value=0.2
                    )
                ),
                dcc.Slider(
                    0.05, 0.8, value=0.2,
                    id='theta_vol_input'
                ),
                html.P('conversion speed towards the long-term volatility'),
                html.Div(
                    dcc.Input(
                        id='kappa_output', value=1.5
                    )
                ),
                dcc.Slider(
                    0, 5, value=1.5,
                    id='kappa_input'
                ),
                html.P('volatility of volatility'),
                html.Div(
                    dcc.Input(
                        id='volvol_output',
                        value=0.2
                    )
                ),
                dcc.Slider(
                    0.05, 0.8, value=0.2,
                    id='volvol_input'
                ),
                html.P('Maturity'),
                html.Div(
                    dcc.Input(
                        id='T_output',
                        value=1
                    )
                ),
                dcc.Slider(
                    0.1, 5, value=1,
                    id='T_input'
                ),
            ])
        ]),
        dbc.Row([
            html.Hr(),
            dbc.Col([
                html.H4('Performance')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.P('Selected Simulation'),
                html.Div(
                    dcc.Input(
                        id='path_number_input_2',
                        value=0,
                        type='number'
                    )
                ),
                html.Div([
                    html.Hr(),
                    html.Div(
                        id='last_value_performance'
                    )
                ]),
                dbc.Row([
                    dcc.Graph(
                        id='development_performance'
                    )
                ])
            ])
        ])
    ])
])

@app.callback(
    Output('rho_output', 'value'),
    Output('my_output', 'value'),
    Output('sigma_0_output', 'value'),
    Output('theta_vol_output', 'value'),
    Output('kappa_output', 'value'),
    Output('volvol_output', 'value'),
    Output('T_output', 'value'),
    Input('rho_input', 'value'),
    Input('my_input', 'value'),
    Input('sigma_0_input', 'value'),
    Input('theta_vol_input', 'value'),
    Input('kappa_input', 'value'),
    Input('volvol_input', 'value'),
    Input('T_input', 'value'),
)
def update_sliders(rho_input, my_input, sigma_0_input, theta_vol_input, kappa_input, volvol_input, T_input):
    return rho_input, my_input, sigma_0_input, theta_vol_input, kappa_input, volvol_input, T_input


@app.callback(
    Output('graph_output', 'figure'),
    Output('price-data', 'data'),
    Input('rho_output', 'value'),
    Input('my_output', 'value'),
    Input('sigma_0_output', 'value'),
    Input('theta_vol_output', 'value'),
    Input('kappa_output', 'value'),
    Input('volvol_output', 'value'),
    Input('T_output', 'value'),
    Input('paths_input', 'value'),
)
def update_graph_output(rho, my, sigma_0, theta_vol, kappa, volvol, T, paths):
    # Heston_Simulation returns price, call_claim, variance, and put_claim
    global price
    price = Heston_Simulation2(
        rho, r = my, sigma_0 = sigma_0, theta = theta_vol, kappa = kappa, volvol = volvol, T = T, S0=100,  num_paths = paths
    )[0]
    price[:,0] = 100
    # Create traces for each price path
    traces = []
    for i in range(min(paths, len(price))):
        # Calculate the time steps based on T_output
        time_steps = np.linspace(0, T, len(price[i]))
        trace = go.Scatter(
            x=time_steps,
            y=price[i],
            mode='lines',
            name=f'Path {i + 1}',
            showlegend=False,
        )
        traces.append(trace)
        # Calculate price returns for each path
        price_returns = [prices[-1] / prices[0] - 1 for prices in price]

        hist_trace = go.Histogram(
            x=price_returns,
            marker=dict(color='blue'),
            name='Price Returns',
            opacity=0.7,
            nbinsx=20,
        )
        # Set histnorm='probability' to normalize the histogram
        hist_trace.histnorm = 'probability'

        # Increase the number of bins for smoother appearance
        hist_trace.nbinsx = 50

        # Layout for the figure
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Heston Model Simulation', 'Returns Histogram'))

        # Add the main graph (line chart) to the first subplot
        for trace in traces:
            fig.add_trace(trace, row=1, col=1)

        # Add the histogram to the second subplot
        fig.add_trace(hist_trace, row=1, col=2)

        # Set x-axis title for the first subplot
        fig.update_xaxes(title_text='Time to Maturity', row=1, col=1)

        # Set x-axis title for the second subplot
        fig.update_xaxes(title_text='Returns', row=1, col=2)

        # Set y-axis title for both subplots
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)

        # Hide legend for the entire figure
        fig.update_layout(showlegend=False)

        # Set the overall figure layout
        fig.update_layout(
            title_text='Heston Model Simulation and Price Returns Histogram',
            height=500,
            width=1000,
        )
        return fig, price

@app.callback(
        Output('development_performance', 'figure'),
        Input('price-data', 'data'),
        Input('path_number_input_2', 'value'),
        Input('T_output', 'value')
    )
def path_selection(price, path2, T):
        path = price[path2 - 1]
        time_steps = np.linspace(0, T, len(path))
        trace = go.Scatter(
            x=time_steps,
            y=path,
            mode='lines',
            name=f'Path {path2 + 1}',
            showlegend=False,
        )
        # Layout for the figure
        fig2 = make_subplots(rows=1, cols=1)
        # Add the main graph (line chart) to the first subplot
        fig2.add_trace(trace, row=1, col=1)
        fig2.update_xaxes(title_text='Time to Maturity', row=1, col=1)
        fig2.update_yaxes(title_text='Price', row=1, col=1)
        fig2.update_layout(showlegend=False)
        fig2.update_layout(
            title_text='Selected Simulation',
            height=500,
            width=1000,
        )
        return fig2

    # if __name__ == '__main__':
    # app.run_server(debug=True, host='127.0.0.1', port=8051)
app.run_server(debug=True, port=8051)


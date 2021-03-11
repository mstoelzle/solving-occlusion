from collections import deque
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import random
import torch
from typing import *

from src.enums import *
from src.utils.log import get_logger

logger = get_logger("live_inference_plotter")


def plot_live_inference():
    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id='live-graph', animate=True),
            dcc.Interval(
                id='graph-update',
                interval=1000,
                n_intervals=0
            ),
        ]
    )

    X = deque(maxlen=20)
    X.append(1)

    Y = deque(maxlen=20)
    Y.append(1)

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_graph_scatter(n):
        X.append(X[-1] + 1)
        Y.append(Y[-1] + Y[-1] * random.uniform(-0.1, 0.1))

        data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode='lines+markers'
        )

        return {'data': [data],
                'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]), yaxis=dict(range=[min(Y), max(Y)]), )}

    app.run_server()

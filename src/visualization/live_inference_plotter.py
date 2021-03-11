from collections import deque
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import plotly
import plotly.graph_objects as go
import random
import time
import torch
from typing import *

from src.enums import *
from src.utils.log import get_logger

logger = get_logger("live_inference_plotter")


def plot_live_inference(purpose_hdf5_group: h5py.Group):
    data_hdf5_group = purpose_hdf5_group["data"]
    occ_dem_dataset = data_hdf5_group[ChannelEnum.OCC_DEM.value]
    comp_dem_dataset = data_hdf5_group[ChannelEnum.COMP_DEM.value]

    occ_dems = np.array(occ_dem_dataset)
    comp_dems = np.array(comp_dem_dataset)

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id='live-graph', animate=True),
            dcc.Interval(
                id='graph-update',
                interval=1000,
                n_intervals=0,
                max_intervals=comp_dem_dataset.shape[0]
            ),
        ]
    )

    @app.callback(
        Output('live-graph', 'figure'),
        [Input('graph-update', 'n_intervals')]
    )
    def update_graph_scatter(i):
        data_occ = plotly.graph_objs.Surface(
            z=occ_dems[i],
            name='Occluded DEM',
            colorscale="viridis",
            opacity=0.9,
            showscale=False
        )

        data_comp = plotly.graph_objs.Surface(
            z=comp_dems[i],
            name='Composed DEM',
            colorscale="viridis",
            colorbar={"len": 0.75, "lenmode": "fraction"},
            opacity=0.5,
            showscale=True
        )

        layout = go.Layout(title='Occluded & Composed DEM',  width=1000, height=900,
                           scene={"aspectratio": {"x": 1, "y": 1, "z": 0.3}})

        return {'data': [data_occ, data_comp], 'layout': layout}

    app.run_server()

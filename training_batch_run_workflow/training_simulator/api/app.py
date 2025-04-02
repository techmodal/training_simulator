import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import requests
# Import data from default.py
from default import DEFAULT_PARAMETERS

# Initialize Dash app
app = dash.Dash(__name__, requests_pathname_prefix='/dashboard/')
app.title = "Course Progression Model"

# Initial data parameters
def simulate_data(years, capacity, duration, attrition):
    print('aaa')
    progressing = [capacity]
    attrition_data = []
    hold = []

    for year in range(1, years + 1):
        dropped = progressing[-1] * attrition
        progressed = progressing[-1] - dropped
        holding = progressed * (1 / duration)

        attrition_data.append(dropped / capacity)
        hold.append(holding)
        progressing.append(progressed - holding)

    return progressing[:-1], attrition_data, hold

# Default simulation
years = 10
capacity_default = DEFAULT_PARAMETERS['pipeline']['stage1']['capacity_progressing']
attrition_default = DEFAULT_PARAMETERS['pipeline']['stage1']['drop_out_progressing']
duration_default = DEFAULT_PARAMETERS['pipeline']['stage1']['time_progressing']

progressing_data, attrition_data, hold_data = simulate_data(
    years, capacity_default, duration_default, attrition_default
)

# Layout for the Dash app
app.layout = html.Div([
    html.Div([
        html.H2("Simulation Settings", style={"textAlign": "center"}),

        html.Label("Select Course", style={"marginTop": "20px"}),
        dcc.Dropdown([
            {"label": "Course 1", "value": "course1"},
            {"label": "Course 2", "value": "course2"},
        ], value="course1", id="course-selector"),

        html.Label("Year", style={"marginTop": "20px"}),
        dcc.Slider(
            min=1, max=10, step=1, value=1, id="year-slider",
            marks={i: str(i) for i in range(1, 11)}
        ),

        html.H3("Course Settings", style={"marginTop": "30px"}),
        html.Div([
            html.Label("Capacity"),
            dcc.Input(id="course-capacity", type="number", value=capacity_default),

            html.Label("Duration", style={"marginTop": "10px"}),
            dcc.Input(id="course-duration", type="number", value=duration_default),

            html.Label("Attrition", style={"marginTop": "10px"}),
            dcc.Input(id="course-attrition", type="number", value=attrition_default, step=0.01),
        ], style={"marginBottom": "20px"}),

        html.Button("Run Model", id="run-model", n_clicks=0, style={"marginTop": "20px", "width": "100%", "backgroundColor": "#007bff", "color": "white"}),
    ], style={"width": "25%", "float": "left", "padding": "20px", "borderRight": "1px solid #ddd"}),

    html.Div([
        html.H2("Simulation Results", style={"textAlign": "center", "marginTop": "10px"}),
        dcc.Graph(id="progressing-graph", style={"marginTop": "20px"}),
        dcc.Graph(id="attrition-graph", style={"marginTop": "20px"}),
        dcc.Graph(id="hold-graph", style={"marginTop": "20px"}),
    ], style={"width": "70%", "float": "right", "padding": "20px"})
])

@app.callback(
    [
        Output("progressing-graph", "figure"),
        Output("attrition-graph", "figure"),
        Output("hold-graph", "figure"),
    ],
    [
        Input("run-model", "n_clicks")
    ],
    [
        State("year-slider", "value"),
        State("course-selector", "value"),
        State("course-capacity", "value"),
        State("course-duration", "value"),
        State("course-attrition", "value"),
    ]
)
def update_output_div(n_clicks, year, selected_course, capacity, duration, attrition):
    # Simulate data for the selected course
    if n_clicks > 0 and year and selected_course and capacity and duration and attrition:
        ''' Send request to FastAPI backend
        the format of json should be {stagenum}_capacity_progressing,{stagenum}_time_progressing
        and {stagenum}_capacity_progressing where stagenum is the course number or name as specified in career_pathway.csv
        this format is required to run the model simulations'''
        #in this example only one stage - stage1 is passed# For the other stages default values will be selected.
        response = requests.post(
            "http://127.0.0.1:8055/simulate/",
            json={"stage1_capacity_progressing": [capacity],
                  "stage1_time_progressing": [duration],
                  "stage1_drop_out_progressing": [attrition],
                  }
        )
        if response.status_code == 200:
            data = response.json()
            #convert json to dataframe
            df = json.loads(data)
            print(df['progressing_stage1_count'])

    progressing, attrition_data, hold = simulate_data(year, capacity, duration, attrition)

    # Create figures
    progressing_fig = go.Figure()
    progressing_fig.add_trace(go.Scatter(y=progressing, mode="lines", name=selected_course))
    progressing_fig.update_layout(title="Progressing", xaxis_title="Year", yaxis_title="Count")

    attrition_fig = go.Figure()
    attrition_fig.add_trace(go.Scatter(y=attrition_data, mode="lines", name=selected_course))
    attrition_fig.update_layout(title="Attrition", xaxis_title="Year", yaxis_title="Rate")

    hold_fig = go.Figure()
    hold_fig.add_trace(go.Scatter(y=hold, mode="lines", name=selected_course))
    hold_fig.update_layout(title="Hold", xaxis_title="Year", yaxis_title="Count")


    return progressing_fig, attrition_fig, hold_fig



if __name__ == "__main__":
    app.run_server(debug=True)

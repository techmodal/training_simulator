import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
import json
import default  # Import the default.py file containing DEFAULT_PARAMETERS

# Extract course data from default.py instead of API request
COURSE_DATA = default.DEFAULT_PARAMETERS["pipeline"]

# Define dropdown options based on the local file
dropdown_options = [
    {"label": f"Stage {i} ({key})", "value": key}
    for i, key in enumerate(COURSE_DATA.keys(), start=1)
]

API_URL = "http://127.0.0.1:8055"

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

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H2("Simulation Settings", style={"textAlign": "center", "color": "#0064C4", "fontSize": "20px"}),
            html.Label("Select Course(s)", style={"marginTop": "20px"}),
            dcc.Dropdown(
                id="course-selector",
                options=dropdown_options,
                value=["stage1"],
                multi=True,
                style={"marginBottom": "20px"}
            ),
            html.Label("Year", style={"marginTop": "20px"}),
            dcc.Slider(
                id="year-slider",
                min=1, max=10, step=1, value=1,
                marks={i: str(i) for i in range(1, 11)},
                tooltip={"always_visible": True}
            ),
            html.H3("Course Settings", style={"marginTop": "30px", "color": "#0064C4", "fontSize": "18px"}),
            html.Div(id="course-settings-container", style={"marginBottom": "10px", "fontSize": "14px"}),
            html.Button("Run Model", id="run-model", n_clicks=0,
                        style={"marginTop": "20px", "width": "100%", "padding": "10px", "backgroundColor": "#0064C4", "color": "white", "border": "none", "borderRadius": "4px", "fontSize": "16px", "cursor": "pointer"}),
        ], style={"width": "30%", "padding": "10px", "backgroundColor": "#f9f9f9", "borderRight": "2px solid #eee", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"}),
        html.Div([
            html.H2("Simulation Results", style={"textAlign": "center", "color": "#0064C4"}),
            dcc.Graph(id="progressing-graph", style={"marginTop": "20px"}),
            dcc.Graph(id="attrition-graph", style={"marginTop": "20px"}),
            dcc.Graph(id="hold-graph", style={"marginTop": "20px"}),
        ], style={"width": "70%", "padding": "20px"})
    ], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-between", "flexWrap": "nowrap", "alignItems": "flex-start"})
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#fff", "padding": "20px"})

# Callback to update course settings dynamically
@app.callback(
    Output("course-settings-container", "children"),
    [Input("course-selector", "value")]
)
def update_course_settings(selected_courses):
    if not selected_courses:
        return []
    children = []
    for course in selected_courses:
        data = COURSE_DATA[course]
        children.append(html.Div([
            html.H4(f"Settings for {course}", style={"color": "#333", "fontSize": "16px"}),
            html.Label("Capacity"),
            dcc.Input(id={"type": "course-capacity", "index": course}, type="number", value=data["capacity_progressing"], style={"width": "90%", "padding": "5px", "marginBottom": "10px"}),
            html.Label("Duration"),
            dcc.Input(id={"type": "course-duration", "index": course}, type="number", value=data["time_progressing"], style={"width": "90%", "padding": "5px", "marginBottom": "10px"}),
            html.Label("Attrition"),
            dcc.Input(id={"type": "course-attrition", "index": course}, type="number", value=data["drop_out_progressing"], step=0.01, style={"width": "90%", "padding": "5px", "marginBottom": "10px"}),
            html.Hr(style={"marginTop": "20px"})
        ], style={"marginBottom": "10px", "padding": "8px", "border": "1px solid #ddd", "borderRadius": "4px", "backgroundColor": "#fff"}))
    return children


@app.callback(
    [Output("progressing-graph", "figure"),
     Output("attrition-graph", "figure"),
     Output("hold-graph", "figure")],
    [Input("run-model", "n_clicks")],
    [State("year-slider", "value"),
     State("course-selector", "value"),
     State({"type": "course-capacity", "index": ALL}, "value"),
     State({"type": "course-duration", "index": ALL}, "value"),
     State({"type": "course-attrition", "index": ALL}, "value")]
)
def update_plots(n_clicks, year, selected_courses, capacities, durations, attritions):
    print('test')
    if not selected_courses:
        selected_courses = []
    if not isinstance(selected_courses, list):
        selected_courses = [selected_courses]
    print(selected_courses)
    print(capacities)
    print(durations)
    print(attritions)
    progressing = []
    attrition_data = []
    hold = []
    progressing_fig = go.Figure()
    attrition_fig = go.Figure()
    hold_fig = go.Figure()
    # Simulate data for the selected course
    if n_clicks > 0 and selected_courses:
        ''' Send request to FastAPI backend
        the format of json should be {stagenum}_capacity_progressing,{stagenum}_time_progressing
        and {stagenum}_capacity_progressing where stagenum is the course number or name as specified in career_pathway.csv
        this format is required to run the model simulations'''
        #in this example only one stage - stage1 is passed# For the other stages default values will be selected.
        dict_params={}
        for i, course in enumerate(selected_courses):
            dict_params[f'{selected_courses[i]}_capacity_progressing']=[capacities[i]]
            dict_params[f'{selected_courses[i]}_time_progressing'] = [durations[i]]
            dict_params[f'{selected_courses[i]}_drop_out_progressing'] = [attritions[i]]

        response = requests.post(
            f"{API_URL}/simulate/",
            json=dict_params
        )
        if response.status_code == 200:
            data = response.json()
            #convert json to dataframe
            df = pd.read_json(data)

            df= df.T
            df=df.groupby(np.arange(len(df)) // 12).mean().reset_index()

            print(df)

            for i, course in enumerate(selected_courses):
                progressing = df[f'progressing_{selected_courses[i]}_count']
                attrition_data = df[f'left_{selected_courses[i]}_count']
                hold =df[f'hold_{selected_courses[i]}_count']

            # Create figures
                progressing_fig.add_trace(go.Scatter(y=progressing, mode="lines", name=course))
                progressing_fig.update_layout(title="Progressing", xaxis_title="Year", yaxis_title="Count")

                attrition_fig.add_trace(go.Scatter(y=attrition_data, mode="lines", name=course))
                attrition_fig.update_layout(title="Attrition", xaxis_title="Year", yaxis_title="Rate")

                hold_fig.add_trace(go.Scatter(y=hold, mode="lines", name=course))
                hold_fig.update_layout(title="Hold", xaxis_title="Year", yaxis_title="Count")


    return progressing_fig, attrition_fig, hold_fig

if __name__ == "__main__":
    app.run_server(debug=True)

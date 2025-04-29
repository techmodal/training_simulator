import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import dcc, html, Input, Output, State, ALL

import default  # Import the default.py file containing DEFAULT_PARAMETERS

# Extract course data from default.py instead of API request
COURSE_DATA = default.DEFAULT_PARAMETERS["pipeline"]

# Define dropdown options based on the local file
dropdown_options = [
    {"label": f"Stage {i} ({key})", "value": key}
    for i, key in enumerate(COURSE_DATA.keys(), start=1)
]

API_URL = "http://127.0.0.1:8055"

app = dash.Dash(
    __name__,
    requests_pathname_prefix='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"]
)
app.title = "Course Progression Model"


# Initial data parameters
def simulate_data(years, capacity, duration, attrition):
    #print('aaa')
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


app.layout = dbc.Container([
    dbc.Row([
        # Left panel
        dbc.Col([
            html.H3("Simulation Settings", className="text-center mb-4", style={"color": "#707070"}),
            html.P("Select Course(s)", className="mb-2 fw-bold", style={"color": "#0064C4"}),
            dcc.Dropdown(
                id="course-selector",
                options=dropdown_options,
                value=["stage1"],
                multi=True,
                className="mb-4"
            ),

            html.P("Year", className="mb-2 fw-bold", style={"color": "#0064C4"}),

            dcc.Slider(
                id="year-slider",
                min=0, max=10, step=1, value=6,
                marks={i: str(i) for i in range(1, 11)},
                tooltip={"always_visible": False}
            ),

            html.P("Course Settings", className="mb-2 mt-4 fw-bold", style={"color": "#0064C4"}),

            html.Div(id="course-settings-container", className="mb-4"),

            dbc.Button([
                html.I(className="bi bi-play-fill me-2"),
                "Run Model"
            ], id="run-model", n_clicks=0,
                color="primary", className="w-100",
                style={
                    "backgroundColor": "#0064C4",
                    "borderRadius": "6px",
                    "border": "none",
                    "boxShadow": "inset 0 1px 1px rgba(255,255,255,0.1)"
                }
            )
        ], md=3, style={
            "backgroundColor": "#f4f4f4",
            "padding": "20px",
            "minHeight": "100vh",
            "borderRight": "1px solid #dee2e6"
        }),

        # Right panel
        dbc.Col([
            html.H3("Simulation Results", className="text-center mb-4", style={"color": "#707070"}),

            html.Div([
                dcc.Loading(
                    type="circle",
                    color="#0064C4",
                    style={
                        "position": "absolute",
                        "top": "200px",
                        "left": "50%",
                        "transform": "translateX(-50%)",
                        "zIndex": "1000"
                    },
                    children=[
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Progressing", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody([
                                    dcc.Graph(id="progressing-graph", style={"height": "250px"})
                                ], style={"padding": "10px"})
                            ], className="mb-4 shadow-sm",
                                style={"backgroundColor": "#fff", "borderRadius": "5px 5px 0 0"}),

                            dbc.Card([
                                dbc.CardHeader("Attrition", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody([
                                    dcc.Graph(id="attrition-graph", style={"height": "250px"})
                                ], style={"padding": "10px"})
                            ], className="mb-4 shadow-sm",
                                style={"backgroundColor": "#fff", "borderRadius": "5px 5px 0 0"}),

                            dbc.Card([
                                dbc.CardHeader("Hold", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody([
                                    dcc.Graph(id="hold-graph", style={"height": "250px"})
                                ], style={"padding": "10px"})
                            ], className="mb-0 shadow-sm",
                                style={"backgroundColor": "#fff", "borderRadius": "5px 5px 0 0"})
                        ])
                    ]
                )
            ], style={"position": "relative", "minHeight": "600px"})
        ], md=8, style={"padding": "20px"})
    ])
], fluid=True, className="bg-light")


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

        children.append(
            dbc.Card([
                dbc.CardBody([
                    html.P(f"Settings for {course}", className="mb-3 fw-bold"),

                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-people-fill me-2"),
                            "Capacity"
                        ], className="w-75"),
                        dbc.Input(
                            id={"type": "course-capacity", "index": course},
                            type="number",
                            value=data["capacity_progressing"],
                            className="w-25"
                        )
                    ], className="mb-3"),

                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-hourglass-split me-2"),
                            "Duration"
                        ], className="w-75"),
                        dbc.Input(
                            id={"type": "course-duration", "index": course},
                            type="number",
                            value=data["time_progressing"],
                            className="w-25"
                        )
                    ], className="mb-3"),

                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-exclamation-triangle-fill me-2"),
                            "Attrition"
                        ], className="w-75"),
                        dbc.Input(
                            id={"type": "course-attrition", "index": course},
                            type="number",
                            step=0.01,
                            value=data["drop_out_progressing"],
                            className="w-25"
                        )
                    ])
                ])
            ], className="mb-4 shadow-sm")
        )

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
    #print(selected_courses)
    #print(capacities)
    #print(durations)
    #print(attritions)
    progressing = []
    attrition_data = []
    hold = []
    progressing_fig = go.Figure()
    attrition_fig = go.Figure()
    hold_fig = go.Figure()
    # Simulate data for the selected course
    if n_clicks > 0 and year and selected_courses:
        ''' Send request to FastAPI backend
        the format of json should be {stagenum}_capacity_progressing,{stagenum}_time_progressing
        and {stagenum}_capacity_progressing where stagenum is the course number or name as specified in career_pathway.csv
        this format is required to run the model simulations'''
        # in this example only one stage - stage1 is passed# For the other stages default values will be selected.
        dict_params = {}
        for i, course in enumerate(selected_courses):
            dict_params[f'{selected_courses[i]}_capacity_progressing'] = [capacities[i]]
            dict_params[f'{selected_courses[i]}_time_progressing'] = [durations[i]]
            dict_params[f'{selected_courses[i]}_drop_out_progressing'] = [attritions[i]]

        response = requests.post(
            f"{API_URL}/simulate/",
            json=dict_params
        )
        if response.status_code == 200:
            data = response.json()
            # convert json to dataframe
            df = pd.read_json(data)

            df = df.T
            df = df.groupby(np.arange(len(df)) // 12).mean().reset_index()
            df.index += 1

            print(df)

            for i, course in enumerate(selected_courses):
                progressing = df[f'progressing_{selected_courses[i]}_count']
                attrition_data = df[f'left_{selected_courses[i]}_count']
                hold = df[f'hold_{selected_courses[i]}_count']
               # print(hold[:year])

                # Create figures
                progressing_fig.add_trace(go.Scatter(y=progressing[:year], mode="lines", name=course))
                progressing_fig.update_layout(xaxis_title="Year", yaxis_title="Count",
                                              margin=dict(l=20, r=20, t=20, b=20))

                attrition_fig.add_trace(go.Scatter(y=attrition_data[:year], mode="lines", name=course))
                attrition_fig.update_layout(xaxis_title="Year", yaxis_title="Rate", margin=dict(l=20, r=20, t=20, b=20))

                hold_fig.add_trace(go.Scatter(y=hold[:year], mode="lines", name=course))
                hold_fig.update_layout(yaxis_range=[0, 1.1 * max(year)])
                hold_fig.update_layout(xaxis_title="Year", yaxis_title="Count", margin=dict(l=20, r=20, t=20, b=20))

    return progressing_fig, attrition_fig, hold_fig


if __name__ == "__main__":
    app.run_server(debug=True)

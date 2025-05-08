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
courses = list(COURSE_DATA.keys())
courses.remove("init")
# Define dropdown options based on the local file
dropdown_options = [
    {"label": f"Course {i}", "value": key}
    for i, key in enumerate(courses, start=1)
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
    # print('aaa')
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
            html.H3("Course(s) Parameters", className="text-center mb-4", style={"color": "#707070"}),
            html.P("Select Course(s)", className="mb-2 fw-bold", style={"color": "#0064C4"}),
            dcc.Dropdown(
                id="course-selector",
                options=dropdown_options,
                value=["course1"],
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

            # Landing page message
            html.Div(id="intro-message", children=[
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            dbc.CardBody([
                                html.H4([
                                    html.Span([
                                        html.I(className="bi bi-person-lines-fill me-2", id="intro-icon"),
                                        dbc.Tooltip(
                                            "Click 'Run Model' to begin the simulation.",
                                            target="intro-icon",
                                            placement="right",
                                            style={"fontSize": "0.85rem"}
                                        )
                                    ]),
                                    "Workforce Pipeline Simulator"
                                ], className="mb-3", style={"color": "#0064C4"}),

                                html.P(
                                    "This tool allows you to simulate workforce training pipelines by modelling how agents (trainees) move through different training stages over time."),
                                html.P(
                                    "It helps evaluate how course capacity, duration, and attrition rates affect the ability to deliver trained personnel."),

                                html.H5([
                                    "How to use:"
                                ], className="mt-4 mb-2", style={"color": "#343a40"}),

                                html.Ul([
                                    html.Li([
                                        html.I(className="bi bi-ui-checks me-2", style={"color": "#6c757d"}),
                                        "Select one or more courses using the dropdown on the left."
                                    ]),
                                    html.Li([
                                        html.I(className="bi bi-sliders2 me-2", style={"color": "#6c757d"}),
                                        "Adjust parameters such as capacity, duration, and attrition for each selected course."
                                    ]),
                                    html.Li([
                                        html.I(className="bi bi-play-circle me-2", style={"color": "#6c757d"}),
                                        "Click the 'Run Model' button to simulate the pipeline."
                                    ]),
                                    html.Li([
                                        html.I(className="bi bi-bar-chart-line me-2", style={"color": "#6c757d"}),
                                        "Simulation results will be shown here as interactive charts."
                                    ])
                                ], style={"lineHeight": "1.8", "listStyleType": "none", "paddingLeft": "0"}),

                                html.P(
                                    "The model uses agent-based simulation to explore how individuals progress through a training pipeline.")
                            ], style={"maxWidth": "100%"})
                        ], md=6),

                        dbc.Col([
                            html.Img(
                                src="/dashboard/assets/pipeline_diagram.png",
                                style={
                                    "maxWidth": "100%",
                                    "minHeight": "280px",
                                    "objectFit": "contain",
                                    "padding": "10px",
                                    "borderRadius": "10px"
                                }
                            )
                        ], md=6, className="d-flex align-items-center justify-content-center")
                    ], align="center")
                ], className="shadow", style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "5px",
                    "border": "1px solid #dee2e6",
                    "padding": "20px",
                    "width": "100%"
                })
            ]),

            # Graphs section (initially hidden)
            html.Div(id="graph-section", children=[
                html.H3("Simulation Results", className="text-center mb-4", style={"color": "#707070"}),
                html.Div(id="summary-metrics", className="mb-4", style={
                    "textAlign": "center",
                    "fontWeight": "bold",
                    "color": "#343a40",
                    "fontSize": "1.1rem"
                }),
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    color="#0064C4",
                    fullscreen=False,
                    style={
                        "marginTop": "-350px",
                        "textAlign": "center",
                        "minHeight": "150px"
                    },
                    children=[
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Progressing", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody(
                                    dcc.Graph(id="progressing-graph", style={"height": "250px"}),
                                    style={"padding": "10px", "overflowX": "auto"}
                                )
                            ], className="mb-4 shadow-sm"),

                            dbc.Card([
                                dbc.CardHeader("Attrition", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody(
                                    dcc.Graph(id="attrition-graph", style={"height": "250px"}),
                                    style={"padding": "10px", "overflowX": "auto"}
                                )
                            ], className="mb-4 shadow-sm"),

                            dbc.Card([
                                dbc.CardHeader("Hold", className="fw-bold",
                                               style={"backgroundColor": "#f4f4f4", "color": "#0064C4"}),
                                dbc.CardBody(
                                    dcc.Graph(id="hold-graph", style={"height": "250px"}),
                                    style={"padding": "10px", "overflowX": "auto"}
                                )
                            ], className="mb-0 shadow-sm")
                        ]),
                    ]
                )
            ], style={"visibility": "hidden", "height": "0", "overflow": "hidden"})

        ], md=9, style={"padding": "20px"})
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
                        ]),
                        dbc.Input(
                            id={"type": "course-capacity", "index": course},
                            type="number",
                            value=data["capacity_progressing"],
                            className="text-end"
                        ),
                        dbc.InputGroupText([
                            html.I(
                                id={"type": "tooltip-capacity", "index": course},
                                className="bi bi-question-circle-fill text-muted",
                                style={"cursor": "pointer"}
                            )
                        ])
                    ], className="mb-3"),

                    dbc.Tooltip(
                        "Maximum number of trainees on the course.",
                        target={"type": "tooltip-capacity", "index": course},
                        placement="top"
                    ),

                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-hourglass-split me-2"),
                            "Duration"
                        ]),
                        dbc.Input(
                            id={"type": "course-duration", "index": course},
                            type="number",
                            value=data["time_progressing"],
                            className="text-end"
                        ),
                        dbc.InputGroupText([
                            html.I(
                                id={"type": "tooltip-duration", "index": course},
                                className="bi bi-question-circle-fill text-muted",
                                style={"cursor": "pointer"}
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Tooltip(
                        "Number of months trainees need to complete this stage.",
                        target={"type": "tooltip-duration", "index": course},
                        placement="top"
                    ),

                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-exclamation-triangle-fill me-2"),
                            "Attrition"
                        ]),
                        dbc.Input(
                            id={"type": "course-attrition", "index": course},
                            type="number",
                            step=0.01,
                            value=data["drop_out_progressing"],
                            className="text-end"
                        ),
                        dbc.InputGroupText([
                            html.I(
                                id={"type": "tooltip-attrition", "index": course},
                                className="bi bi-question-circle-fill text-muted",
                                style={"cursor": "pointer"}
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Tooltip(
                        "Probability of trainees expected to drop out during this stage.",
                        target={"type": "tooltip-attrition", "index": course},
                        placement="top"
                    ),
                    dbc.InputGroup([
                        dbc.InputGroupText([
                            html.I(className="bi bi-book me-2"),
                            "Resource"
                        ]),
                        dbc.Input(
                            id={"type": "course-resources", "index": course},
                            type="number",
                            value=90,
                            className="text-end"
                        ),
                        dbc.InputGroupText([
                            html.I(
                                id={"type": "tooltip-resources", "index": course},
                                className="bi bi-question-circle-fill text-muted",
                                style={"cursor": "pointer"}
                            )
                        ])
                    ]),
                    dbc.Tooltip(
                        "Ratio of available course resources (e.g. training instructors, equipment))",
                        target={"type": "tooltip-resources", "index": course},
                        placement="top"
                    ),

                ])
            ], className="mb-4 shadow-sm")
        )

    return children


@app.callback(
    [
        Output("progressing-graph", "figure"),
        Output("attrition-graph", "figure"),
        Output("hold-graph", "figure"),
        Output("summary-metrics", "children"),
    ],
    [
        Input("run-model", "n_clicks"),
        State("year-slider", "value"),
        State("course-selector", "value"),
        State({"type": "course-capacity", "index": ALL}, "value"),
        State({"type": "course-duration", "index": ALL}, "value"),
        State({"type": "course-attrition", "index": ALL}, "value"),
    ]
)
def update_plots(n_clicks, year, selected_courses, capacities, durations, attritions):
    #print('test')
    if not selected_courses:
        selected_courses = []
    if not isinstance(selected_courses, list):
        selected_courses = [selected_courses]
    # print(selected_courses)
    # print(capacities)
    # print(durations)
    # print(attritions)
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

            #print(df[['complete_training_pathway1_complete_count','complete_training_pathway2_complete_count','complete_training_pathway3_complete_count']])

            for i, course in enumerate(selected_courses):
                #check if column does not exist i.e. noone has left then just initialise as 0
                #not to throw an exception
                if f'progressing_{selected_courses[i]}_count' not in df.columns:
                    df[f'progressing_{selected_courses[i]}_count']=0
                if f'hold_{selected_courses[i]}_count' not in df.columns:
                    df[f'hold_{selected_courses[i]}_count'] = 0
                if f'left_{selected_courses[i]}_count' not in df.columns:
                    df[f'left_{selected_courses[i]}_count'] = 0

                progressing = df[f'progressing_{selected_courses[i]}_count']
                attrition_data = df[f'left_{selected_courses[i]}_count']
                hold = df[f'hold_{selected_courses[i]}_count']
                # print(hold[:year])

                # Create figures
                progressing_fig.add_trace(go.Scatter(y=progressing[:year], mode="lines", name=course))
                progressing_fig.update_layout(
                    autosize=True,
                    xaxis_title="Year",
                    yaxis_title="Count",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                attrition_fig.add_trace(go.Scatter(y=attrition_data[:year], mode="lines", name=course))
                attrition_fig.update_layout(
                    autosize=True,
                    xaxis_title="Year",
                    yaxis_title="Rate",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                hold_fig.add_trace(go.Scatter(y=hold[:year], mode="lines", name=course))
                hold_fig.update_layout(yaxis_range=[0, max(hold[:year]) * 1.1])
                hold_fig.update_layout(
                    autosize=True,
                    yaxis_range=[0, max(hold[:year]) * 1.1],
                    xaxis_title="Year",
                    yaxis_title="Count",
                    margin=dict(l=20, r=20, t=20, b=20)
                )

    summary_text = ""
    if n_clicks > 0 and year and selected_courses and response.status_code == 200:
        # assuming only 1 course selected
        last_year = year - 1  # because index starts from 0
        complete_path1=df['complete_training_pathway1_complete_count']
        complete_path2 = df['complete_training_pathway2_complete_count']
        complete_path3 = df['complete_training_pathway3_complete_count']

        final_pathway1_count = int(complete_path1[last_year])
        final_pathway2_count = int(complete_path2[last_year])
        final_pathway3_count = int(complete_path3[last_year])
        final_attrition = int(attrition_data[:year].sum())
        final_hold = int(hold[last_year])
        total_started = int(progressing.iloc[0])

        summary_text = html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Trade Ready: Spec 1", className="card-title text-success"),
                        html.H4(f"{final_pathway1_count}", className="card-text")
                    ])
                ], color="light", inverse=False), md=4),

                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Trade Ready: Spec 2", className="card-title text-primary"),
                        html.H4(f"{final_pathway2_count}", className="card-text")
                    ])
                ], color="light", inverse=False), md=4),

                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Trade Ready: Spec 3", className="card-title text-warning"),
                        html.H4(f"{final_pathway3_count}", className="card-text")
                    ])
                ], color="light", inverse=False), md=4)
            ], className="mb-4")
        ])

    return progressing_fig, attrition_fig, hold_fig, summary_text


@app.callback(
    [Output("intro-message", "style"),
     Output("graph-section", "style")],
    Input("run-model", "n_clicks")
)
def toggle_landing_and_graphs(n_clicks):
    if n_clicks == 0:
        return {}, {"visibility": "hidden", "height": "0", "overflow": "hidden"}
    return {"display": "none"}, {"visibility": "visible", "height": "auto", "overflow": "visible"}


if __name__ == "__main__":
    app.run_server(debug=True)

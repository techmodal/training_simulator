import json
import os
import sys
from typing import List
import time
import datetime
import copy
import pandas as pd
import pyarrow as pa
import mesa
from .model import PipelineModel
from .structure import Results, Stage
from .analysis import (
    make_average_path,
    get_agents_data,
    make_average_times_path,
)
import networkx as nx
from .default import DEFAULT_PARAMETERS

import argparse

SAVE_FREQUENCY=10

def check_restream(model_params, career_pathway_file):
    """param override fucntionality to set career pathway weights based on hard-coded ratio maps in default py. Documentation - https://defencedigital.atlassian.net/wiki/spaces/DRK/pages/485032241/Streaming+Ratio+s

    Args:
        model_params (_type_): iteration param set
        career_pathway_file (_type_): path to csv file for stage streaming weights

    Returns:
        df: pandas df of streaming weighst with values overriden if applicable
    """
    df = pd.read_csv(career_pathway_file)
    #df.iloc[1,2] = DEFAULT_CONFIG['streaming_ratio_map'][model_params["streaming"]]['stage4']
    #df.iloc[2,2] = DEFAULT_CONFIG['streaming_ratio_map'][model_params["streaming"]]['stage7']
    #df.iloc[3,2] = DEFAULT_CONFIG['streaming_ratio_map'][model_params["streaming"]]['stage10']

    return df

#     return results
def run_batch(
    parameters: dict = DEFAULT_PARAMETERS,
    #career_pathway_file="training_simulator/career_pathway.csv"
    career_pathway_file="career_pathway.csv"

) -> List[Results]:
    df = check_restream(parameters, career_pathway_file)
    iterations = parameters["simulation"]["iterations"]
    steps = parameters["simulation"]["steps"]
    model_params_json = json.dumps(parameters)

    cp = nx.from_pandas_edgelist(
        df,
        source="fromstage",
        target="tostage",
        edge_attr=True,
        create_using=nx.DiGraph())


    results = mesa.batch_run(
        PipelineModel,
        parameters={
            "parameters": model_params_json,
            "career_pathway": json.dumps(nx.to_dict_of_dicts(cp)),
        },
        iterations=iterations,
        max_steps=steps,
        data_collection_period=1,
    )

    return results

def flatten_pivot(d: pd.DataFrame, col_str: str) -> pd.DataFrame:
    d.columns = d.columns.to_series().str.join("_")
    d.columns = d.columns + "_" + col_str
    d = d.reset_index()
    return d

def process_row(row, df_network, career_pathway_file):
    params = DEFAULT_PARAMETERS
    params_new_values = json.loads(row)
    iteration_params = copy.deepcopy(params_new_values)
    df_networkexp = df_network
    if iteration_params.get('fj_streaming'):
        params["fj_streaming"]=iteration_params['fj_streaming']

    for k in iteration_params.keys():
        if (k!="fj_streaming") :
            pk = k.split("_", 1)

            params["pipeline"][pk[0]][pk[1]] = iteration_params[k]
    params = copy.deepcopy(params)
    print(f"params: {params}")

    sim_results = run_batch(parameters=params, career_pathway_file=career_pathway_file)

    a_df = get_agents_data(sim_results)
    # drop dulplicated for left_raf
    left_df = a_df.loc[a_df["State"] == "left_raf"]
    left_df = left_df.drop_duplicates(
        subset=["RunId", "AgentID", "Stage", "State"]
    )
    left = make_average_path(left_df)
    a_df['time_stage_state'] = pd.Series(a_df.groupby(['RunId','AgentID','Stage','State']).cumcount()+1, dtype=pd.ArrowDtype(pa.float64()))


    average_progressing = make_average_times_path(a_df)
    n_df = make_average_path(a_df.loc[a_df["State"] != "left_raf"])
    # pivot table flatten
    left = flatten_pivot(left, "count")
    n_df = flatten_pivot(n_df, "count")
    average_progressing = flatten_pivot(average_progressing, "time")
    average_progressing = average_progressing.round(1)
    flat_df = pd.merge(n_df, left, on="Step", how="outer").fillna(0)
    flat_df = pd.merge(
        flat_df, average_progressing, on="Step", how="outer"
    ).fillna(0)

    # adding exit count for each stage except INIT
    for stage in Stage:
        if (stage.value != Stage.INIT.value):
            left_agents = a_df.loc[(a_df['State'] == 'left_raf') & (a_df['Stage'] == stage.value)][
                'AgentID'].tolist()
            progressing_df = a_df[a_df['AgentID'].isin(left_agents) == False]
            stage_df = progressing_df.loc[
                (progressing_df['Stage'] == stage.value) & (progressing_df['State'] == 'progressing')]
            stage_df = stage_df.drop_duplicates(subset=['RunId', 'AgentID', 'Stage', 'State'], keep='last')
            stage_exit = make_average_path(stage_df)
            stage_exit = flatten_pivot(stage_exit, 'exit_count')
            col = stage_exit.columns.values.tolist()
            col.remove('Step')
            stage_exit[col] = stage_exit[col].shift(1)
            flat_df = pd.merge(flat_df, stage_exit, on='Step', how='outer').fillna(0)

    dates = pd.date_range(start='11/01/2023', periods=flat_df.shape[0], freq='MS')
    flat_df['year_month'] = dates.strftime('%Y_%m')

    for k in params_new_values.keys():
        col = "param_" + k
        flat_df[col] = params_new_values[k]
    return flat_df, params['version']









    
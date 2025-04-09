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
import boto3
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

def print_start_info(start_time, runinfo_text):
    print(f"*"*50)
    print(f"Starting a new model run ({runinfo_text})")
    print(f"*"*50)
    print(f"Iteration start time: {datetime.datetime.fromtimestamp(start_time).isoformat()}")

def print_end_info(start_time):
    end_time = time.time()
    print(f"Iteration end time: {datetime.datetime.fromtimestamp(end_time).isoformat()}")
    print("Time taken for this run: %s seconds " % (end_time - start_time))
    print("\n\n")

def write_df_out_to_parquet(df_out, dir_path, file_name):
    df_out.columns = [c.replace(' ', '_') .strip().lower() for c in df_out.columns]
    full_file_path = dir_path + file_name
    make_dir(dir_path)
    print(f"Saving output to {full_file_path}")
    if 'complete_rpas_complete_count' not in df_out.columns:
        df_out['complete_rpas_complete_count'] = 0.0
    if 'complete_rpas_complete_time' not in df_out.columns:
        df_out['complete_rpas_complete_time'] = 0.0
    df_out.to_parquet(full_file_path, index= False)

# this is now run from a notebook:
def run_with_best_parm_mean_results(
        row: str, 
        container_id : str, 
        batch_run_start_time: str,
        career_pathway_file: str = "aircrew_batch_run_workflow/training_simulator/training_simulator/career_pathway.csv",
):
    start_time = time.time()
    print_start_info(start_time, 'Best Parameter Mean as input')

    #str(json.loads(row)[0]) json.loads(row) turns '[{... into [{..., [0] turns it into {...,
    # and then we need to turn it back into a string for the next function
    df_network = pd.read_csv(career_pathway_file)
    flat_df, params_version = process_row(str(json.loads(row)[0]).replace("'", '"'), df_network, career_pathway_file)
    df_out = flat_df
    df_out.columns = [c.replace(' ', '_') .strip().lower() for c in df_out.columns]
    print_end_info(start_time)
    return df_out

def run_batch_from_s3_parameters(
    s3_file_path: str,
    output_path: str,
    container_id: str,
    batch_run_start_time: str,
    career_pathway_file: str = "training_simulator/training_simulator/career_pathway.csv",
):
    # Add a training '/' if dont already exists
    output_path = os.path.join(output_path, "")
    try:
        df = pd.read_csv(s3_file_path, sep="  ", header=None)
        df_network = pd.read_csv(career_pathway_file)
        df_out = pd.DataFrame()
        next_row = get_next_row_from_last_checkpoint(output_path, DEFAULT_PARAMETERS, container_id, batch_run_start_time)
        if next_row == 0:
            print(f"No previous output file found for container-{container_id}. Starting the job from beginning.")
        total_runs = df.shape[0]
        if next_row > total_runs:
            print("WARNING: Output exists for all interations already. Nothing left to process. Ending the job!")
            exit(0)
        count = next_row + 1
        part_count = int(next_row/SAVE_FREQUENCY)
        for row in df[0][next_row:]:
            start_time = time.time()
            print_start_info(start_time, f"{count}/{total_runs}")
            flat_df, params_version = process_row(row, df_network, career_pathway_file)

            if df_out.empty:
                df_out = flat_df
            else:
                df_out = pd.concat([df_out, flat_df], ignore_index=True)
            if count % SAVE_FREQUENCY == 0 :
                # df_out['version'] = str(params['version'])

                dir_path = f"{output_path}version={params_version}/output_creation_dt={batch_run_start_time}/"
                file_name = f"container-{container_id}__part-{part_count}.parquet"
                write_df_out_to_parquet(df_out, dir_path, file_name)

                part_count = part_count + 1
                df_out = pd.DataFrame()
            count = count + 1
            print_end_info(start_time)


        if not df_out.empty:
            dir_path = f"{output_path}version={params_version}/output_creation_dt={batch_run_start_time}/"
            file_name = f"container-{container_id}__part-{part_count}.parquet"
            write_df_out_to_parquet(df_out, dir_path, file_name)


    except pd.errors.EmptyDataError:
        print("WARNING: File is empty, nothing to run.")

def make_dir(directory: str):
    # to_parquet() method will throw an error if the path doesnt exist (on linux)
    if not directory.startswith("s3://") and not os.path.exists(directory):
        os.makedirs(directory)

def get_next_row_from_last_checkpoint(output_path: str, default_params: dict, container_id : str, batch_run_start_time: str):
    path_to_output_files = build_partition_path(output_path, default_params, batch_run_start_time)
    print(f"Looking for previous checkpoints in {path_to_output_files}..")
    existing_output_files = [file for file in get_list_of_files(path_to_output_files) if f"container-{container_id}__part" in file]
    print(f"Existing_output_files: {existing_output_files}")
    next_row = 0
    if len(existing_output_files) != 0:
        last_output_file= sorted(existing_output_files, key=lambda e : int(e.split("part-")[1].split(".")[0]) )[-1]
        print(f"The last output file saved is: {last_output_file}")
        part_file_num = int(last_output_file.split("part-")[1].split(".")[0])
        if part_file_num == 0:
            next_row = SAVE_FREQUENCY
        else:
            next_row = (part_file_num + 1) * SAVE_FREQUENCY
    return next_row

def get_list_of_files(path_to_output_files: str):
    if path_to_output_files.startswith("s3://"):
        path_parts=path_to_output_files.replace("s3://","").split("/")
        bucket_name=path_parts.pop(0)
        key="/".join(path_parts)
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        return [ object_summary.key.split("/")[-1] for object_summary in bucket.objects.filter(Prefix=key)]
    else:
        if os.path.exists(path_to_output_files):
            return os.listdir(path_to_output_files)
        else:
            return []


def build_partition_path(output_path: str, params: str, batch_run_start_time: str):
    return f"{output_path}version={params['version']}/output_creation_dt={batch_run_start_time}/"
         

def main(args):
    parser = argparse.ArgumentParser(description="Get the args.")
    parser.add_argument("-p", "--params", type=str, default= "", required=True)
    parser.add_argument("-o", "--output", type=str, default= "")
    parser.add_argument("-c", "--containers", type=str, default= "", required=True)
    parser.add_argument("-s", "--start_time", type=str, default= "", required=True)
    parser.add_argument("-bpm", "--best_parm_mean", type=bool)

    parser.add_argument("-dd", "--dagdir", type=str, default= "")

    args = parser.parse_args(args)
    params= args.params
    output= args.output
    containers= args.containers
    start_time= args.start_time
    dagdir = args.dagdir
        
    # Access the JSON data from the environment variable
    parameter_path = output + "input/" + params
    output_path = output + "output/"

    container_id = containers
    batch_run_start_time = start_time

    print(f"Attempting to load parameter file from: {parameter_path}")
    print(f"output path: {output_path}")
    print(f"container_id: {container_id}")
    print(f"batch run start_time: {batch_run_start_time}")

    overall_start_time = time.time()

    if args.best_parm_mean:
        df_out = run_with_best_parm_mean_results(params, container_id, batch_run_start_time, career_pathway_file="./training_simulator/training_simulator/career_pathway.csv")
        return df_out
    else:
        run_batch_from_s3_parameters(parameter_path, output_path, container_id, batch_run_start_time)
    print(f"Total time taken for this batch of runs: {(time.time() - overall_start_time)} seconds ")

if __name__ == "__main__":
    main(sys.argv[1:])
    
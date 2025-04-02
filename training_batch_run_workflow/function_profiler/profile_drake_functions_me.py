import time
import cProfile
import pstats
# from mpi4py.MPI import COMM_WORLD
import sys
sys.path.append("../training_simulator")
sys.path.append("../training_simulator/training_simulator")

from typing import List
import pandas as pd
import pyarrow as pa

from aircrew_simulator.run import run_batch

from aircrew_simulator.structure import (
    TraineeBase,
    PipelineModelBase,
    Stage,
    StageManager,
    State,
    Results
)
from aircrew_simulator.analysis import (
    make_average_path,
    get_agents_data,
    make_time_series,
    make_average_times_path,
    make_stage_state_times_table,
    make_quantile_path,
)
from aircrew_simulator.model import PipelineModel

def run():
    steps = 12 * 3
    model_params =  {
    "version": "2.1.4",
    "wsoair_streaming": 0,
    "wsopisr_streaming": 1,
    "wsopisrew_streaming": 1,
    "simulation": {"steps": 120, "start_month": 4, "iterations": 30},
    "init_trainees": {
        "mags": {"progressing": 0, "hold": 0},
        "isrfdn": {"progressing": 0, "hold": 0},
        "matutor": {"progressing": 0, "hold": 0},
        "wsoair": {"progressing": 0, "hold": 0},
        "isrland": {"progressing": 0, "hold": 0},
        "landfly": {"progressing": 0, "hold": 0},
    },

    "pipeline": {
        "miot": {"new_trainees": 2, "input_rate": 6, "time_hold": 120},
        "desnco": {"new_trainees": 12, "input_rate": 6, "time_hold": 120},
        "mags": {
            "drop_out_progressing": 0,
            "drop_out_hold": 0,
            "capacity_progressing": 21,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "isrfdn": {
            "drop_out_progressing": 0,
            "drop_out_hold": 0,
            "capacity_progressing": 14,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "matutor": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0.0,
            "capacity_progressing": 15,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "wsoair": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "time_progressing": 3,
            "capacity_progressing": 3,
            "pathway_complete": "wsoair_complete",
            "time_hold": 120,
        },
        "isrland": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "landfly": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 2,
            "pathway_complete": "isrland_complete",
            "time_hold": 120,
        },
        "fwalm": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 6,
            "pathway_complete": "alm_complete",
            "time_hold": 120,
        },
        "fwasling": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 6,
            "time_progressing": 3,
            "time_hold": 120,
        },
        "lang": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 15,
            "time_hold": 120,
        },
        "isracoa": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 4,
            "pathway_complete": "acoa_complete",
            "time_hold": 120,
        },
        "isrew": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 6,
            "time_progressing": 3,
            "pathway_complete": "ew_complete",
            "time_hold": 120,
        },
        "brc": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 5,
            "time_progressing": 7,
            "time_hold": 120,
        },
        "arc": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 6,
            "time_progressing": 5,
            "time_hold": 120,
        },
        "mar": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 6,
            "time_progressing": 2,
            "pathway_complete": "cmn_complete",
            "time_hold": 120,
        },
    },
    "schedule": {
        "mags": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "isrfdn": [2, 4, 6, 9],
        "matutor": [1, 2, 4, 5],
        "wsoair": [1, 6, 8],
        "isrland": [1, 4, 7, 11],
        "landfly": [3, 6, 7, 9],
        "fwalm": [1, 5, 9],
        "fwasling": [1, 3, 5, 8, 11],
        "lang": [5, 10],
        "isracoa": [1, 4, 7, 11],
        "isrew": [1, 4, 7, 11],
        "brc": [1, 3, 6, 8, 10],
        "arc": [3, 5, 7, 10, 12],
        "mar": [1, 3, 5, 7, 10],
    },
}

    simulation_data = run_batch(model_params)
    average_progressing = pd.DataFrame()
    def get_agents_data(simulation_data: List[Results]) -> pd.DataFrame:
        
        df_tmp = pd.DataFrame(simulation_data).reset_index()[["RunId", "Step", "AgentID", "Stage", "State"]]

        df_tmp.RunId = pd.Series(df_tmp.RunId, dtype=pd.ArrowDtype(pa.int64()))
        df_tmp.Step = pd.Series(df_tmp.Step, dtype=pd.ArrowDtype(pa.int64()))
        df_tmp.AgentID = pd.Series(df_tmp.AgentID.astype(str), dtype=pd.ArrowDtype(pa.string()))
        df_tmp.Stage = pd.Series(df_tmp.Stage, dtype=pd.ArrowDtype(pa.string()))
        df_tmp.State = pd.Series(df_tmp.State, dtype=pd.ArrowDtype(pa.string()))
        
        return df_tmp

    def make_stage_state_times_table(agent_data: pd.DataFrame) -> pd.DataFrame:
        stage_state_df =pd.DataFrame(
            agent_data.groupby(["RunId", "Step", "Stage", "State"])['time_stage_state'].mean(),
        ).reset_index()
        return stage_state_df.pivot(
            index=["RunId", "Step"], columns=["State", "Stage"], values="time_stage_state"
        ).fillna(0)

    def make_average_times_path(agent_data) -> pd.DataFrame:
        time_series_data_df = make_stage_state_times_table(agent_data)
        return time_series_data_df.groupby("Step").mean()
    
    try:
        a_df = get_agents_data(simulation_data)
        a_df['time_stage_state'] = pd.Series(a_df.groupby(['RunId','AgentID','Stage','State']).cumcount()+1, dtype=pd.ArrowDtype(pa.float64()))
        average_progressing = make_average_times_path(a_df)  
    except Exception as e:
        print(e)
    
    return average_progressing

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('cumulative')
    time_as_string = str(round(time.time()))
    stats.dump_stats(f"profile_drake_functions_{time_as_string}.prof")
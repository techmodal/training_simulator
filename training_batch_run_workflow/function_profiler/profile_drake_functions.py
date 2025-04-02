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
from aircrew_simulator.stages import MIOT
from aircrew_simulator.structure import (
    PilotBase,
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
    model_params = {"version": "2.0.6-test",
    "simulation": {"steps": 120, "start_month": 4, "iterations": 30},
    "fj_streaming":"med",
    "init_pilots": {
        "mags": {"progressing": 0, "hold": 0},
        "eft": {"progressing": 0, "hold": 0},
        "bft": {"progressing": 0, "hold": 0},
        "fjlin": {"progressing": 0, "hold": 0},
        "ajt1": {"progressing": 0, "hold": 0},
        "ajt2": {"progressing": 0, "hold": 0},
        "melin": {"progressing": 0, "hold": 0},
        "mept": {"progressing": 0, "hold": 0},
        "mexo": {"progressing": 0, "hold": 0},
        "brt": {"progressing": 0, "hold": 0},
        "art": {"progressing": 0, "hold": 0},
        "artmar": {"progressing": 0, "hold": 0},
    },

        "pipeline": {
        "miot": {"new_pilots": 11, "input_rate": 1, "time_hold": 120},
        "mags": {
            "drop_out_progressing": 0.12,
            "drop_out_hold": 0,
            "capacity_progressing": 21,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "eft": {
            "drop_out_progressing": 0.09,
            "drop_out_stream": 0.6,
            "drop_out_hold": 0,
            "capacity_progressing": 11,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "fjlin": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0.0,
            "capacity_progressing": 10,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "bft": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "time_progressing": 13,
            "capacity_progressing": 4,
            "time_hold": 120,
        },
        "ajt1": {
            "drop_out_progressing": 0.2,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 10,
            "time_hold": 120,
        },
        "ajt2": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 6,
            "pathway_complete": "fj_complete",
            "time_hold": 120,
        },
        "melin": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "mept": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 8,
            "pathway_complete": "me_complete",
            "time_hold": 120,
        },
        "mexo": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 2,
            "time_progressing": 6,
            "pathway_complete": "mexo_complete",
            "time_hold": 120,
        },
        "brt": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 10,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "art": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 5,
            "time_hold": 120,
        },
        "artmar": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 9,
            "time_progressing": 2,
            "time_hold": 120,
        },
    },
    "schedule": {
        "mags": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "eft": [2, 3, 5, 6, 7, 8, 9, 10, 11],
        "fjlin": [1, 3, 4, 6, 7, 9, 11],
        "bft": [1, 3, 4, 5],
        "ajt1": [1, 2, 3, 5, 8, 10, 11],
        "ajt2": [2, 3, 4, 6, 7, 9, 10],
        "melin": [1, 3, 4, 5, 6, 8, 9, 10, 11],
        "mept": [1, 3, 4, 6, 8, 9, 11, 12],
        "mexo": [3, 7, 11],
        "brt": [1, 3, 6, 9, 11],
        "art": [2, 4, 6, 8, 10, 12],
        "artmar": [2, 5, 7, 9, 11],
    },
}

    simulation_data = run_batch(model_params)
    
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
        average_progressing=pd.DataFrame()
        time_series_data_df = make_stage_state_times_table(agent_data)
        return time_series_data_df.groupby("Step").mean()
    
    try:
        a_df = get_agents_data(simulation_data)
        a_df['time_stage_state'] = pd.Series(a_df.groupby(['RunId','AgentID','Stage','State']).cumcount()+1, dtype=pd.ArrowDtype(pa.float64()))
        average_progressing = make_average_times_path(a_df)  
    except Exception as e:
        print(e)
    


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    stats = pstats.Stats(pr).sort_stats('cumulative')
    time_as_string = str(round(time.time()))
    stats.dump_stats(f"profile_drake_functions_{time_as_string}.prof")
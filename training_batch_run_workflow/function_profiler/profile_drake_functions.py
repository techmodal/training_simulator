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

from training_simulator.run import run_batch
from training_simulator.stages import INIT
from training_simulator.structure import (
    TraineeBase,
    PipelineModelBase,
    Stage,
    StageManager,
    State,
    Results
)
from training_simulator.analysis import (
    make_average_path,
    get_agents_data,
    make_time_series,
    make_average_times_path,
    make_stage_state_times_table,
    make_quantile_path,
)
from training_simulator.model import PipelineModel

def run():
    steps = 12 * 3
    model_params = {"version": "2.1.3",
    "simulation": {"steps": 120, "start_month": 4, "iterations": 5},
    "streaming":1,
    "init_trainees": {
        "course1": {"progressing": 0, "hold": 0},
        "course2": {"progressing": 0, "hold": 0},
        "course3": {"progressing": 0, "hold": 0},
        "course4": {"progressing": 0, "hold": 0},
        "course5": {"progressing": 0, "hold": 0},
        "course6": {"progressing": 0, "hold": 0},
        "course7": {"progressing": 0, "hold": 0},

    },

    "pipeline": {
        "init": {"new_trainees": 10, "input_rate": 1, "time_hold": 120},
        "course1": {
            "drop_out_progressing": 0.12,
            "drop_out_hold": 0,
            "capacity_progressing": 21,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "course2": {
            "drop_out_progressing": 0.05,
            "drop_out_stream": 0.6,
            "drop_out_hold": 0,
            "capacity_progressing": 11,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "course3": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0.0,
            "capacity_progressing": 10,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "course4": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "time_progressing": 13,
            "capacity_progressing": 4,
            "pathway_complete": "training_pathway1_complete",
            "time_hold": 120,
        },
        "course5": {
            "drop_out_progressing": 0.2,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 10,
            "pathway_complete": "training_pathway2_complete",
            "time_hold": 120,
        },
        "course6": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "course7": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 2,
            "pathway_complete": "training_pathway3_complete",
            "time_hold": 120,
        },


    },
    "schedule": {
        "course1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "course2": [2, 3, 5, 6, 7, 8, 9, 10, 11],
        "course3": [1, 3, 4, 6, 7, 9, 11],
        "course4": [1, 4, 7, 10],
        "course5": [1, 4, 7, 10],
        "course6": [2, 3, 4, 6, 7, 9, 10],
        "course7": [1, 3, 4, 5, 6, 8, 9, 10, 11],


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
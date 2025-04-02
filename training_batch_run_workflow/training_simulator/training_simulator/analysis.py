from typing import List

import pandas as pd
import pyarrow as pa
import sys

from structure import Results


def get_agents_data(simulation_data: List[Results]) -> pd.DataFrame:
    df_tmp = pd.DataFrame(simulation_data).reset_index()[["RunId", "Step", "AgentID", "Stage", "State"]]

    df_tmp.RunId = pd.Series(df_tmp.RunId, dtype=pd.ArrowDtype(pa.int64()))
    df_tmp.Step = pd.Series(df_tmp.Step, dtype=pd.ArrowDtype(pa.int64()))
    df_tmp.AgentID = pd.Series(df_tmp.AgentID.astype(str), dtype=pd.ArrowDtype(pa.string()))
    df_tmp.Stage = pd.Series(df_tmp.Stage, dtype=pd.ArrowDtype(pa.string()))
    df_tmp.State = pd.Series(df_tmp.State, dtype=pd.ArrowDtype(pa.string()))
        
    return df_tmp


def get_stage_state_count_data(agent_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        agent_data.groupby(["RunId", "Step", "Stage", "State"]).size(),
        columns=["Count"],
    ).reset_index()


def get_time_series_data(stage_state_count_data: pd.DataFrame) -> pd.DataFrame:
    return stage_state_count_data.pivot(
        index=["RunId", "Step"], columns=["State", "Stage"], values="Count"
    ).fillna(0)


def make_time_series(simulation_data: List[Results]) -> pd.DataFrame:
    agents_data_df = get_agents_data(simulation_data)
    stage_state_count_data_df = get_stage_state_count_data(agents_data_df)
    time_series_data_df = get_time_series_data(stage_state_count_data_df)
    return time_series_data_df


def make_average_path(simulation_data: List[Results], model_params: dict) -> pd.DataFrame:
    time_series_data_df = make_time_series(simulation_data)
    return time_series_data_df.groupby("Step").sum() / model_params['simulation']['iterations']


def make_quantile_path(simulation_data: List[Results], quantile: float) -> pd.DataFrame:
    time_series_data_df = make_time_series(simulation_data)
    return time_series_data_df.groupby("Step").quantile(quantile)


def make_stage_state_times_table(agent_data: pd.DataFrame) -> pd.DataFrame:
    stage_state_df = pd.DataFrame(
        agent_data.groupby(["RunId", "Step", "Stage", "State"])[
            "time_stage_state"
        ].mean()
    ).reset_index()
    return stage_state_df.pivot(
        index=["RunId", "Step"], columns=["State", "Stage"], values="time_stage_state"
    ).fillna(0.0)



def make_average_times_path(agent_data) -> pd.DataFrame:
    time_series_data_df = make_stage_state_times_table(agent_data)
    return time_series_data_df.groupby("Step").mean()

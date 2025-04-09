import sys

# setting path
sys.path.append('')
import json
import networkx as nx
import mesa
import pandas as pd
from typing import List
from training_batch_run_workflow.training_simulator.training_simulator.model import PipelineModel
from training_batch_run_workflow.training_simulator.training_simulator.analysis import (
    make_average_path,
    get_agents_data,
    make_average_times_path,
)
from training_batch_run_workflow.training_simulator.training_simulator.run import run_batch
import time
import matplotlib.pyplot as plt
from training_batch_run_workflow.training_simulator.training_simulator.structure import Results, Stage, State
from default import DEFAULT_PARAMETERS
import itertools


PARAM_OVERRIDES = {
    "init_new_trainees": [11],
    "stage1_drop_out_progressing": [0.1],
    "stage1_capacity_progressing": [2],
    "stage1_time_progressing": [9],
    "stage2_drop_out_progressing": [0.2],
    "stage2_capacity_progressing": [2],
    "stage2_time_progressing": [6],

}
# PARAM_OVERRIDES = {
# "miot_new_pilots": [11, 13, 14],
# "restream_matutor_wsoair": [0.6],
# "restream_matutor_isrland": [0.4],
# }

#PARAM_OVERRIDES = {
    #"miot_new_pilots": [11, 13, 14],
 #   "streaming": [1,2],
   # "failure":[1],

#}


def plot_network(career_pathway_file="career_pathway.csv"):
    with open(career_pathway_file, "rb") as inf:
        next(inf, "")  # skip a line

        G = nx.read_edgelist(
            inf,
            nodetype=str,
            delimiter=",",
            data=(("weight", float),),
            create_using=nx.DiGraph(),
        )
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()




def flatten_pivot(d: pd.DataFrame, col_str: str) -> pd.DataFrame:
    d.columns = d.columns.to_series().str.join("_")
    d.columns = d.columns + "_" + col_str
    d = d.reset_index()
    return d


def parameter_sweep(
    param_overrides: dict,
    parameters: dict = DEFAULT_PARAMETERS,
    career_pathway_file: str = "training_simulator/career_pathway.csv",
):
    keys, values = zip(*param_overrides.items())
    print(keys)
    print(values)
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
   # print(experiments)
    df_concat = pd.DataFrame()

    for e in experiments:

        if e.get('streaming'):
            parameters["streaming"]=e['streaming']
            print(parameters["streaming"])


        for k in e.keys():
            if (k!="streaming" ) :
                pk = k.split("_", 1)
                parameters["pipeline"][pk[0]][pk[1]] = e[k]

        sim_results = run_batch(parameters=parameters)

        a_df = get_agents_data(sim_results)
        # a_df.to_csv('agent_data.csv')
        # drop dulplicated for left
        left_df = a_df.loc[a_df["State"] == "left"]
        left_df = left_df.drop_duplicates(subset=["RunId", "AgentID", "Stage", "State"])
        left = make_average_path(left_df,model_params=parameters)
        a_df["time_stage_state"] = (
            a_df.groupby(["RunId", "AgentID", "Stage", "State"]).cumcount() + 1
        )
       # average_progressing = make_average_times_path(a_df)
        n_df = make_average_path(a_df.loc[a_df["State"] != "left"], model_params=parameters)
        # pivot table flatten
        left = flatten_pivot(left, "count")
        n_df = flatten_pivot(n_df, "count")
       # average_progressing = flatten_pivot(average_progressing, "time")
        #average_progressing = average_progressing.round(1)

        flat_df = pd.merge(n_df, left, on="Step", how="outer").fillna(0)
        # flat_df = pd.merge(flat_df, average_progressing, on="Step", how="outer").fillna(
        #     0
        # )

        for stage in Stage:
            if stage.value != Stage.INIT.value:
                left_agents = a_df.loc[
                    (a_df["State"] == "left") & (a_df["Stage"] == stage.value)
                ]["AgentID"].tolist()
                progressing_df = a_df[a_df["AgentID"].isin(left_agents) == False]
                stage_df = progressing_df.loc[
                    (progressing_df["Stage"] == stage.value)
                    & (progressing_df["State"] == "progressing")
                ]
                stage_df = stage_df.drop_duplicates(
                    subset=["RunId", "AgentID", "Stage", "State"], keep="last"
                )
                stage_exit = make_average_path(stage_df, model_params=parameters)
                stage_exit = flatten_pivot(stage_exit, "exit_count")
                col = stage_exit.columns.values.tolist()
                col.remove("Step")
                stage_exit[col] = stage_exit[col].shift(1)
                flat_df = pd.merge(flat_df, stage_exit, on="Step", how="outer").fillna(
                    0
                )

    #     dates = pd.date_range(start="11/01/2023", periods=flat_df.shape[0], freq="MS")
    #     flat_df["year_month"] = dates.strftime("%Y_%m")
        for k in e.keys():
            col = "param_" + k
            flat_df[col] = e[k]
        if df_concat.empty:
            df_concat = flat_df
        else:
            df_concat = pd.concat([df_concat, flat_df], ignore_index=True)

    # df_concat["version"] = str(parameters["version"])
    df_concat.columns = [c.replace(" ", "_").strip() for c in df_concat.columns]
    #check if no left column add as 0'
    for stage in Stage:
        if f'left_{stage}_count' not in df_concat.columns:
            df_concat[f'left_{stage}_count']=0
    return df_concat


if __name__ == "__main__":
    start_time = time.time()
    # plot_network()
    df = parameter_sweep(
        param_overrides=PARAM_OVERRIDES, parameters=DEFAULT_PARAMETERS
    )
    # plot_network()
   # print(df.info())
    df.set_index('Step')
    test = df.to_json(orient='index')
    print(test)
    test2 = pd.read_json(test)
    test2=test2.T
    print(test2.head())
    df.to_json('outputs.json', orient='index')

    df.to_csv("network_fj1.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

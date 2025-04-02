DEFAULT_PARAMETERS = {"version": "2.1.3",
    "simulation": {"steps": 120, "start_month": 4, "iterations": 5},
    "streaming":1,
    "init_pilots": {
        "stage1": {"progressing": 0, "hold": 0},
        "stage2": {"progressing": 0, "hold": 0},
        "stage3": {"progressing": 0, "hold": 0},
        "stage4": {"progressing": 0, "hold": 0},
        "stage5": {"progressing": 0, "hold": 0},
        "stage6": {"progressing": 0, "hold": 0},
        "stage7": {"progressing": 0, "hold": 0},
        "stage8": {"progressing": 0, "hold": 0},
        "stage9": {"progressing": 0, "hold": 0},
        "stage10": {"progressing": 0, "hold": 0},
        "stage11": {"progressing": 0, "hold": 0},
        "stage12": {"progressing": 0, "hold": 0},
    },

    "pipeline": {
        "miot": {"new_pilots": 10, "input_rate": 1, "time_hold": 120},
        "stage1": {
            "drop_out_progressing": 0.12,
            "drop_out_hold": 0,
            "capacity_progressing": 21,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "stage2": {
            "drop_out_progressing": 0.05,
            "drop_out_stream": 0.6,
            "drop_out_hold": 0,
            "capacity_progressing": 11,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "stage3": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0.0,
            "capacity_progressing": 10,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "stage4": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "time_progressing": 13,
            "capacity_progressing": 4,
            "time_hold": 120,
        },
        "stage5": {
            "drop_out_progressing": 0.2,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 10,
            "time_hold": 120,
        },
        "stage6": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 6,
            "pathway_complete": "training_pathway1_complete",
            "time_hold": 120,
        },
        "stage7": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "stage8": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 8,
            "pathway_complete": "training_pathway2_complete",
            "time_hold": 120,
        },
        "stage9": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 2,
            "time_progressing": 6,
            "pathway_complete": "training_pathway3_complete",
            "time_hold": 120,
        },
        "stage10": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 10,
            "time_progressing": 6,
            "time_hold": 120,
        },
        "stage11": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 5,
            "time_hold": 120,
        },
        "stage12": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 9,
            "time_progressing": 2,
            "pathway_complete": "training_pathway4_complete",
            "time_hold": 120,
        },
    },
    "schedule": {
        "stage1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "stage2": [2, 3, 5, 6, 7, 8, 9, 10, 11],
        "stage3": [1, 3, 4, 6, 7, 9, 11],
        "stage4": [1, 4, 7, 10],
        "stage5": [1, 2, 3, 5, 8, 10, 11],
        "stage6": [2, 3, 4, 6, 7, 9, 10],
        "stage7": [1, 3, 4, 5, 6, 8, 9, 10, 11],
        "stage8": [1, 3, 4, 6, 8, 9, 11, 12],
        "stage9": [3, 7, 11],
        "stage10": [1, 3, 6, 9, 11],
        "stage11": [2, 4, 6, 8, 10, 12],
        "stage12": [2, 5, 7, 9, 11],
    },
}
INTERVENE_SCENARIO = {
    2 : {

    }
}

DEFAULT_CONFIG = {
    "streaming_ratio_map" : {
        3:{
            'stage4':0.45,
            'stage7':0.30,
            'stage10':0.15
            },
        2:{
            'stage4':0.45,
            'stage7':0.25,
            'stage10':0.20
            },
        1:{
            'stage4':0.45,
            'stage7':0.20,
            'stage10':0.25
            },
        0:{
            'stage4':0.4,
            'stage7':0.25,
            'stage10':0.25
            }
    }
}

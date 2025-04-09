DEFAULT_PARAMETERS = {"version": "2.1.3",
    "simulation": {"steps": 120, "start_month": 4, "iterations": 5},
    "streaming":1,
    "init_trainees": {
        "stage1": {"progressing": 0, "hold": 0},
        "stage2": {"progressing": 0, "hold": 0},
        "stage3": {"progressing": 0, "hold": 0},
        "stage21": {"progressing": 0, "hold": 0},
        "stage211": {"progressing": 0, "hold": 0},
        "stage22": {"progressing": 0, "hold": 0},
        "stage23": {"progressing": 0, "hold": 0},
        "stage231": {"progressing": 0, "hold": 0},
        "stage31": {"progressing": 0, "hold": 0},
        "stage32": {"progressing": 0, "hold": 0},
        "stage33": {"progressing": 0, "hold": 0},
    },

    "pipeline": {
        "init": {"new_trainees": 10, "input_rate": 1, "time_hold": 120},
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
        "stage21": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "time_progressing": 13,
            "capacity_progressing": 4,
            "time_hold": 120,
        },
        "stage22": {
            "drop_out_progressing": 0.2,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 10,
            "time_hold": 120,
        },
        "stage23": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 6,
            "pathway_complete": "training_pathway1_complete",
            "time_hold": 120,
        },
        "stage211": {
            "drop_out_progressing": 0.0,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "stage231": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 4,
            "time_progressing": 8,
            "pathway_complete": "training_pathway2_complete",
            "time_hold": 120,
        },
        "stage31": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 2,
            "time_progressing": 6,
            "pathway_complete": "training_pathway3_complete",
            "time_hold": 120,
        },
        "stage32": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 10,
            "time_progressing": 2,
            "time_hold": 120,
        },
        "stage33": {
            "drop_out_progressing": 0.05,
            "drop_out_hold": 0,
            "capacity_progressing": 8,
            "time_progressing": 2,
            "time_hold": 120,
        },

    },
    "schedule": {
        "stage1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "stage2": [2, 3, 5, 6, 7, 8, 9, 10, 11],
        "stage3": [1, 3, 4, 6, 7, 9, 11],
        "stage21": [1, 4, 7, 10],
        "stage211": [1, 4, 7, 10],
        "stage22": [2, 3, 4, 6, 7, 9, 10],
        "stage23": [1, 3, 4, 5, 6, 8, 9, 10, 11],
        "stage231": [1, 3, 4, 6, 8, 9, 11, 12],
        "stage31": [3, 7, 11],
        "stage32": [1, 3, 6, 9, 11],
        "stage33": [2, 4, 6, 8, 10, 12]

    },
}
INTERVENE_SCENARIO = {
    2 : {

    }
}


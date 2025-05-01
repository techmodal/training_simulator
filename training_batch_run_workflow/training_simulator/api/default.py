DEFAULT_PARAMETERS = {"version": "2.1.3",
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


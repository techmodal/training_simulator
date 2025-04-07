import random
from dataclasses import dataclass
from typing import List, Dict
import math
import uuid
from numba import jit
import sys

from training_batch_run_workflow.training_simulator.training_simulator.structure import PilotBase, Stage, StageManager, State


@dataclass
class MIOT(StageManager):
    new_pilots: int
    input_rate: int
    time_hold: int
    stage_id = "miot"

    def drop_out(self, pilot: PilotBase) -> bool:
        return pilot.time_in_stage_state > self.time_hold

    def get_next_stage(self) -> Stage:
        return Stage.MAGS

    def time_to_progress(self, pilot: PilotBase) -> bool:
        return True

    def leave_hold(self, unique_id: int) -> bool:
        return False

    def activate_stage(self, duration: int, capacity: int):
        return


@dataclass
class PipelineStage(StageManager):
    stage_id: str
    stage_name: str
    drop_out_progressing: float
    drop_out_hold: float
    capacity_progressing: int
    time_progressing: int
    time_hold: int
    adjacent_stages: Dict[str, Dict]
    hold_progressing_pilots: list[int]

    capacity_dropping: float = 0
    yearly_attrition: int = 0
    drop_out_stream: float = 0
    starting_time: int = 0
    pathway_complete: str = None
    retraining_probability: float = 0
    exit_cohort_capacity: int = 0
    restream_trainees = {}

    def drop_out(self, pilot: PilotBase) -> bool:
        coin_flip = random.uniform(0, 1)
        if pilot.state == State.PROGRESSING:
            return self.drop_out_progressing > coin_flip
        return (
            self.drop_out_hold > coin_flip or pilot.time_in_stage_state > self.time_hold
        )

    def get_next_stage(self) -> str:
        adjacent_vals = self.adjacent_stages.items()
        next_stage_weights = []

        for k, v in adjacent_vals:
            if not self.restream_trainees.get(k, 0):
                self.restream_trainees[k] = 0

            next_stage_weights.append([k, v["weight"]])
        if len(next_stage_weights) == 0:
            """ Documentation - https://defencedigital.atlassian.net/wiki/spaces/DRK/pages/498073623/OCU+Ready+Journey+Complete """
            return self.pathway_complete
        elif (len(next_stage_weights) == 1) and (next_stage_weights[0][1] == 1):
            return next_stage_weights[0][0]
        else:
            stages, weights = map(list, zip(*next_stage_weights))
            if sum(weights) < 1 and self.stage_name == "eft":
                next_stage_weights.append(["rpas_complete", 1.0 - sum(weights)])
                self.restream_trainees["rpas_complete"] = 0
            elif sum(weights) < 1:
                next_stage_weights.append(["restream", 1.0 - sum(weights)])
                self.restream_trainees["restream"] = 0
            ##check if the sum of weight is 1. If it is we don't need drop restream if yes we add another stage and weight
            ##to stages and weights
            random.shuffle(next_stage_weights)
            for i in range(len(next_stage_weights)):
                if self.check_restream_cap(self.restream_trainees.get(next_stage_weights[i][0], 0),
                                           self.exit_cohort_capacity,next_stage_weights[i][1]):

                    next_stage_weights[i][1] = 0
                    break            
            stages, weights = map(list, zip(*next_stage_weights))
            next_stage = random.choices(stages, weights=weights, k=1)[0]
            self.restream_trainees[next_stage] += 1

        return next_stage


    def check_restream_cap(self, restream_trainees, capacity, weight):
        if restream_trainees >= ceilmult(capacity,weight):
            return True
        return False


    def time_to_progress(self, pilot: PilotBase) -> bool:
        if pilot.state == State.PROGRESSING:
            return pilot.time_in_stage_state >= self.time_progressing
        elif pilot.state == State.HOLD:
            return True
        return False

    def leave_hold(self, unique_id: int) -> bool:
        if unique_id in self.hold_progressing_pilots:
            return True
        return False

    def reset_stage(self, step: int):
        """reset_stage(self) generate new stage id and reset starting time to 0"""
        self.starting_time = 0
        self.stage_id = f"{str(step)}_{str(uuid.uuid4().int)[:5]}"

        return

    def activate_stage(self, duration: int, capacity: int):
        self.active = True

@jit
def ceilmult(a,b):
    return math.ceil(a*b)
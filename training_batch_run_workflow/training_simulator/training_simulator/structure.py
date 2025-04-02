import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from typing import List, TypedDict

import mesa


class State(enum.Enum):
    PROGRESSING = "progressing"
    HOLD = "hold"
    LEFT_RAF = "left"
    LEFT_STREAM = "left_stream"
    TRAINING_COMPLETE = "complete"


class Stream(enum.Enum):
    PILOT = "pilot"
    OTHER = "other"


# no longer used for stages other than MIOT
class Stage(enum.Enum):
    MIOT = "miot"
    MAGS = "mags"
    EFT = "eft"
    FJLin = "fjlin"
    BFT = "bft"
    AJT1 = "ajt1"
    AJT2 = "ajt2"
    MELin = "melin"
    MEPT = "mept"
    MEXO = "mexo"
    BRT = "brt"
    ART = "art"
    ARTMAR = "artmar"


@dataclass
class PipelineModelBase(mesa.Model):
    stage_map: dict
    stage_variance_map: dict
    input_params: dict
    init_pilots: dict
    schedules: dict
    start_month: int
    step_count: int


@dataclass
class PilotBase(mesa.Agent):
    unique_id: int
    model: PipelineModelBase
    stage: str
    stage_id: str
    state: State
    stream: Stream
    time_in_stage_state: int
    month_step: int


class StageManager(ABC):
    @abstractmethod
    def drop_out(self, pilot: PilotBase) -> bool:
        ...

    @abstractmethod
    def leave_hold(self, unique_id: int) -> bool:
        ...
        
    @abstractmethod
    def get_next_stage(self):
        ...

    @abstractmethod
    def time_to_progress(self, pilot: PilotBase) -> bool:
        ...

    @abstractmethod
    def activate_stage(self, duration: int, capacity: int):
        ...

class Results(TypedDict):
    RunId: int
    iteration: int
    Step: int
    parameters: str
    AgentID: int
    Stage: str
    State: str
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from typing import List, TypedDict

import mesa


class State(enum.Enum):
    PROGRESSING = "progressing"
    HOLD = "hold"
    LEFT_PIPELINE = "left"
    LEFT_STREAM = "left_stream"
    TRAINING_COMPLETE = "complete"


class Stream(enum.Enum):
    TRAINEE = "trainee"
    OTHER = "other"


# no longer used for stages other than INIT
class Stage(enum.Enum):
    INIT = "init"
    STAGE1 = "stage1"


@dataclass
class PipelineModelBase(mesa.Model):
    stage_map: dict
    stage_variance_map: dict
    input_params: dict
    init_trainees: dict
    schedules: dict
    start_month: int
    step_count: int


@dataclass
class TraineeBase(mesa.Agent):
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
    def drop_out(self, trainee: TraineeBase) -> bool:
        ...

    @abstractmethod
    def leave_hold(self, unique_id: int) -> bool:
        ...
        
    @abstractmethod
    def get_next_stage(self):
        ...

    @abstractmethod
    def time_to_progress(self, trainee: TraineeBase) -> bool:
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
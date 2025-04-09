import json
import random
import uuid
import pandas as pd
from random import randint
from typing import Dict, List, Tuple, Union
import copy
import mesa
import math
import networkx as nx
from numba import jit

from .stages import INIT, PipelineStage
from .structure import TraineeBase, PipelineModelBase, Stage, StageManager, State, Stream


class Trainee(TraineeBase):
    """
    Agents can be initialized into all the stages and states to reflect a realistic system state, where pipeline is not
    empty. If stage and state and not specified then it is assumed they are starting from the first stage of the pipeline.
    """

    def __init__(
        self,
        model: PipelineModelBase,
        stage: str = Stage.INIT.value,
        stage_id: str = "init",
        state: State = State.PROGRESSING,
        time_in_stage_state: int = 0,
        stream=Stream.TRAINEE,
    ) -> None:
        self.unique_id = uuid.uuid4().int
        self.model = model
        self.stage = stage
        self.stage_id = stage_id
        self.state = state
        self.time_in_stage_state = time_in_stage_state
        self.month_step = self.model.start_month
        self.stream = stream

    def _compute_stage_state(self, month_step: int) -> Tuple[str, State]:
        if self.stage not in self.model.stage_map.keys():
            return self.stage, State.TRAINING_COMPLETE
        elif self.state == State.LEFT_PIPELINE:
            return self.stage, self.state

        current_stage = self.model.stage_map[self.stage]

        # agent leaving RAF?
        if current_stage.drop_out(self):
            if current_stage.capacity_dropping > current_stage.yearly_attrition:
                current_stage.yearly_attrition += 1
                return self.stage, State.LEFT_PIPELINE

        # agent not completed my current stage?
        if not current_stage.time_to_progress(self):
            # Need to be in state for longer
            return self.stage, self.state

        #########
        # STAY OR LEAVE HOLDS
        #########

        # agent in hold selected to progress?
        if current_stage.leave_hold(self.unique_id):
            return current_stage.stage_name, State.PROGRESSING
        
        # agent needs to stay in the hold
        if self.state == State.HOLD:
            return current_stage.stage_name, self.state
            
        #########
        # MOVE TO NEXT STAGE
        #########
        
        # Check if this is the start stage then assign next stage from career path
        if type(current_stage).__name__ == "INIT":
            start_node = [n for n, d in self.model.cp.in_degree() if d == 0]
            next_stage = start_node[0]
        else:
            next_stage = current_stage.get_next_stage()

        # agent moves to next stage and in a state of hold
        if next_stage not in self.model.stage_map.keys():
            return next_stage, State.TRAINING_COMPLETE
        return next_stage, State.HOLD

    def _update_stages(self, new_stage: str, new_state: State) -> None:
        ...

    def _update_time_in_state(self, new_stage: str, new_state: State):
        ...

    def step(self) -> None:
        self.month_step = (self.model.step_count + self.model.start_month) % 12
        self.month_step = 12 if self.month_step == 0 else self.month_step
        new_stage, new_state = self._compute_stage_state(self.month_step)

        if new_state != self.state or new_stage != self.stage:
            self.time_in_stage_state = 1
        else:
            self.time_in_stage_state += 1

        self.stage = new_stage
        if self.stage in self.model.stage_map.keys():
            self.stage_id = self.model.stage_map[new_stage].stage_id
        self.state = new_state


class PipelineModel(PipelineModelBase):
    def __init__(self, parameters: Union[dict, str], career_pathway) -> None:
        if isinstance(parameters, str):
            self.parameters = json.loads(parameters)

        iteration_params = copy.deepcopy(self.parameters)
        self.input_params = iteration_params["pipeline"]

        self.running = True
        self.cp = nx.from_dict_of_dicts(
            json.loads(career_pathway), create_using=nx.DiGraph()
        )

        self.stage_map = build_stage_map(parameters=iteration_params, cp=self.cp)
        self.schedules = build_schedule(iteration_params)
        self.start_month = iteration_params["simulation"]["start_month"]
        self.schedule = mesa.time.RandomActivation(self)
        self.step_count = 0
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Stage": lambda a: a.stage,
                "StageId": lambda a: a.stage_id,
                "State": lambda a: a.state.value,
                "Month_Step": lambda a: a.month_step,
            },
            model_reporters={"stagemap": lambda m: m.trainee_nums()},
        )
        self._initialise_stage_attrition_caps(params=iteration_params)
        self._build_warm_start_df(params=iteration_params)
        self._initialise_trainees(params=iteration_params)

    def trainee_nums(self):
        return self.stage_map[Stage.INIT.value].new_trainees

    
    def _max_throughput_func(self, stage: str, stream_ratio: int = 1) -> float:
        """calculate maximum annual thorughput of a stage based on stage's frequency, capcaity and stream ratio

        Args:
            stage (str): stage name
            stream_ratio (int, optional): streaming ratio for post-EFT stages . Defaults to 1.

        Returns:
            float: _description_
        """
        return len(self.schedules[stage]) * self.stage_map[stage].capacity_progressing * stream_ratio

       
    def _find_minimum_course_throughput(self, params: dict, stage: str) -> list:
        """calculate the upstream bottlenecks staing from any stage

        Args:
            params (dict): iteration param set
            stage (str): stage name

        Returns:
            list: list of upstream stages and thier own max thorughputs starting upstream from a specific stage
        """
        prev_stage_throughtput = []
        prev_stage_throughtput.append(self._max_throughput_func(stage))
        for course_pair in list(nx.edge_dfs(self.cp,stage, orientation='reverse')):
            prev_stage_throughtput.append(self._max_throughput_func(course_pair[0]))
        return min(prev_stage_throughtput)


    def _initialise_stage_attrition_caps(self, params: dict) -> None:
        """set all stage's annual attrition caps based on thier attrition rate and max throuphut they will recive inclusive of upstream bottlenecks

        Args:
            params (dict): iteration param set
        """
        for stage in self.cp.nodes:
            if stage != Stage.INIT.value:
                self.stage_map[stage].capacity_dropping = ceilmult(self._find_minimum_course_throughput(params = params, stage = stage), (self.stage_map[stage].drop_out_progressing if stage != 'stage4' else 0))
  

    def _build_warm_start_df(self, params: dict) -> None:
        """build the warm_start df where each col is a stage's cohort, and a row is a month. df shows 36 months of cohorts starting for each course, and each cohort's time progressing value as each month passes.

        Args:
            params (dict): iteration param set
        """
        dfs = []

        for k, _ in params["schedule"].items():
            
            df = pd.DataFrame(index=range(1,37))
            df[f'{k}_start'] = ''
            count = 0
            for i in df.index:
                if i%12 == 0 and 12 in params["schedule"][k]:
                    df.loc[i,f'{k}_start'] = 1
                elif i%12 in params["schedule"][k]:
                    df.loc[i,f'{k}_start'] = 1
                else:
                    df.loc[i,f'{k}_start'] = 0  
            df[f'{k}_cum'] = df.rolling(window=params['pipeline'][k]['time_progressing'], min_periods=1).sum()
            for i in range(1,37):
                if df.loc[i,f'{k}_start'] == 1:
                    df[f'{k}_{count}_left'] = ''
                    df.loc[i, f'{k}_{count}_left'] = params['pipeline'][k]['time_progressing']
                    count += 1
                    
            for i in range(1,37):
                for col in df.loc[:,df.columns.str.endswith('left')].columns:
                    if df.loc[i,col]:
                        if df.loc[i,col] > 1 and i+1 <= 36:
                            df.loc[i+1,col] = df.loc[i,col] -1
                            
            
            df = df.replace('',pd.NA)
            df = df.dropna(axis='columns', how='all')
            dfs.append(df)

            final_df = pd.concat(dfs, axis=1)

            final_df = final_df.tail(12).reset_index(drop=True)
            final_df.index = range(1,13)
            
            self.final_df = final_df
            
        
    def _initialise_trainees(self, params: dict, hold: bool = True, warm_start_month: int = 4) -> None:
        """initialise trainees at start of model run to be a warm start, by inserting x trainees onto each stage and y trainees onto each stage's pre-stage hold if bool is true.
        Documentation - https://defencedigital.atlassian.net/wiki/spaces/DRK/pages/573734938/Warm+Start

        Args:
            params (dict): iteration param set
            hold (bool, optional): should a fraction of warm start trainees also be set to that stage's pre-stage hold. Defaults to True.
            warm_start_month (int, optional): initialisation month, PO wants Drake to start from April. Defaults to 4.
        """
        
        def split_trainees(stage: str, trainees: int, month: int) -> list:
            """split annual throuput of trainees across x courses based on the number of cohorts in flight in warm start month

            Args:
                stage (str): stage name
                trainees (int): number of trainees to split
                month (int): initilisation month

            Returns:
                list: tuples of init trainees to add to a stage with a time prgressing values base don cohort duration from warm start df
            """
            if stage == 'art':
                df = self.final_df.loc[:,self.final_df.columns.str.startswith(stage) & ~self.final_df.columns.str.contains('artmar')]
            else:
                df = self.final_df.loc[:,self.final_df.columns.str.startswith(stage)]
            df = df.loc[month].to_frame().T.dropna(axis='columns', how='all')
            splits = df[f'{stage}_cum'].item()
            
            trainee_starts = []

            for i in range(-1,-1 * int(splits) - 1,-1):
                trainee_starts.append((round(trainees/splits,0), params['pipeline'][stage]['time_progressing'] - df.iloc[:,i].item()))
            return trainee_starts
        
        def total_init_trainees(stage: str, month: int) -> float:
            """compute how many trainees to split in split_trainees()

            Args:
                stage (str): stage name
                month (int): warm start month

            Returns:
                float: trainee count to split
            """
            return round(self.final_df.loc[month, stage +'_cum'] * avg_course_attendance[stage],2)
        
        avg_course_attendance = {}
        warm_start = {}
        for stage in self.cp.nodes:
            if stage != Stage.INIT.value:
                avg_course_attendance[stage] = round(self._find_minimum_course_throughput(params = params, stage = stage)/len(self.schedules[stage]), 2)
                warm_start[stage] = split_trainees(stage, total_init_trainees(stage, warm_start_month), warm_start_month)
        
        
        for stage in self.cp.nodes:
            if stage != Stage.INIT.value:
                stage_id = self.stage_map[stage].stage_id 
                for start in warm_start[stage]:
                    for _ in range(int(start[0])):
                        self.schedule.add(
                            Trainee(
                                self,
                                stage=stage,
                                stage_id=stage_id,
                                state=State.PROGRESSING,
                                time_in_stage_state=start[1],
                            )
                        )
                        # hold code uses %2 to result in ~1/2 as many warm start trainees for a stage being init'ed to a stage's pre-stage hold, modulo number can be changed to increase/decrease voulme of hold trainees
                        if hold and _%2 == 0:
                            self.schedule.add(
                                Trainee(
                                    self,
                                    stage=stage,
                                    stage_id=stage_id,
                                    state=State.HOLD,
                                    time_in_stage_state=1,
                                )     
                            )

    def _add_trainees(
        self,
    ) -> None:
        new_trainees = self.stage_map[Stage.INIT.value].new_trainees

        for _ in range(new_trainees):
            trainee = Trainee(self)
            self.schedule.add(trainee)

    def step(self) -> None:
        for stage in self.cp.nodes:
            self.stage_map[stage].exit_cohort_capacity = 0
            self.stage_map[stage].restream_trainees = {}

            month = (self.step_count + self.start_month) % 12
            month = 12 if month == 0 else month
            
            if month == 4:
                self.stage_map[stage].yearly_attrition = 0
                
            if stage != Stage.INIT.value:
                
                if month in self.schedules[stage]:
                    self.stage_map[stage].reset_stage(self.step_count)
                
                for trainee in self.schedule.agents:
                    if (
                        (stage == trainee.stage)
                        and ("progressing" == trainee.state.value)
                        and (
                            trainee.time_in_stage_state
                            >= self.stage_map[stage].time_progressing
                        )
                    ):
                        self.stage_map[stage].exit_cohort_capacity += 1
                
                #Pseudo-FIFO code             
                if month in self.schedules[stage]:
                    self.stage_map[stage].hold_progressing_trainees = []
                    
                    trainees_to_progress = self.stage_map[stage].capacity_progressing
                    hold_trainees_weighting = []
                    hold_trainees_id = []
                    
                    for trainee in self.schedule.agents:
                        if (stage == trainee.stage) and ('hold' == trainee.state.value):
                            hold_trainees_weighting.append(trainee.time_in_stage_state)
                            hold_trainees_id.append(trainee.unique_id)
                            
                    for _ in range(min(trainees_to_progress, len(hold_trainees_id))):
                        choice = random.choices(hold_trainees_id, hold_trainees_weighting, k=1)[0]
                        self.stage_map[stage].hold_progressing_trainees.append(choice)
                        idx = hold_trainees_id.index(choice)
                        del hold_trainees_id[idx]
                        del hold_trainees_weighting[idx]
                            
        if self.schedule.steps % self.stage_map[Stage.INIT.value].input_rate == 0:
            self._add_trainees()

        self.datacollector.collect(self)
        self.step_count += 1
        self.schedule.step()


def build_stage_map(parameters: dict, cp: nx.DiGraph) -> Dict[str, StageManager]:
    stagemap = {}
    stagemap[Stage.INIT.value] = INIT(**(parameters["pipeline"][Stage.INIT.value]))

    for node in cp.nodes:
        # generate 5 digit int id
        unique_id = f"0_{str(uuid.uuid4().int)[:5]}"
        neighbors = cp.adj[node]
        stage = PipelineStage(
            **(parameters["pipeline"][node]),
            hold_progressing_trainees=[],
            stage_id=unique_id,
            stage_name=node,
            adjacent_stages=neighbors,
        )

        stagemap[node] = stage

    return stagemap

def build_schedule(parameters: dict) -> Dict[str, List]:
    return parameters["schedule"]


def compute_cohort_stratification(
    pipeline_parameters: dict, stage_map: Dict[str, StageManager], cp: nx.DiGraph
):
    # INIT has a different label
    recruiting_duration = pipeline_parameters[Stage.INIT.value]["input_rate"]
    cohort_steps = {
        Stage.INIT.value: [1],
    }
    for current_stage in cp.nodes:
        if current_stage in cohort_steps:
            # Already computed this value, either special case or weird loopback
            continue
        current_duration = pipeline_parameters[current_stage]["time_progressing"]
        cohort_steps[current_stage] = list(
            range(1, current_duration + 1, recruiting_duration)
        )
    return cohort_steps

@jit
def ceilmult(a,b):
    return math.ceil(a*b)


# Project Pipeline Simulator

## Problem Statement
There is a requirement to model flow of actors through a pipeline of tasks/stages and understand how different capacity and timelines of different stages impact the overall pipeline duration.

## The Model
To address the problem a high-level model of the pipeline is implemented to evaluate the effects of different interventions through time. The model is implemented as a stochastic process connecting different stages with states and is taking into account transition probabilities derived from existing datasets.
- Diagram of the Pipeline
Diagram below represents the stages and the flow of the agents through the pipeline.
![diagram](./notebooks/pipeline.png)
To provide required granularity for the pipeline time step of one month is considered.

## Modelling Strategy
- Agent based modelling (ABM) approach has been chosen over traditional optimisation and Markov Chain Monte Carlo techniques due to lack of historical data, complexity of the system and long term time horizon requirements.
In this simulation each actor in the pipeline is represented as an agent. The agents are given rules and initial configurations that govern the interactions with their environment through time. The ABM simulates both individual trajectories and population-level outcomes, which are generated from the bottom up. 
This model is governed by rules as well as stochastic probability i.e. when reaching end of Stage2, drop out with probability of x%. Stochasticity in the model allows capturing emergent phenomena from the interactions of the individual agents with their environment. In this simulation the agents do not interact with each other.
Note that stochasticity allows to detect emergent behaviour (any behaviour that has not been explicitly coded) that arises from agent/environment interactions.

### Key Input Parameters
steps are the time units that are used to measure the pipeline duration.
- **Number of trainees:** number of new trainees (agents) entering the system 
- **Stage Duration**  maximum length of time required for agents to completed given stage. 
- **Capacity:** max number of trainees permitted in one stage/course
- **Duration** duration of course in months. 
- **Attrition:** max percentage of trainees leaving stage/state.

### Key Outputs

- number of agents in each stage/state at any time step throughout the pipeline.
- average number of agents on hold at given stage.course.
- average number of agents that left the stage/course
- average number of agents completed given training pathway


### Concept of Stages and States in the Model 
The journey of agents is modelled as a sequence of stages that each agent has to go through. There are 6 stages in the pipeline - STage0, Stage1, Stage2, Stage3, Stage4 and Stage5.

The model assumes that excluding initial stage during any stage each agent can be in three different states: 
- **progressing:** progressing through a stage in the pipeline
- **hold:** being on hold before entering next stage in the pipeline
- **left:** drop out and leaving the pipeline

In each stage all the states (excluding 'left' state) have  duration feature that is the minimum length of time before agent can change the stage/state. Furthermore, **Progressing** state has capacity feature that is the fixed number of agents that can be progressing in that stage at any point of time.
Note that in hold stage once duration has been exceeded the agent will be automatically transferred to left state. The transition mechanism between states is governed by attrition rates. There is no transition from 'left' state so the agent will be removed from progressing the pipeline.
The flow between state/stages  can be seen on a diagram above. 


### Advantages
- Easy to implement and test hypotheses in the absence of aggregate data and understanding the overall complex system behaviour.
- Extensibility and Flexibility: Agent based modelling approach will allow to easily incorporate different features for the agents such as age, gender etc and add new rules if required in the future. There is also a potential to introduce interaction between agents.
- Provision of  data driven outcomes on testing different interventions and scenarios.
- Long term forecasting potential.

### Limitations
- Model cannot capture more granular issues that are highly specific to one or the other stage in the pipeline (i.e. the availability of trainers and aircraft in OCU)
- Model is not linked to live data sets and hence only represents a snapshot of when the parameters where last estimated.
- Outputs are stochastic so this can be challenging to validate.

## Code Structure
- Mesa
Mesa framework was used to implement aircrew training ABM. Mesa is an Apache2 licensed framework in Python. It allows developers to quickly create models using built-in core components and analyze their results using Python's data analysis tools. 
- Stage Managers


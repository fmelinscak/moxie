#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunner

from model import OptimSciEnv, make_discounted_sum_utility

# Set up simulation constants
N_LABS = 5
STEP_RESOURCES = 20
MAX_STEPS = 50 # How many research cycles per simulation
ITERATIONS = 200 # How many replicates of each scenario (parameter set)
P_STUDY_PUBLISHED = 1.0
P_REPLICATION = 0.20 # Only used when replication_strategy != "none"
DISCOUNT_FACTOR = 0.97 # Implies ~20% utility after 50 cycles, 
                       # compared with present utility
STUDY_INTAKE_CAPACITY = 2 * (N_LABS - 1) 
# Note: study intake capacity is set such that all the labs get all new
# studies, even in the worst case (lab goes first one cycle, and last in next,
# and all the labs in between publish a study)
LANDSCAPE_TYPES = ["kind", "wicked"]
DESGIN_STRATEGIES = ["random", "random_ofat"]
REPLICATION_STRATEGIES = ["random", "targeted"] # Only used in sims with repl.


# Create discounted cumulative utility function for scoring simulations
discounted_utility_func = make_discounted_sum_utility(DISCOUNT_FACTOR)

# Set up model reporter functions
model_reporters={
        "final_cumsum_true_util": lambda m: m.cumsum_true_util,
        "final_disc_cumsum_true_util": discounted_utility_func,
        "all_run_data": lambda m: m.datacollector
    }


# *** Run the simulations *without* replication***

# Set fixed parameters of the model
fixed_params = {
    "n_labs": N_LABS,
    "step_resources": STEP_RESOURCES,
    "p_replication": 0.0,
    "replication_strategy" : "none",
    "p_study_published" : P_STUDY_PUBLISHED,
    "study_intake_capacity": STUDY_INTAKE_CAPACITY
}

# Set variable parameters of the model
variable_params = {
    "landscape_type": LANDSCAPE_TYPES,
    "design_strategy": DESGIN_STRATEGIES
}

# Set up batch
batch_run = BatchRunner(
    OptimSciEnv,
    fixed_parameters=fixed_params,
    variable_parameters=variable_params,
    iterations=ITERATIONS,
    max_steps=MAX_STEPS,
    model_reporters=model_reporters
)

# Run batch and obtain the data
batch_run.run_all()
df_without_replication = batch_run.get_model_vars_dataframe()


# *** Run the simulations *with* replication***

# Set fixed parameters of the model
fixed_params = {
    "n_labs": N_LABS,
    "step_resources": STEP_RESOURCES,
    "p_replication": P_REPLICATION,
    "p_study_published" : P_STUDY_PUBLISHED,
    "study_intake_capacity": STUDY_INTAKE_CAPACITY
}

# Set variable parameters of the model
variable_params = {
    "landscape_type": LANDSCAPE_TYPES,
    "design_strategy": DESGIN_STRATEGIES,
    "replication_strategy": REPLICATION_STRATEGIES
}

# Set up batch
batch_run = BatchRunner(
    OptimSciEnv,
    fixed_parameters=fixed_params,
    variable_parameters=variable_params,
    iterations=ITERATIONS,
    max_steps=MAX_STEPS,
    model_reporters=model_reporters
)

# Run batch and obtain the data
batch_run.run_all()
df_with_replication = batch_run.get_model_vars_dataframe()


# *** Merge two datasets and save the merged one ***
df_with_replication["Run"] += len(df_without_replication) # Renumber runs
df_all_sims = pd.concat([df_without_replication, df_with_replication], \
    ignore_index=True, sort=False)

# Save the pickle and CSV to disk
df_all_sims.to_pickle(f"moxie_sims_{ITERATIONS}iter.pickle")
df_all_sims.to_csv(f"moxie_sims_{ITERATIONS}iter.csv")
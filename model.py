from collections import defaultdict, namedtuple
import copy

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
from scipy.stats import multivariate_normal


class EpistemicLandscape:
    """
    Represents the (noisy) epistemic landscape in which agents are searching
    for a peak in utility.
    """
    def __init__(self, model, utility_func, noise_sigma, dim):
        self._model = model
        self._utility_func = utility_func
        self._noise_sigma = noise_sigma
        self._dim = dim

    def eval_solution(self, solution):
        utility = self._utility_func(solution)
        noise = self._model.random.gauss(0, self._noise_sigma)
        return  utility + noise

    def get_dim(self):
        return self._dim


def kind_utility_func(x):
    """
    An easy-to-optimize 2-dimensional utility function with a single,
    non-correlated gaussian peak.
    """
    rv = multivariate_normal([0.7, 0.3], [[0.25, 0.0], [0.0, 0.1]])
    return rv.pdf(x)


def wicked_utility_func(x):
    """
    A difficult-to-optimize 2-dimensional utility function (embedded within a larger number
    of irrelevant dimensions) with a broad misleading non-optimal gaussian, and a narrow,
    highly correlated optimal gaussian peak
    """
    # TODO: implement wicked utility function
    pass


class Study:
    def __init__(self, study_id, lab_id, is_published, study_type,
        target_dims, study_results):
        self.study_id = study_id
        self.lab_id = lab_id
        self.is_published = is_published
        self.study_type = study_type
        self.target_dims = target_dims
        self.study_results = study_results

    def __repr__(self):
           return (f'{self.__class__.__name__}('
               f'{self.__dict__!r})')


class Knowledgebase:
    def __init__(self):
        self._accepted_studies = {}
        self._solution_obs_utilities = defaultdict(list)
        self._solution_summary = defaultdict(dict)

    def __repr__(self):
           return (f'{self.__class__.__name__}('
               f'{self._accepted_studies!r})')

    def __len__(self):
        return len(self._accepted_studies)

    def receive_study(self, study):
        raise NotImplementedError

    def process_study_solutions(self, study):
         for solution, obs_utilities in study.study_results.items():
            # Compute summaries of the current solution utilities
            obs_n = len(obs_utilities) # Count
            obs_mean = np.mean(obs_utilities) # Mean
            obs_ssq = np.sum((obs_utilities - obs_mean)**2) # Sum of squared distances from mean
            # Update summaries if solution previously evaluated
            if solution in self._solution_obs_utilities:
                old_n = self._solution_summary[solution]["n"]
                old_mean = self._solution_summary[solution]["mean"]
                old_ssq = self._solution_summary[solution]["ssq"]
                total_n = old_n + obs_n
                total_mean = (old_n*old_mean + obs_n*obs_mean) / total_n
                # Update SSQ using Chen's parallel algorithm
                # see: https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=928348206#Parallel_algorithm
                total_ssq = old_ssq + obs_ssq + (old_mean-obs_mean)**2 * old_n * obs_n / total_n
            else: # Create summaries if solution new
                total_n = obs_n
                total_mean = obs_mean
                total_ssq = obs_ssq
            # Calculate derived summaries
            total_sd = np.sqrt(total_ssq/(total_n-1))
            total_se = total_sd / np.sqrt(total_n)
            # Store summaries
            self._solution_summary[solution]["n"] = total_n
            self._solution_summary[solution]["mean"] = total_mean
            self._solution_summary[solution]["ssq"] = total_ssq
            self._solution_summary[solution]["sd"] = total_sd
            self._solution_summary[solution]["se"] = total_se 
            # Add observed utilities to solution-wise dict
            self._solution_obs_utilities[solution].extend(obs_utilities)

    def get_study_ids(self):
        return self._accepted_studies.keys()

    def get_study_by_id(self, study_id):
        return self._accepted_studies[study_id]


class LocalKnowledgebase(Knowledgebase):
    def __init__(self):
        super().__init__()

    def receive_study(self, study):
        # Add study to local kbase
        self._accepted_studies[study.study_id] = study
        # Process study solutions (add to database and summary)
        self.process_study_solutions(study)   


class GlobalKnowledgebase(Knowledgebase):
    def __init__(self, model):
        super().__init__()
        self._next_study_id = 0
        self.model = model

    def receive_study(self, study):
        # Assign unique id to submitted study if necessary
        if study.study_id is None:
            study.study_id = self._next_study_id
            self._next_study_id += 1

        # Accept (stochastically) study to kbase
        if self.model.random.random() < self.model.p_study_published:
            study.is_published = True
            self._accepted_studies[study.study_id] = study
            # Process study solutions (add to database and summary)
            self.process_study_solutions(study)
        else:
            study.is_published = False
        
        return study.study_id, study.is_published


class OptimSciEnv(Model):
    """
    A model of an optimization-centric research environment
    """
    def __init__(self, n_labs, step_resources, landscape_type, \
        design_strategy, study_intake_capacity, p_study_published, seed=None):
        super().__init__()
        self.n_labs = n_labs
        self.step_resources = step_resources
        self.study_intake_capacity = study_intake_capacity
        self.p_study_published = p_study_published
        
        # Initialize global knowledge base (i.e. record of published studies)
        self.global_kbase = GlobalKnowledgebase(self)

        # Initialize the epistemic landscape ('kind' or 'wicked')
        if landscape_type == "kind":
            noise_sigma = 0.1
            utility_func = kind_utility_func
            dim = 2
        if landscape_type == "wicked":
            noise_sigma = 0.3
            utility_func = wicked_utility_func
            dim = 16

        self.landscape = EpistemicLandscape(self, utility_func, noise_sigma, dim)

        # Initialize the schedule, create labs (agents), and add them to the schedule
        self.schedule = RandomActivation(self)

        for i in range(self.n_labs):
            lab = Lab(i, self, design_strategy)
            self.schedule.add(lab)

        # Initialize the data collector and collect initial data
        self.datacollector = DataCollector()
        self.datacollector.collect(self)

    def step(self):
        # Run single schedule step
        self.schedule.step()

        # Collect data
        self.datacollector.collect(self)


class Lab(Agent):
    """
    An agent representing a research lab (team of researchers or single researcher)
    """

    def __init__(self, lab_id, model, design_strategy):
        self.lab_id = lab_id
        super().__init__(lab_id, model)
        self._design_strategy = design_strategy
        self._local_kbase = LocalKnowledgebase()
        self._balance_resources = 0
        self._landscape_dim = model.landscape.get_dim()

    def step(self):
        print("\nAgent {} activated".format(self.lab_id))
        self.request_resources()
        self.update_local_kbase()
        new_study = self.conduct_study()
        self.submit_study(new_study)
    

    def request_resources(self):
        self._balance_resources += self.model.step_resources
        print("Got {new_resources} new resources. New balance is {balance}"\
            .format(new_resources=self.model.step_resources, balance=self._balance_resources))

    def update_local_kbase(self):
        # Find new studies in global kbase
        global_study_ids = self.model.global_kbase.get_study_ids()
        local_study_ids = self._local_kbase.get_study_ids()
        new_study_ids = [id for id in global_study_ids if not id in local_study_ids]

        # Stochastically incorporate new studies into local kbase
        n_retain = min(len(new_study_ids), self.model.study_intake_capacity)
        retained_study_ids = self.random.sample(new_study_ids, n_retain)
        for id in retained_study_ids:
            retained_study = self.model.global_kbase.get_study_by_id(id)
            self._local_kbase.receive_study(retained_study)

    def conduct_study(self):
        print("Conducting study...")
        # Select solutions to test and allocate resources to testing them
        if self._design_strategy == "random":
            study_plan = self.random_design()

        # Spend resources on evaluating solutions against the landcape
        study_results = {}
        for (solution, resources) in study_plan.items():
            utilities = [self.model.landscape.eval_solution(solution) \
                for _ in range(resources)]
            study_results[solution] = utilities
            self._balance_resources -= resources
        
        print("Study results:\n", study_results)

        # Pack results into a study (no study ID before submitting to global kbase)
        new_study = Study(study_id=None, lab_id=self.lab_id, is_published=False,
            study_type="original", target_dims=None, study_results=study_results)
        print("Study:\n", new_study)
        return new_study

    def submit_study(self, study):
        # Submit study to global knowledgebase
        print("Submitting study...\n")
        study.study_id, study.study_published = \
            self.model.global_kbase.receive_study(study)
        print("Global kbase:\n", self.model.global_kbase)

        # Add study (now with id) to local knowledgebase
        self._local_kbase.receive_study(study)
        print("Local kbase:\n", self._local_kbase)

    
    def random_design(self):
        # Generate a random solution
        solution = tuple([self.random.random() \
            for _ in range(self._landscape_dim)])
        
        # Create study plan
        study_plan = {solution: self._balance_resources}
        study_plan = {(0.3, 0.7): self._balance_resources}

        return study_plan


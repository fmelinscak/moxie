from collections import namedtuple
import copy

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from scipy.stats import multivariate_normal


class EpistemicLandscape:
    """
    Represents the (noisy) epistemic landscape in which agents are searching
    for a peak in utility.
    """
    def __init__(self, model, utility_func, noise_sigma, dim):
        self.model = model
        self.utility_func = utility_func
        self.noise_sigma = noise_sigma
        self.dim = dim

    def eval_solution(self, solution):
        utility = self.utility_func(solution)
        noise = self.model.random.gauss(0, self.noise_sigma)
        return  utility + noise

    def get_dim(self):
        return self.dim


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
        self.accepted_studies = []

    def receive_study(self, study):
        raise NotImplementedError


class LocalKnowledgebase(Knowledgebase):
    def __init__(self):
        super().__init__()

    def receive_study(self, study):
        self.accepted_studies.append(study)


class GlobalKnowledgebase(Knowledgebase):
    def __init__(self):
        super().__init__()
        self.next_study_id = 0
    
    def receive_study(self, study):
        # Assign unique id to submitted study if necessary
        if study.study_id is None:
            study.study_id = self.next_study_id
            self.next_study_id += 1

        # Accept study to kbase
        # TODO: make this step non-certain
        study.is_published = True
        self.accepted_studies.append(study)
        
        return study.study_id, study.is_published


class OptimSciEnv(Model):
    """
    A model of an optimization-centric research environment
    """
    def __init__(self, n_labs, step_resources, landscape_type, design_strategy):
        super().__init__()
        self.n_labs = n_labs
        self.step_resources = step_resources
        
        # Initialize global knowledge base (i.e. record of published studies)
        self.global_kbase = GlobalKnowledgebase()

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
        self.design_strategy = design_strategy
        self.local_kbase = LocalKnowledgebase()
        self.balance_resources = 0
        self.landscape_dim = model.landscape.get_dim()

    def step(self):
        print("Agent {} activated".format(self.lab_id))
        self.request_resources()
        self.update_local_kbase()
        new_study = self.conduct_study()
        self.submit_study(new_study)
    

    def request_resources(self):
        self.balance_resources += self.model.step_resources
        print("Got {new_resources} new resources. New balance is {balance}"\
            .format(new_resources=self.model.step_resources, balance=self.balance_resources))

    def update_local_kbase(self):
        # TODO: make the transfer imperfect and use the object interface
        self.local_kbase.accepted_studies = copy.deepcopy(self.model.global_kbase.accepted_studies)

    def conduct_study(self):
        print("Conducting study...")
        # Select solutions to test and allocate resources to testing them
        if self.design_strategy == "random":
            study_plan = self.random_design()

        # Spend resources on evaluating solutions against the landcape
        study_results = {}
        for (solution, resources) in study_plan.items():
            utilities = [self.model.landscape.eval_solution(solution) \
                for _ in range(resources)]
            study_results[solution] = utilities
            self.balance_resources -= resources
        
        print("Study results:\n", study_results)

        # Pack results into a study (no study ID before submitting to global kbase)
        new_study = Study(study_id=None, lab_id=self.lab_id, is_published=False,
            study_type="novel", target_dims=None, study_results=study_results)
        print("Study:\n", new_study)
        return new_study

    def submit_study(self, study):
        # Submit study to global knowledgebase
        print("Submitting study:\n")
        study.study_id, study.study_published = \
            self.model.global_kbase.receive_study(study)
        print("Global kbase:\n", self.model.global_kbase.accepted_studies)

        # Add study (now with id) to local knowledgebase
        self.local_kbase.receive_study(study)
        print("Local kbase:\n", self.local_kbase.accepted_studies)

    
    def random_design(self):
        # Generate a random solution
        solution = tuple([self.model.random.random() \
            for _ in range(self.landscape_dim)])
        
        # Create study plan
        study_plan = {solution: self.balance_resources}

        return study_plan


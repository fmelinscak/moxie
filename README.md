# `moxie` - a computational model of optimization-centric science

`moxie` is an agent-based model of optimization-centric science, written in Python using the [Mesa](https://github.com/projectmesa/mesa) package.
The model conceptualizes science as a navigation problem on an "epistemic landscape", i.e., an optimization problem.
Each point on this landscape is one possible solution to the scientific problem, and the "elevation" of the point represents the utility of the solution (NB: the landscapes can also be high dimensional).
Such research programs can most often be found in applied sciences dealing with complex biological, social, or technological systems, and in which progress is made in large degree through trial-and-error.

## Installation

All the necessary dependencies of the model can be installed using [Conda](https://docs.conda.io/en/latest/).

After cloning the project repository, create the Conda environment using the terminal:
```
conda env create -f environment.yml
```

## Getting started

After creating the Conda environment, before running the simulations, you need to activate the environment using:
```
conda activate moxie
```

After that you can run the example script using:
```
python3 run_simulations.py
```
Note, however, that the simulations may take a few hours to complete.
If you wish to directly proceed to analyze simulation results, you can download simulation data from [OSF](https://osf.io/hxecg/).
Once you have the simulation data, you can use the Jupyter Notebook `analyze_simulations.ipynb` to analyze it.
For this, you can either install Jupyter into the Conda `moxie` environment, or you can use Jupyter from the base Conda environment, but you will need the [`nb_conda_kernels`](https://github.com/Anaconda-Platform/nb_conda_kernels) extension for Jupyter.

## Contact

Filip Melinscak (<filip.melinscak@gmail.com>)

## License
[MIT](https://choosealicense.com/licenses/mit/)

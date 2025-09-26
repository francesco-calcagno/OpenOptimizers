![Python](https://img.shields.io/badge/python-%3E%3D%203.13.15-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![GitHub all releases](https://img.shields.io/github/downloads/francesco-calcagno/OpenOptimizers/total)

<div align="center">
    <img src="./img/OpenOptimizers-logo.png" width="600" />
</div>

# OpenOptimizers
OpenOptimizers is a repository collecting minimization algorithms fully implemented in Python.

## Table of Contents

1. [Getting Started](#start)
    - [Installation](#installation)
    - [Requirements](#requirements)
    - [Basic usage](#run)
    - [Implemented methods](#implemented)
    - [Tutorial notebooks](#exercises)
2. [Contribute](#contribute)
3. [How to cite](#cite)

## Getting Started <a name="start"></a>

### Installation <a name="installation"></a>
Download the most-updated version of the `OpenOptimizers` repository, locating it in your desired directory.
```
mkdir $PATH/OpenOptimizers
git clone git@github.com:/francesco-calcagno/OpenOptimizers.git
```
Now, you are ready to use it!

### Requirements <a name="requirements"></a>
The following `python` libraries are required:
- `numpy`

Optional (for running tutorial/exercise code):
- `matplotlib`
- `rdkit`

### Basic usage <a name="run"></a>
`OpenOptimizers` is a modular repository. All optimizers are handled by a master class `OptimizersWheel()`, which is contained in `master.py`.
`OptimizersWheel()` defines all arguments that are common to all methods:
```
class OptimizersWheel:
    
    def __init__(
        self, 
        parameters,
        bounds=(-1000, 1000), 
        maxiter=100, 
        random_samples=None, 
        convergence_threashold=0.0001,
        window_epochs=3,
        random_state=168548645, 
        verbose=False, 
        *args, 
        **kwargs
        ):
        
        '''
        Args:
            parameters (int, list): number of parameters to generate guess parameters or list of
                                    user-decided parameters.
            bounds (tuple): lower and upper parameters limit. Default: (-1000, 1000).
            maxiter (int): maximum number of iterations. Default: 100.
            random_samples (bool, int): number of random samples to generate guess parameters.
                                    This is activated only if type(parameters)==int. Default: None.
            convergence_threashold (float): threashold value to reach convergence. Default: 0.0001.
            window_epochs (int): number of epochs to compute standard deviation value to compare to
                                    the convergence_threashold. Default: 3.
            random_state (int): seed value for pseudo-random number generators. Default: 168548645.
            verbose (bool): instruction to print extra messages.
        '''
...
...
...
```

Optimizers are, thus, sub-classes that leverage `OptimizersWheel` adding method-specific arguments. Each optimizer is organized in its own file, e.g. `GD.py`.
Optimizers share the same implementation scheme:
1. read function's parameters as lists
2. contain the `minimize()` method
3. return the sub-optimal solution, its related parameters, and a dictionary as call-back function.


#### Example
Given a function `f(x)`:
```
def f(x):
  '''
  Args:
      x (list): list of parameters
  Returns:
      val (float): loss value
  '''
  val = x[0]**2
  return val
```

The minimization task is completed with the `GD` optimizer using the following syntax:

```
from src.GD import GD as gd

# Optimizers settings
parameters = [1.3] 
bounds = (0.2, 1.4) 
maxiter = 10000
convergence_threashold = 0.0001
learning_rate = 0.01
epsilon = 1e-8

opt = gd(
    # OptimizersWheel()'s arguments
    parameters=parameters,  #list of initial parameters
    bounds=bounds,          #constrains for constrained optimization
    maxiter=maxiter,        #maximum number of iterations
    random_samples=None,    #random initialization
    convergence_threashold=convergence_threashold,  #convergence threashold
    window_epochs=3,        #number of iterations used to compute stdv on cost function's property for convergence
    random_state=168548645, #seed for pseudo random numbers generation 
    verbose=False,
    # GD's arguments
    learning_rate=learning_rate,
    epsilon=epsilon,
)

y_best, x_best, history = opt.minimize(f)
```

### Tutorial notebooks <a name="exercises"></a>
Two tutorial Jupyter Notebooks are provided to showcase how different methods implemented in `OpenOptimizerz` perform in minimization tasks. These notebooks are meant for undergrad students who are already familiar with Python. 


## Implemented methods <a name="implemented"></a>
Currently, the following optimizers are implemented:
1. Pattern Search (PS) --> `from src.PS import PS as ps`
2. Gradient Descendent (GD) --> `from src.GD import GD as gd`
3. Adam --> `from src.Adam import Adam as adam`

## Contribute <a name="contribute"></a>
`OpenOptimizers` is an open-source project and everybody is welcome to contribute.
If you have (or want to) implement an optimizer that is missing, please either create a pull request or contact Francesco Calcagno (francesco.calcagno@unibo.it).

## How to cite <a name="cite"></a>
If you use `OpenOptimizer` in a project or in a scientific paper, please cite the following reference:
```
@software{calcagno2025OpenOptimizers,
  author       = {Calcagno, Francesco},
  title        = {OpenOptimizers},
  year         = 2025,
  publisher    = {GitHub},
  version      = {1.0.0},
  url          = {https://github.com/francesco-calcagno/OpenOptimizers/}
}
```



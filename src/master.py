import numpy as np
import random

'''
This module handles the base machinery
that is common to all optimizers. 
'''

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

        super().__init__(*args, **kwargs)

        assert isinstance(parameters, (int, list)), f"'parameters' --> Expected int or list, got {type(parameters).__name__}"
        assert isinstance(bounds, tuple), f"'bounds' --> Expected tuple, got {type(bounds).__name__}"
        assert isinstance(maxiter, int), f"'maxiter' --> Expected int, got {type(maxiter).__name__}"
        assert isinstance(random_samples, (int, type(None))), f"'random_samples' --> Expected int or NoneType, got {type(random_samples).__name__}"
        assert isinstance(convergence_threashold, float), f"'convergence_threashold' --> Expected float, got {type(convergence_threashold).__name__}"
        assert isinstance(window_epochs, int), f"'window_epochs' --> Expected int, got {type(window_epochs).__name__}"
        assert isinstance(random_state, int), f"'random_state' --> Expected int, got {type(random_state).__name__}"
        assert isinstance(verbose, bool), f"'verbose' --> Expected bool, got {type(verbose).__name__}"

        self.parameters = parameters
        self.bounds = bounds
        self.lb = np.array(self.bounds[0], dtype=float)
        self.ub = np.array(self.bounds[1], dtype=float)

        if isinstance(self.parameters, int):
            self.n_parameters = self.parameters
            self.parameters = np.random.uniform(low=self.lb, high=self.ub, size=self.n_parameters).tolist()
            assert isinstance(random_samples, (int)), f"'random_samples' --> Expected int, got {type(random_samples).__name__}"
            assert random_samples > 0, f"'random_samples' must be > 0"

        else:
            self.n_parameters = len(parameters)
            self.parameters = np.array(parameters).tolist()
            random_samples = None
        
        self.maxiter = maxiter
        self.random_samples = random_samples
        self.convergence_threashold = convergence_threashold
        self.window_epochs = window_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.iteration = 0
        self.converged = False

        np.random.seed(self.random_state)
        random.seed(self.random_state)



from src.master import OptimizersWheel
import numpy as np
import random

'''
This module implements the Adam optimizer.

Please, always refer and credit the original work describing Adam: 
D.P. Kingma and J. Ba, arXiv 2014, https://arxiv.org/abs/1412.6980.
'''

class Adam(OptimizersWheel):
    
    def __init__(
        self, 
        learning_rate=0.01, 
        beta1 = 0.9,
        beta2 = 0.999,
        *args, 
        **kwargs
        ):

        '''
        Args:
            learning_rate (float): learning rate of the optimizer. Default: 0.01.
            beta1 (float): hyperparameter used for computing running averages. 
                        Default: 0.9.
            beta2 (float): hyperparameter used for computing running averages. 
                        Default: 0.999.
        '''
        
        super().__init__(*args, **kwargs)
        
        assert isinstance(learning_rate, float), f"'learning_rate' --> Expected float, got {type(learning_rate).__name__}"
        assert learning_rate < 1., "Fatal error! learning_rate must be < 1."

        assert isinstance(beta1, float), f"'beta1' --> Expected float, got {type(beta1).__name__}"
        assert beta1 < 1., "Fatal error! beta1 must be < 1."

        assert isinstance(beta2, float), f"'beta2' --> Expected float, got {type(beta2).__name__}"
        assert beta2 < 1., "Fatal error! beta2 must be < 1."

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = np.zeros_like(self.parameters, dtype=float)
        self.v = np.zeros_like(self.parameters, dtype=float)
        self.eps = 1e-8


    def numerical_gradient(
        self, 
        objective_function
        ):

        '''
        Numerical gradient calculation.

        Args:
            objective_function (func): target function to minimize.
        Returns:
            gradient (np.array): gradient value.
        '''

        gradient = np.zeros_like(self.parameters)

        # TODO: parallelize this loop
        for i in range(len(self.parameters)):
            tmp = self.parameters[i]
            
            # displacement
            self.parameters[i] = tmp + self.eps
            y_p = objective_function(self.parameters)
            self.parameters[i] = tmp - self.eps
            y_m = objective_function(self.parameters)

            gradient[i] = (y_p - y_m) / (2 * self.eps)

            self.parameters[i] = tmp

        return gradient


    def minimize(
        self, 
        objective_function, 
        ):

        '''
        Args:
            objective_function (func): target function to minimize.
        Returns:
            y_best
            parameters_best (np.array): parameters that returns the (sub)optimal 
                        solution, i.e., self.y_best.
            history (dic): dictionary keeping trace of explored parameters,
                        y-values, and information.
        '''

        self.parameters = np.clip(self.parameters, self.lb, self.ub)

        if self.verbose:
            print("Initial random search of sub-optimal initial parameters...")

        self.y_best = float(objective_function(self.parameters))
        history = {"x": [self.parameters.copy()], "y": [self.y_best], "info": [f"gradient = NaN"]}

        ###################################
        ###       RANDOM SAMPLING       ###
        ###################################        
        if isinstance(self.random_samples, (int)):
            for random_step in range(self.random_samples-1): # "-1" is because of the first y calculation above
                random_guesses = np.random.uniform(low=self.lb, high=self.ub, size=self.n_parameters).tolist()
                y = float(objective_function(random_guesses))
                history["x"].append(random_guesses.copy())
                history["y"].append(y)
                history["info"].append(f"gradient = NaN")

                if y <= self.y_best:
                    self.y_best = y
                    self.parameters = random_guesses
        ###################################
        ###          FINISHED           ###
        ###       RANDOM SAMPLING       ###
        ###################################                    

        if self.verbose:
            if isinstance(self.random_samples, (int)):
                print(f"\nBest random initial parameters: {self.parameters} with y value: {self.y_best}")
            else:
                print(f"\nInitial parameters: {self.parameters} with y value: {self.y_best}")

        self.parameters = np.array(self.parameters, dtype=float)

        for iteration in range(1, self.maxiter+1):

            gradient = self.numerical_gradient(objective_function)

            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

            m_hat = self.m / (1 - self.beta1**iteration)
            v_hat = self.v / (1 - self.beta2**iteration)

            self.parameters -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            self.parameters = np.clip(self.parameters, self.lb, self.ub)

            y = float(objective_function(self.parameters))

            self.iteration += 1

            history["x"].append(self.parameters.copy())
            history["y"].append(y)
            history["info"].append(f"gradient = {gradient}")

            ###################################
            ###     CONVERGENCE CHECK       ###
            ###################################
            try:
                recent_values = history["y"][-self.window_epochs*self.n_parameters]
                convergence_value = float(np.std(recent_values))
            except:
                continue

            if self.iteration == self.maxiter+1:
                self.converged = True
                if self.verbose:
                    print("Maximum iteration reached.")

            elif convergence_value <= self.convergence_threashold and iteration > int(0.2*(self.maxiter+1)):
                self.converged = True
                if self.verbose:                    
                    print("Convergence threashold reached.")

            if self.converged:
                break
        
        return y, self.parameters, history



from src.master import OptimizersWheel
import numpy as np
import random

'''
This module implements the Gradient Descent optimizer.
'''

class GD(OptimizersWheel):
    
    def __init__(
        self, 
        learning_rate=0.01, 
        eps=1e-8,
        *args, 
        **kwargs
        ):

        '''
        Args:
            learning_rate (float): learning rate of the optimizer. Default: 0.01.
            eps (float): +/- displacement for numerical calculation of the gradient.
                        Default: 0.00000001.
        '''
        
        super().__init__(*args, **kwargs)
        
        assert isinstance(learning_rate, float), f"'learning_rate' --> Expected float, got {type(learning_rate).__name__}"
        assert learning_rate < 1., "Fatal error! learning_rate must be < 1."

        assert isinstance(eps, float), f"'eps' --> Expected float, got {type(eps).__name__}"

        self.learning_rate = learning_rate
        self.eps = eps


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
        
        #TODO: parallelize this loop
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
            self.parameters = np.clip(self.parameters - self.learning_rate * gradient, self.lb, self.ub)
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
        y_best = min(history['y'])     
        idx = history["y"].index(y_best) 
        parameters_best = history["x"][idx]

        return y, self.parameters, history




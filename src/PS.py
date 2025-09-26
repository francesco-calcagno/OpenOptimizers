from src.master import OptimizersWheel
import numpy as np
import random

'''
This module implements the coordinate-wise Pattern Search optimizer.
'''

class PS(OptimizersWheel):
    
    def __init__(
        self, 
        beta=0.99,
        *args, 
        **kwargs
        ):

        '''
        Args:
            beta (float): this is the damping value to decrease the step size
                        of the optimizer. Namely, step_size *= beta.
                        Default: 0.99.
        '''
        
        super().__init__(*args, **kwargs)
        
        assert isinstance(beta, float), f"'beta' --> Expected float, got {type(beta).__name__}"
        assert beta > 0. and beta < 1., "Fatal error! Beta must be [0., 1.]"

        self.beta = beta

    

    def minimize(
            self, 
            objective_function
            ):
        
        '''
        Args:
            objective_function (func): target function to minimize.
        Returns:
            self.y_best
            self.parameters (np.array): parameters that returns the (sub)optimal 
                        solution, i.e., self.y_best.
            history (dic): dictionary keeping trace of explored parameters,
                        y-values, and information.
        '''
        
        step_size = abs(self.lb)+abs(self.ub)/2
        self.parameters = np.clip(self.parameters, self.lb, self.ub)        
        if self.verbose:
            print("Initial random search of sub-optimal initial parameters...")
        
        self.y_best = float(objective_function(self.parameters))
        history = {"x": [self.parameters.copy()], "y": [self.y_best], "info": [f"step size = {step_size}"]}

        ###################################
        ###       RANDOM SAMPLING       ###
        ###################################        
        if isinstance(self.random_samples, (int)):
            for random_step in range(self.random_samples-1): # "-1" is because of the first y calculation above
                random_guesses = np.random.uniform(low=self.lb, high=self.ub, size=self.n_parameters).tolist()
                y = float(objective_function(random_guesses))
                history["x"].append(random_guesses.copy())
                history["y"].append(y)
                history["info"].append(f"step size = {step_size}")

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
        
        ###################################
        ###          MAIN LOOP          ###
        ###################################
        for iteration in range(1, self.maxiter+1):    
            self.iteration += 1
            accepted_plus = False
            indices = list(range(self.n_parameters))
            random.shuffle(indices)
            for index in indices:
                # Direction param + step size
                x_new = self.parameters.copy(); x_new[index] = float(np.clip(self.parameters[index]+step_size, self.lb, self.ub))
                y = float(objective_function(x_new))
                delta = y - self.y_best

                if delta < 0.:
                    self.parameters, self.y_best = x_new, y
                    history["x"].append(self.parameters.copy())
                    history["y"].append(float(y))
                    history["info"].append(f"step size = {step_size}")
                    accepted_plus = True
                    

            accepted_minus = False
            random.shuffle(indices)
            #TODO: parallelize this loop
            for index in indices:
                # Direction param - step size
                x_new = self.parameters.copy(); x_new[index] = float(np.clip(self.parameters[index]-step_size, self.lb, self.ub))
                y = float(objective_function(x_new))
                delta = y - self.y_best
                
                if delta < 0.:
                    self.parameters, self.y_best = x_new, y
                    history["x"].append(self.parameters.copy())
                    history["y"].append(float(y))
                    history["info"].append(f"step size = {step_size}")
                    accepted_minus = True
                    
            if not accepted_plus and not accepted_minus:
                step_size *= self.beta
            ###################################
            ###        END MAIN LOOP        ###
            ###################################


            ###################################
            ###     CONVERGENCE CHECK       ###
            ###################################
            try:
                recent_values = history["y"][-self.window_epochs*self.n_parameters] 
                convergence_value = float(np.std(recent_values))
            except:
                convergence_value = np.nan
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
            
        return self.y_best, self.parameters, history



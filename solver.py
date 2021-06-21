import logging
import os
import time
import numpy as np
import multiprocessing

try:
    from generateSamples import load_mlp_generateSamples
except ModuleNotFoundError:
    pass


class MLPSolver(object):
    """The MLP approximation algorithm for semilinear parabolic PDEs."""
    
    def __init__(self, config, mlp):
        """
        Constructor for the MLPSolver class
        
        Parameters
        ---------
        config: dict
            Dict from the configuration file of the PDE under consideration
        mlp: Equation class object
            Class equation object (definition of the terminal function, nonlinearity f,
            and associated SDE approximation) of the PDE of interest    
            
        """
        self.eqn_config = config.eqn_config
        self.eval_config = config.eval_config
        self.mlp = mlp
        self.multiprocess = True if(self.eval_config.multiprocess) else False
        
        
    def train(self, samples=None):
        """
        Computes realizations of the MLP approximation algorithm.
        
        Parameters
        ---------
        samples: dict
            Dict of the pre-generated samples in advanced mode from generateSamples.py
        
        """
        start_train_time = time.time()
        training_history = []
                
        # begin mlp iterations (different realizations)
        for theta in range(1,self.eval_config.num_realization + 1):
            start_realization_time = time.time()
            if(self.eqn_config.gradient_dependent):
                if samples == None:
                    sol = (self.mlp_call_grad(self.mlp.f,
                                                self.mlp.g,
                                                self.mlp.dXsample,
                                                self.mlp.dIsample,
                                                self.mlp.num_iteration,
                                                self.mlp.num_iteration,
                                                self.mlp.x_init,
                                                self.mlp.start_time,
                                                self.multiprocess,
                                                [samples, (self.mlp.num_iteration,), 0])[:,0])
                else:
                    sol = (self.mlp_call_grad(self.mlp.f,
                                                self.mlp.g,
                                                self.mlp.dXsample,
                                                self.mlp.dIsample,
                                                self.mlp.num_iteration,
                                                self.mlp.num_iteration,
                                                self.mlp.x_init,
                                                self.mlp.start_time,
                                                self.multiprocess,
                                                [samples[theta], (self.mlp.num_iteration,), 0])[:,0])
            else:
                if samples == None:
                    sol = (self.mlp_call(self.mlp.f,
                                            self.mlp.g,
                                            self.mlp.dXsample, 
                                            self.mlp.num_iteration,
                                            self.mlp.num_iteration,
                                            self.mlp.x_init, 
                                            self.mlp.start_time,
                                            self.multiprocess,
                                            [samples, (self.mlp.num_iteration,), 0]))
                else:
                    sol = (self.mlp_call(self.mlp.f,
                                            self.mlp.g, 
                                            self.mlp.dXsample,
                                            self.mlp.num_iteration,
                                            self.mlp.num_iteration,
                                            self.mlp.x_init,
                                            self.mlp.start_time, 
                                            self.multiprocess, 
                                            [samples[theta], (self.mlp.num_iteration,), 0]))
                    
            elapsed_time = time.time() - start_realization_time
            cost = self.costEval(samples, theta)
            training_history.append([theta, sol, elapsed_time, cost])
            if self.eval_config.verbose:
                logging.info("Realization: %d, Solution: %s, Elapsed Time: %f, Cost#RV: %s" % (theta, sol, elapsed_time, cost))
                
        overall_elapsed_time = time.time() - start_train_time
        logging.info("Overall Elapsed Time %f" % overall_elapsed_time)
        
        return np.array(self.MLPlogging(training_history, samples))
    
    
    def errors(self, realization_sol):
        """
        Computes the L1 error, relative L1 error, L2 error, and relative L2 error for an given set of realizations
        
        Parameters
        ---------
        realization_sol: np.ndarray
            Contains the solutions of different realizations
            
        Returns
        -------
            L1 error, relative L1 error, L2 error, and relative L2 error for realization_sol
            with respect to the reference solution in self.eval_config
            
        """
        diff = (np.stack(realization_sol, axis=0)
                        - np.array([self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)],]*self.eval_config.num_realization))
        norm_ref_sol = np.linalg.norm(self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)])
        errors = np.zeros(5)
        errors[0] = np.sum(np.linalg.norm(diff, axis=1)) / self.eval_config.num_realization # L1 error
        errors[1] = np.sum(np.linalg.norm(diff, axis=1)) / (self.eval_config.num_realization * norm_ref_sol) # rel L1 error
        errors[2] = np.linalg.norm(diff) / np.sqrt(self.eval_config.num_realization) # L2 error
        errors[3] = np.linalg.norm(diff) / (np.sqrt(self.eval_config.num_realization) * norm_ref_sol) # rel L2 error
        if(self.eval_config.num_realization > 1): errors[4] = np.linalg.norm(diff) / np.sqrt(self.eval_config.num_realization - 1) # empirical SD
        
        return errors
    
    
    def costEval(self, samples, theta):
        """
        Evaluates the costs, that are the number of scalar random variables which have to be drawn in
        one realization of the MLP algorithm.
        
        Parameters
        ---------
        samples: dict
            Dict of the pregenerated samples in advanced mode from generateSamples.py
        theta: int
            index of the realization of the current MLP run.
        """
        
        if (samples == None): 
            return "Not evaluated"
        else:
            cost = 0
            for keys in samples[theta].keys():
                if(self.mlp.sampleNeeded[keys[4]]):
                    if(keys[4] == 2):
                        for k in range(len(samples[theta][keys])):
                            if hasattr(samples[theta][keys][k], 'shape'):
                                N = samples[theta][keys][k].shape[0]
                                d = samples[theta][keys][k].shape[1]
                                n = 5 # here implement logic
                                cost += n * N * d * 2 # additional cost for Ikpw
                    elif(keys[4] == 3):
                        for k in range(len(samples[theta][keys])):
                            if hasattr(samples[theta][keys][k], 'shape'):
                                N = samples[theta][keys][k].shape[0]
                                d = samples[theta][keys][k].shape[1]
                                n = 5 # here implement logic
                                M = d*(d-1)//2
                                cost += n * N * d * 2 + N * M #  +++ additional cost for Iwik
                    elif(keys[4] == 4):
                        for k in range(len(samples[theta][keys])):
                            if hasattr(samples[theta][keys][k], 'shape'):
                                N = samples[theta][keys][k].shape[0]
                                d = samples[theta][keys][k].shape[1]
                                n = 5 # here implement logic
                                M = d*(d-1)//2
                                cost += n * N * d * 2 + N * d + N * M #  +++ additional cost for Imr
                    else:
                        cost += len(np.vstack(samples[theta][keys]).flatten())
            return cost
                        
    
    def MLPlogging(self, training_history, samples):
        """
        Computes the solution average, the L1 error, the relative L1 error, the L2 error,
        relative L2 error, the maximal number of evaluations of 1D random variables per realization,
        and the average elapsed time per realization for a given training_history
        
        Parameters
        ---------
        training_history: np.ndarray
            Contains for each realization an array [iteration number, realization values, elapsed time]
            
        """
        errors = self.errors(np.array(training_history)[:,1])
        (training_history.append([self.mlp.dim,
                                    self.mlp.num_iteration,
                                    np.mean(np.array(training_history)[:,1], axis=0),
                                    self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)],
                                    errors[0],
                                    errors[1],
                                    errors[2],
                                    errors[3],
                                    errors[4],
                                    np.mean(np.hstack(np.array(training_history)[:,2]))]))
        
        logging.info("Reference Solution: %s," % training_history[-1][3])       
        logging.info("Solution Average: %s," % training_history[-1][2])
        logging.info("L1 Error: %f" % training_history[-1][4])
        logging.info("Relative L1 Error: %f" % training_history[-1][5])
        logging.info("L2 Error: %f" % training_history[-1][6])
        logging.info("Relative L2 Error: %f" % training_history[-1][7])
        logging.info("Empirical Standard Deviation: %f" % training_history[-1][8])
        logging.info("Average Elapsed Time per Realization: %f" % training_history[-1][9])
        if(samples != None): logging.info("Average Sampled 1D RV per Realization: %d" % np.mean(np.hstack(np.array(training_history[-(len(samples)+1):-1])[:,3])))
        
        
        return training_history
        
        
    def mlp_call_multiprocess(self, f, g, dXsample, n: int, M: int, l: int, x: np.ndarray, t_approx: float, result: np.ndarray, active: bool, advanced: list):
        """
        Auxiliary function for mlp_call routine when multiprocess == True. This function computes the Monte Carlo sum
        involving the difference of the nonlinearity f.
        
        Parameters
        ---------
        f : callable(t,x,v)
            Computes the gradient-dependent nonlinearity of a semilinear
            partial differential equation
        g: callable(x)
            Computes the terminal condition of the pde at terminal time T
        dXsample: callable(t_start, t_end, x, dW)
            Solves the SDE for a given approximation method
        dIsample: callable(t_start, t_end, x, dW, dX)
            Solves the derivative SDE for a given approximation method
        n: int
            Iteration step
        M: int
            Basis of number multilevel samples
        l: int
            Iteration step of the MLP approxmation in the Monte Carlo 
            sum involving the difference of the nonlinearity of a
            MLP approximation at iteration step n
        x: array
            Value of the state space which is approximated by the
            MLP approximation
        t_approx: float
            Time of the time-space point which is approximated by
            the MLP approximation
        result: np.ndarray
            Either None or an array. If result is an array, then mlp_call_multiprocess
            is called by a seperate process/core.
        active: boolean
            Checks if this routine is called within a new physical core
        advanced: list
            If advanced[0] == None, then the routine is called in the non-advanced mode, i.e. no pre-generated samples
            are used. Otherwise, advanced[0] is the dict of pre-generated samples, advanced[1] carries the history_index
            (see hist_index and pos_index in load_mlp_generateSamples for more information), and advanced[2] carries the current
            recursive depth "level" (see level  load_mlp_generateSamples for more information).
        
        Returns
        -------
        y: float
            Monte Carlo sum value involving the difference of the nonlinearity f of the MLP approximation
            at time t_approx and state x
        """
        
        rhs_f = np.zeros(self.eqn_config.dim_system)
        rhs_f_diff = np.zeros(self.eqn_config.dim_system)
        
        # Case l=0
        if(l==0):
            for i in range(M**n):
                # calculate right-hand side f
                if(advanced[0] == None):
                    # generate sample data and sample state
                    r = np.random.uniform(0,1,1)
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dX = dXsample(t_approx, R, x)[0]
                else:
                    r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dW = (load_mlp_generateSamples(samples=advanced[0],
                                                    pos_index=(advanced[2],2,l,i), 
                                                    hist_index=advanced[1],
                                                    config=self.mlp.sampleNeeded))
                    dX = dXsample(t_approx, R, x, dW)[0]
                rhs_f_diff = rhs_f_diff + f(R, dX[-1], self.mlp_call(f, g, dXsample, 0, M, dX[-1], R, False, advanced))
            rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**n))
        # Case l > 0
        else:       
            for i in range(M**(n-l)):
                # calculate right-hand side f
                if(advanced[0] == None):
                    # generate sample data and sample state
                    r = np.random.uniform(0,1,1)
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dX = dXsample(t_approx, R, x)[0]
                    rhs_f_diff = (rhs_f_diff
                                    + f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False, advanced))
                                    - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False, advanced)))
                else:
                    r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dW = (load_mlp_generateSamples(samples=advanced[0], 
                                                    pos_index=(advanced[2],2,l,i),
                                                    hist_index=advanced[1], 
                                                    config=self.mlp.sampleNeeded))
                    dX = dXsample(t_approx, R, x, dW)[0]
                    rhs_f_diff = (rhs_f_diff
                                    + f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,l,), advanced[2]+1]))
                                    - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,-(l-1),), advanced[2]+1])))
                    
            rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**(n-l)))
        
        if(active):
            for r in range(self.eqn_config.dim_system):
                result[r] = rhs_f[r].flatten()
        else:
            return rhs_f.flatten()

    
    def mlp_call(self, f, g, dXsample, n: int, M: int, x: np.ndarray, t_approx: float, multiprocess: bool, advanced: list):
        """
        Approximates the solution of a parabolic semilinear partial differential equation with gradient-independent
        nonlinearity (or stochastic fixed point equation) at the time-space point (t_approx, x) with the full-history
        recursive multilevel picard iteration approximation algorithm.
        
        Parameters
        ---------
        f : callable(t,x,v)
            Computes the gradient-dependent nonlinearity of a semilinear
            partial differential equation
        g: callable(x)
            Computes the terminal condition of the pde at terminal time T
        dXsample: callable(t_start, t_end, x, dW)
            Solves the SDE for a given approximation method
        n: int
            Iteration step
        M: int
            Basis of number multilevel samples
        x: array
            Value of the state space which is approximated by the
            MLP approximation
        t_approx: float
            Time of the time-space point which is approximated by
            the MLP approximation
        multiprocess: boolean
            Enables or Disables parallel computing version of the MLP
            approximation algorithm
        advanced: list
            If advanced[0] == None, then the routine is called in the non-advanced mode, i.e. no pre-generated samples
            are used. Otherwise, advanced[0] is the dict of pre-generated samples, advanced[1] carries the history_index
            (see hist_index and pos_index in load_mlp_generateSamples for more information), and advanced[2] carries the current
            recursive depth "level" (see level  load_mlp_generateSamples for more information).
        
        Returns
        -------
        y: float
            MLP approximation value at time t_approx and state x
        """
        
        
        if(n==0):
            return np.zeros(self.eqn_config.dim_system)
        else:
            # Here we have the multiprocess version of the MLP algorithm.
            if(multiprocess):
                
                threads = [None] * self.mlp.num_iteration
                results = [None] * self.mlp.num_iteration
                
                for l in range(n - 1, n - min(int(multiprocessing.cpu_count()/2), n+1), - 1):
                    results[l] = multiprocessing.Array("d",self.eqn_config.dim_system)
                    threads[l] = multiprocessing.Process(target=self.mlp_call_multiprocess, args=(f, g, dXsample, n, M, l, x, t_approx, results[l], True, advanced))
                    threads[l].start()
                
                for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):
                    results[l] = self.mlp_call_multiprocess(f, g, dXsample, n, M, l, x, t_approx, None, False, advanced)
                    
                rhs_g = np.zeros(self.eqn_config.dim_system)
                rhs_f = np.zeros(self.eqn_config.dim_system)
                
                # Compute in the meantime the Monte Carlo sum involving the terminal condition g
                for i in range(M**n):
                    # sample state
                    if(advanced[0] == None):
                        dX = dXsample(t_approx, self.mlp.total_time, x)[0]
                    else:
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],1,0,i),
                                                        hist_index=advanced[1], 
                                                        config=self.mlp.sampleNeeded))
                        dX = dXsample(t_approx, self.mlp.total_time, x, dW)[0]
                    rhs_g = rhs_g + g(dX[-1])
                rhs_g = rhs_g/(M**n)
                
                # Gather the results from the threads
                for l in range(n - 1, n - min(int(multiprocessing.cpu_count()/2), n+1), - 1):
                    threads[l].join()
                    rhs_f = rhs_f + results[l]
                    
                for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):               
                    rhs_f = rhs_f + results[l]
                    
            # Here starts the single core version of the MLP approximation algorithm
            else:
                rhs_g = np.zeros(self.eqn_config.dim_system)
                rhs_f = np.zeros(self.eqn_config.dim_system)
                rhs_f_diff = np.zeros(self.eqn_config.dim_system)
                
                # Compute the Monte Carlo sum involving the terminal condition g
                for i in range(M**n):
                    # sample state
                    if(advanced[0] == None):
                        dX = dXsample(t_approx, self.mlp.total_time, x)[0]
                    else:
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],1,0,i),
                                                        hist_index=advanced[1],
                                                        config=self.mlp.sampleNeeded))
                        dX = dXsample(t_approx, self.mlp.total_time, x, dW)[0]
                    rhs_g = rhs_g + g(dX[-1])           
                rhs_g = rhs_g/(M**n)
                
                # Compute the Monte Carlo sum involving the difference of the nonlinearity f
                ## Case l=0
                for i in range(M**n):                   
                    # sample state
                    if(advanced[0] == None):
                        r = np.random.uniform(0,1,1)
                        R = t_approx + (self.mlp.total_time - t_approx)*r
                        dX = dXsample(t_approx, R, x)[0]
                    else:
                        r = advanced[0][(advanced[2],2,0,i,0)+advanced[1]]
                        R = t_approx + (self.mlp.total_time - t_approx)*r
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],2,0,i),
                                                        hist_index=advanced[1],
                                                        config=self.mlp.sampleNeeded))
                        dX = dXsample(t_approx, R, x, dW)[0]    
                    rhs_f_diff = rhs_f_diff + f(R, dX[-1], self.mlp_call(f, g, dXsample, 0, M, dX[-1], R, False, advanced))
                rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**n))
                
                ## Case l > 0
                for l in range(1, n):
                    rhs_f_diff = np.zeros(self.eqn_config.dim_system)
                    for i in range(M**(n-l)):
                        # sample state
                        if(advanced[0] == None):
                            r = np.random.uniform(0,1,1)
                            R = t_approx + (self.mlp.total_time - t_approx)*r
                            dX = dXsample(t_approx, R, x)[0]
                            rhs_f_diff = (rhs_f_diff
                                            + f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False, advanced))
                                            - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False, advanced)))
                        else:
                            r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                            R = t_approx + (self.mlp.total_time - t_approx)*r
                            dW = (load_mlp_generateSamples(samples=advanced[0],
                                                            pos_index=(advanced[2],2,l,i),
                                                            hist_index=advanced[1],
                                                            config=self.mlp.sampleNeeded))
                            dX = dXsample(t_approx, R, x, dW)[0]
                            rhs_f_diff = (rhs_f_diff
                                            + f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,l,), advanced[2]+1]))
                                            - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,-(l-1),), advanced[2]+1])))           
                    rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**(n-l)))
                    
            return rhs_g.flatten() + rhs_f.flatten()
    
    
    def mlp_call_multiprocess_grad(self, f, g, dXsample, dIsample, n: int, M: int, l: int, x: np.ndarray, t_approx: float, result: np.ndarray, active: bool, advanced: list):
        """
        Auxiliary function for mlp_call_grad routine when multiprocess == True. This function computes the Monte Carlo sum
        involving the difference of the nonlinearity f.
        
        Parameters
        ---------
        f : callable(t,x,v)
            Computes the gradient-dependent nonlinearity of a semilinear
            partial differential equation
        g: callable(x)
            Computes the terminal condition of the pde at terminal time T
        dXsample: callable(t_start, t_end, x, dW)
            Solves the SDE for a given approximation method
        dIsample: callable(t_start, t_end, x, dW, dX)
            Solves the derivative SDE for a given approximation method
        n: int
            Iteration step
        M: int
            Basis of number multilevel samples
        l: int
            Iteration step of the MLP approxmation in the Monte Carlo 
            sum involving the difference of the nonlinearity of a
            MLP approximation at iteration step n
        x: array
            Value of the state space which is approximated by the
            MLP approximation
        t_approx: float
            Time of the time-space point which is approximated by
            the MLP approximation
        result: np.ndarray
            Either None or an array. If result is an array, then mlp_call_multiprocess
            is called by a seperate process/core.
        active: boolean
            Checks if this routine is called within a new physical core
        advanced: list
            If advanced[0] == None, then the routine is called in the non-advanced mode, i.e. no pre-generated samples
            are used. Otherwise, advanced[0] is the dict of pre-generated samples, advanced[1] carries the history_index
            (see hist_index and pos_index in load_mlp_generateSamples for more information), and advanced[2] carries the current
            recursive depth "level" (see level  load_mlp_generateSamples for more information).
        
        Returns
        -------
        y: float
            Monte Carlo sum value involving the difference of the nonlinearity f of the MLP approximation
            at time t_approx and state x
        """
        
        
        rhs_f = np.zeros(shape=(self.eqn_config.dim_system,1))
        rhs_f_diff = np.zeros(shape=(self.eqn_config.dim_system,1))
        
        # Case l=0
        if(l==0):
            for i in range(M**n):
                if(advanced[0] == None):
                    # generate sample data and sample state
                    r = np.random.power(self.eqn_config.time_dist_exponent,1)
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    (dX, dW) = dXsample(t_approx, R, x)
                    dI = dIsample(t_approx, R, x, dW, dX)
                else:
                    r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dW = (load_mlp_generateSamples(samples=advanced[0],
                                                    pos_index=(advanced[2],2,l,i),
                                                    hist_index=advanced[1],
                                                    config=self.mlp.sampleNeeded))
                    dX = dXsample(t_approx, R, x, dW)[0]
                    dI = dIsample(t_approx, R, x, dW, dX)
                rhs_f_diff = (rhs_f_diff 
                                + (r**(1-self.eqn_config.time_dist_exponent)) * f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, 0, M, dX[-1], R, False, advanced)) * dI)          
            rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**n)))
        # Case l > 0
        else:       
            for i in range(M**(n-l)):
                if(advanced[0] == None):
                    # generate sample data and sample state
                    r = np.random.power(self.eqn_config.time_dist_exponent,1)
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    (dX, dW) = dXsample(t_approx, R, x)
                    dI = dIsample(t_approx, R, x, dW, dX)
                    rhs_f_diff = (rhs_f_diff 
                                    + ((r**(1-self.eqn_config.time_dist_exponent)) * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False, advanced)) 
                                    - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False, advanced)))) * dI)
                else:
                    r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                    R = t_approx + (self.mlp.total_time - t_approx)*r
                    dW = (load_mlp_generateSamples(samples=advanced[0],
                                                    pos_index=(advanced[2],2,l,i), 
                                                    hist_index=advanced[1],
                                                    config=self.mlp.sampleNeeded))
                    dX = dXsample(t_approx, R, x, dW)[0]
                    dI = dIsample(t_approx, R, x, dW, dX)
                    rhs_f_diff = (rhs_f_diff 
                                    + ((r**(1-self.eqn_config.time_dist_exponent)) 
                                    * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,l,), advanced[2]+1])) 
                                    - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,-(l-1),), advanced[2]+1])))) * dI)
            rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**(n-l))))
        
        if(active):
            for r in range(self.eqn_config.dim_system):
                for d in range(self.mlp.dim + 1):
                    result[r*(self.mlp.dim + 1) + d] = rhs_f[r][d]
        else:
            return rhs_f.reshape((self.mlp.dim_system, self.mlp.dim + 1))


    def mlp_call_grad(self, f, g, dXsample, dIsample, n: int, M: int, x: np.ndarray, t_approx: float, multiprocess: bool, advanced: list):
        """
        Approximates the solution of a parabolic semilinear partial differential equation with gradient-dependent
        nonlinearity (or stochastic fixed point equation) at the time-space point (t_approx, x) with the full-history
        recursive multilevel picard iteration approximation algorithm.
        
        Parameters
        ---------
        f : callable(t,x,v)
            Computes the gradient-dependent nonlinearity of a semilinear
            partial differential equation
        g: callable(x)
            Computes the terminal condition of the pde at terminal time T
        dXsample: callable(t_start, t_end, x, dW)
            Solves the SDE for a given approximation method
        dIsample: callable(t_start, t_end, x, dW, dX)
            Solves the derivative SDE for a given approximation method
        n: int
            Iteration step
        M: int
            Basis of number multilevel samples
        x: array
            Value of the state space which is approximated by the
            MLP approximation
        t_approx: float
            Time of the time-space point which is approximated by
            the MLP approximation
        multiprocess: boolean
            Enables or Disables parallel computing version of the MLP
            approximation algorithm
        advanced: list
            If advanced[0] == None, then the routine is called in the non-advanced mode, i.e. no pre-generated samples
            are used. Otherwise, advanced[0] is the dict of pre-generated samples, advanced[1] carries the history_index
            (see hist_index and pos_index in load_mlp_generateSamples for more information), and advanced[2] carries the current
            recursive depth "level" (see level  load_mlp_generateSamples for more information).
        
        Returns
        -------
        y: float
            MLP approximation value at time t_approx and state x
        """
        
        if(n==0):
            return np.zeros(shape=(self.eqn_config.dim_system, self.mlp.dim + 1))
        else:
            # Here we have the multiprocess version of the MLP algorithm.
            if(multiprocess):
                
                threads = [None] * self.mlp.num_iteration
                results = [None] * self.mlp.num_iteration
                
                # Compute the Monte Carlo summands involving the difference of the nonlinearity f by
                # using for each summand a single process. Bound the number of processes
                # by the amount of physical cores of your system.
                for l in range(n - 1, n - min(int(multiprocessing.cpu_count()/2), n+1), - 1):
                    results[l] = multiprocessing.Array("d",self.eqn_config.dim_system * (self.mlp.dim + 1))
                    threads[l] = multiprocessing.Process(target=self.mlp_call_multiprocess_grad, args=(f, g, dXsample, dIsample, n, M, l, x, t_approx, results[l], True, advanced))
                    threads[l].start()
                    
                for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):
                    results[l] = self.mlp_call_multiprocess_grad(f, g, dXsample, dIsample, n, M, l, x, t_approx, None, False, advanced)
                    
                rhs_g = np.zeros(shape=(self.eqn_config.dim_system,1))
                rhs_f = np.zeros(shape=(self.eqn_config.dim_system,1))
                rhs_f_threads = np.zeros(self.eqn_config.dim_system * (self.mlp.dim + 1))
                
                # Compute in the meantime the Monte Carlo sum involving the terminal condition g
                for i in range(M**n):
                    if(advanced[0] == None):
                        (dX, dW) = dXsample(t_approx, self.mlp.total_time, x) 
                        dI = dIsample(t_approx, self.mlp.total_time, x, dW, dX)
                    else:
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],1,0,i),
                                                        hist_index=advanced[1], 
                                                        config=self.mlp.sampleNeeded))
                        dX = dXsample(t_approx, self.mlp.total_time, x, dW)[0]
                        dI = dIsample(t_approx, self.mlp.total_time, x, dW, dX)
                    rhs_g = rhs_g + (g(dX[-1]) - g(x)) * dI
                    
                rhs_g = np.insert(np.zeros((self.eqn_config.dim_system,self.mlp.dim)),0,g(x).flatten(),axis=1) + (rhs_g/(M**n))
                
                # Gather the results from the threads
                for l in range(n - 1, n - min(int(multiprocessing.cpu_count()/2), n+1), - 1):
                    threads[l].join()
                    rhs_f_threads = rhs_f_threads + results[l]
                    
                rhs_f = rhs_f + rhs_f_threads.reshape(self.eqn_config.dim_system, self.mlp.dim + 1)
                    
                for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):
                    rhs_f = rhs_f + results[l]
            
            # Here starts the single core version of the MLP approximation algorithm
            else:
                rhs_g = np.zeros(shape=(self.eqn_config.dim_system,1))
                rhs_f = np.zeros(shape=(self.eqn_config.dim_system,1))
                rhs_f_diff = np.zeros(shape=(self.eqn_config.dim_system,1))
                
                
                # Compute the Monte Carlo sum involving the terminal condition g
                for i in range(M**n):
                    # sample state
                    if(advanced[0] == None):
                        (dX, dW) = dXsample(t_approx, self.mlp.total_time, x)
                        dI = dIsample(t_approx, self.mlp.total_time, x, dW, dX)
                    else:
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],1,0,i),
                                                        hist_index=advanced[1], 
                                                        config=self.mlp.sampleNeeded))
                        
                        dX = dXsample(t_approx, self.mlp.total_time, x, dW)[0]
                        dI = dIsample(t_approx, self.mlp.total_time, x, dW, dX)
                    
                    rhs_g = rhs_g + (g(dX[-1]) - g(x)) * dI
                rhs_g = np.insert(np.zeros((self.eqn_config.dim_system,self.mlp.dim)),0,g(x).flatten(),axis=1) + (rhs_g/(M**n))         
                
                # Compute the Monte Carlo sum involving the difference of the nonlinearity f
                ## Case l=0
                for i in range(M**n):
                    # sample state
                    if(advanced[0] == None):
                        r = np.random.power(self.eqn_config.time_dist_exponent,1)
                        R = t_approx + (self.mlp.total_time - t_approx)*r
                        (dX, dW) = dXsample(t_approx, R, x)
                        dI = dIsample(t_approx, R, x, dW, dX)
                    else:
                        r = advanced[0][(advanced[2],2,0,i,0)+advanced[1]]
                        R = t_approx + (self.mlp.total_time - t_approx)*r
                        dW = (load_mlp_generateSamples(samples=advanced[0],
                                                        pos_index=(advanced[2],2,0,i),
                                                        hist_index=advanced[1], 
                                                        config=self.mlp.sampleNeeded))
                        dX = dXsample(t_approx, R, x, dW)[0]
                        dI = dIsample(t_approx, R, x, dW, dX)
                    rhs_f_diff = rhs_f_diff + (r**(1-self.eqn_config.time_dist_exponent)) * f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, 0, M, dX[-1], R, False, advanced)) * dI
                                
                rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**n)))
                
                ## Case l > 0
                for l in range(1, n):
                    rhs_f_diff = np.zeros(shape=(self.eqn_config.dim_system,1))
                    for i in range(M**(n-l)):
                    
                        if(advanced[0] == None):
                            r = np.random.power(self.eqn_config.time_dist_exponent,1)
                            R = t_approx + (self.mlp.total_time - t_approx)*r
                            (dX, dW) = dXsample(t_approx, R, x)
                            dI = dIsample(t_approx, R, x, dW, dX)
                            rhs_f_diff = (rhs_f_diff
                                            + ((r**(1-self.eqn_config.time_dist_exponent))
                                            * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False, advanced))
                                            - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False, advanced)))) * dI)
                        else:
                            r = advanced[0][(advanced[2],2,l,i,0)+advanced[1]]
                            R = t_approx + (self.mlp.total_time - t_approx)*r
                            dW = (load_mlp_generateSamples(samples=advanced[0],
                                                            pos_index=(advanced[2],2,l,i),
                                                            hist_index=advanced[1], 
                                                            config=self.mlp.sampleNeeded))
                            dX = dXsample(t_approx, R, x, dW)[0]
                            dI = dIsample(t_approx, R, x, dW, dX)
                            rhs_f_diff = (rhs_f_diff
                                            + ((r**(1-self.eqn_config.time_dist_exponent))
                                            * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,l,), advanced[2]+1]))
                                            - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False, [advanced[0], advanced[1]+(i,-(l-1),), advanced[2]+1])))) * dI)
                    
                    rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**(n-l))))
                    
            return rhs_g.reshape((self.mlp.dim_system, self.mlp.dim + 1)) + rhs_f.reshape((self.mlp.dim_system, self.mlp.dim + 1))
                


import logging
import os
import time
import numpy as np
import multiprocessing


class MLPSolver(object):
	"""The MLP approximation algorithm for semilinear parabolic PDEs."""
	def __init__(self, config, mlp):
		"""
		Constrcutor for the MLPSolver class
		
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
		
		
	def train(self):
		"""
		Computes realizations of the MLP approximation algorithm.
		"""
		start_train_time = time.time()
		training_history = []
				
		# begin mlp iterations (different realizations)
		for theta in range(1,self.eval_config.num_realization + 1):
			start_realization_time = time.time()
			if(self.eqn_config.gradient_dependent):
				sol = self.mlp_call_grad(self.mlp.f, self.mlp.g, self.mlp.dXsample, self.mlp.dIsample, self.mlp.num_iteration, self.mlp.num_iteration, self.mlp.x_init, self.mlp.start_time, self.multiprocess)[:,0]
			else:
				sol = self.mlp_call(self.mlp.f, self.mlp.g, self.mlp.dXsample, self.mlp.num_iteration, self.mlp.num_iteration, self.mlp.x_init, self.mlp.start_time, self.multiprocess)
			elapsed_time = time.time() - start_realization_time
			training_history.append([theta, sol, elapsed_time])
			if self.eval_config.verbose:
				logging.info("Realization: %d, Solution: %s, Elapsed Time: %f" % (theta, sol, elapsed_time))
				
		overall_elapsed_time = time.time() - start_train_time
		logging.info("Overall Elapsed Time %f" % overall_elapsed_time)
		
		return np.array(self.MLPlogging(training_history))
	
	
	def relL2error(self, realization_sol):
		"""
		Computes the relative L2 error for an given set of realizations
		
		Parameters
		---------
		realization_sol: np.ndarray
			Contains the solutions of different realizations
			
		Returns
		-------
			Relative L2 error for realization_sol with respect to the reference solution in
			self.eval_config			
		"""
		return np.linalg.norm(np.stack(realization_sol, axis=0) -np.array([self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)],]*self.eval_config.num_realization)) / (np.sqrt(self.eval_config.num_realization) * np.linalg.norm(self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)]))
		
		
	def costEval(self, M: int, n: int, alpha: int, d: int):
		"""
		Computes the number of one-dimensional random variables which have to be drawn 
		in order to compute one realization of the MLP approximation algorithm V_{M,n}
		where the cost of sampling the associated SDE is bounded by alpha*d, d dimension.
		
		Parameters
		---------
		M: int
			Basis of number multilevel samples
		n: int
			Iteration step
		alpha: int
			Dimension independent cost factor for sampling the associated SDE
		d: int
			Dimension of the state space
		
		Returns
		-------
		Sampling cost for one realization of the MLP approximation algorithm V_{M,n}
		"""
		
		if(n==0):
			return 0
		else:
			costsum = alpha * (M**n) * d
			for l in range(n):
				costsum += (M**(n-l))*(alpha*d + 1 + self.costEval(M,l,alpha,d) + (l > 0) * self.costEval(M,l-1,alpha,d))
			
			return costsum

	
	def MLPlogging(self, training_history):
		"""
		Computes the solution average, the relative L2 error, the maximal number of evaluations of 1D
		random variables per realization, and the average elapsed time per realization for a
		given training_history
		
		Parameters
		---------
		training_history: np.ndarray
			Contains for each realization an array [iteration number, solution value, elapsed time]
		"""
		training_history.append([self.mlp.dim, self.mlp.num_iteration, np.mean(np.array(training_history)[:,1], axis=0), self.eval_config.reference_sol[self.eqn_config.dim.index(self.mlp.dim)], self.relL2error(np.array(training_history)[:,1]), self.costEval(self.mlp.num_iteration, self.mlp.num_iteration, self.eval_config.sample_cost_1D, self.mlp.dim), np.mean(np.hstack(np.array(training_history)[:,2]))])
		logging.info("Solution Average: %s," % training_history[-1][2])
		logging.info("Relative L2 Error: %f" % training_history[-1][4])
		logging.info("Maximal Number of Evaluations of 1D Random Variables per Realization: %d" % training_history[-1][5])
		logging.info("Average Elapsed Time per Realization: %f" % training_history[-1][6])
		
		return training_history
		
		
	def mlp_call_multiprocess(self, f, g, dXsample, n: int, M: int, l: int, x: np.ndarray, t_approx: float, result: np.ndarray, active: bool):
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
				R = t_approx + (self.mlp.total_time - t_approx)*np.random.uniform(0,1,1)
				dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
				dX = dXsample(t_approx, R, x, dW)
				rhs_f_diff = rhs_f_diff + f(R, dX[-1], self.mlp_call(f, g, dXsample, 0, M, dX[-1], R, False))
				
			rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**n))
		# Case l > 0
		else:		
			for i in range(M**(n-l)):
				R = t_approx + (self.mlp.total_time - t_approx)*np.random.uniform(0,1,1)
				dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
				dX = dXsample(t_approx, R, x, dW)
				rhs_f_diff = rhs_f_diff + (f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False)) - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False)))
								  
			rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**(n-l)))
		
		if(active):
			for r in range(self.eqn_config.dim_system):
				result[r] = rhs_f[r].flatten()
		else:
			return rhs_f.flatten()

    
	def mlp_call(self, f, g, dXsample, n: int, M: int, x: np.ndarray, t_approx: float, multiprocess: bool):
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
				
				# Compute the Monte Carlo summands involving the difference of the nonlinearity f by
				# using for each summand a single process. Bound the number of processes
				# by the amount of physical cores of your system.
				for l in range(n - 1, n - min(int(multiprocessing.cpu_count()/2), n+1), - 1):
					results[l] = multiprocessing.Array("d",self.eqn_config.dim_system)
					threads[l] = multiprocessing.Process(target=self.mlp_call_multiprocess, args=(f, g, dXsample, n, M, l, x, t_approx, results[l], True))
					threads[l].start()
					
				for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):
					results[l] = self.mlp_call_multiprocess(f, g, dXsample, n, M, l, x, t_approx, None, False)
					
				rhs_g = np.zeros(self.eqn_config.dim_system)
				rhs_f = np.zeros(self.eqn_config.dim_system)
				
				# Compute in the meantime the Monte Carlo sum involving the terminal condition g
				for i in range(M**n):
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, self.mlp.total_time, x, dW)
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
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, self.mlp.total_time, x, dW)
					rhs_g = rhs_g + g(dX[-1])
			
				rhs_g = rhs_g/(M**n)
				
				# Compute the Monte Carlo sum involving the difference of the nonlinearity f
				## Case l=0
				for i in range(M**n):
					R = t_approx + (self.mlp.total_time - t_approx)*np.random.uniform(0,1,1)
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, R, x, dW)
					rhs_f_diff = rhs_f_diff + f(R, dX[-1], self.mlp_call(f, g, dXsample, 0, M, dX[-1], R, False))
				rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**n))
				
				## Case l > 0
				for l in range(1, n):
					rhs_f_diff = np.zeros(self.eqn_config.dim_system)
					for i in range(M**(n-l)):
						R = t_approx + (self.mlp.total_time - t_approx)*np.random.uniform(0,1,1)
						dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
						dX = dXsample(t_approx, R, x, dW)
						rhs_f_diff = rhs_f_diff + (f(R, dX[-1], self.mlp_call(f, g, dXsample, l, M, dX[-1], R, False)) - f(R, dX[-1], self.mlp_call(f, g, dXsample, l-1, M, dX[-1], R, False)))
						          
					rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (M**(n-l)))
					
			return rhs_g.flatten() + rhs_f.flatten()
	
	
	def mlp_call_multiprocess_grad(self, f, g, dXsample, dIsample, n: int, M: int, l: int, x: np.ndarray, t_approx: float, result: np.ndarray, active: bool):
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
				r = np.random.power(self.eqn_config.time_dist_exponent,1)
				R = t_approx + (self.mlp.total_time - t_approx)*r
				dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
				dX = dXsample(t_approx, R, x, dW)
				dI = dIsample(t_approx, R, x, dW, dX)
				rhs_f_diff = rhs_f_diff + (r**(1-self.eqn_config.time_dist_exponent)) * f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, 0, M, dX[-1], R, False)) * dI
				
			rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**n)))
		# Case l > 0
		else:		
			for i in range(M**(n-l)):
				r = np.random.power(self.eqn_config.time_dist_exponent,1)
				R = t_approx + (self.mlp.total_time - t_approx)*r
				dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
				dX = dXsample(t_approx, R, x, dW)
				dI = dIsample(t_approx, R, x, dW, dX)
				rhs_f_diff = rhs_f_diff + ((r**(1-self.eqn_config.time_dist_exponent)) * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False)) - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False)))) * dI
						
			rhs_f = (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**(n-l))))
		
		if(active):
			for r in range(self.eqn_config.dim_system):
				for d in range(self.mlp.dim + 1):
					result[r*(self.mlp.dim + 1) + d] = rhs_f[r][d]
		else:
			return rhs_f


	def mlp_call_grad(self, f, g, dXsample, dIsample, n: int, M: int, x: np.ndarray, t_approx: float, multiprocess: bool):
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
					threads[l] = multiprocessing.Process(target=self.mlp_call_multiprocess_grad, args=(f, g, dXsample, dIsample, n, M, l, x, t_approx, results[l], True))
					threads[l].start()
					
				for l in range(n - min(int(multiprocessing.cpu_count()/2), n+1), -1, -1):
					results[l] = self.mlp_call_multiprocess_grad(f, g, dXsample, dIsample, n, M, l, x, t_approx, None, False)
					
				rhs_g = np.zeros(shape=(self.eqn_config.dim_system,1))
				rhs_f = np.zeros(shape=(self.eqn_config.dim_system,1))
				rhs_f_threads = np.zeros(self.eqn_config.dim_system * (self.mlp.dim + 1))
				
				# Compute in the meantime the Monte Carlo sum involving the terminal condition g
				for i in range(M**n):
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, self.mlp.total_time, x, dW)
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
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, self.mlp.total_time, x, dW)
					dI = dIsample(t_approx, self.mlp.total_time, x, dW, dX)
					rhs_g = rhs_g + (g(dX[-1]) - g(x)) * dI
					
				rhs_g = np.insert(np.zeros((self.eqn_config.dim_system,self.mlp.dim)),0,g(x).flatten(),axis=1) + (rhs_g/(M**n))
				
				# Compute the Monte Carlo sum involving the difference of the nonlinearity f
				## Case l=0
				for i in range(M**n):
					r = np.random.power(self.eqn_config.time_dist_exponent,1)
					R = t_approx + (self.mlp.total_time - t_approx)*r
					dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
					dX = dXsample(t_approx, R, x, dW)
					dI = dIsample(t_approx, R, x, dW, dX)
					rhs_f_diff = rhs_f_diff + (r**(1-self.eqn_config.time_dist_exponent)) * f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, 0, M, dX[-1], R, False)) * dI
				
				rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**n)))
				
				## Case l > 0
				for l in range(1, n):
					rhs_f_diff = np.zeros(shape=(self.eqn_config.dim_system,1))
					for i in range(M**(n-l)):
						r = np.random.power(self.eqn_config.time_dist_exponent,1)
						R = t_approx + (self.mlp.total_time - t_approx)*r
						dW = np.random.normal(0, 1, size = (self.mlp.num_gridpoint,self.mlp.dim))
						dX = dXsample(t_approx, R, x, dW)
						dI = dIsample(t_approx, R, x, dW, dX)
						rhs_f_diff = rhs_f_diff + ((r**(1-self.eqn_config.time_dist_exponent)) * (f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l, M, dX[-1], R, False)) - f(R, dX[-1], self.mlp_call_grad(f, g, dXsample, dIsample, l-1, M, dX[-1], R, False)))) * dI
						
					rhs_f = rhs_f + (self.mlp.total_time - t_approx) * (rhs_f_diff / (self.eqn_config.time_dist_exponent * (M**(n-l))))
					
			return rhs_g + rhs_f
				


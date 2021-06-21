"""
This module generates samples for the advanced mode of the mlpSolver.
"""

import numpy as np
import math
import warnings
from sdeint.wiener import (Ikpw, Iwik)
from sdeint2.wiener_extension import (Imr, Itildekp, Ihatkp)

#turn off warnings on Windows 10
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

def load_mlp_generateSamples(samples, pos_index, hist_index, config):
	"""
	Loads the pre-generate samples in the mlp_call or mlp_call_grad routine
	by index.
	
	Parameters
	---------
	samples: dict
		Dict of the samples of the current mlp realization
	pos_index: tuple
		Index which determines at which position in the MLP algortihm the sample is drawn.
		It is of fixed length 6 of the form (level, M, n, sum, l,i)
		where level \in {0,1,...,M-1}, with e.g. level = 0 corresponding
		to all samples which are drawn in the top recursive level of the
		MLP algorithm; where M, n, l, i correspond to the sum index in
		the MLP algorithm; and sum \in {1,2}, where sum = 1 corresponds
		to the summand where the terminal condition g is involved and
		sum = 2 corresponds to the summand where the nonlinearity f is
		involved.
	hist_index: tuple
		Index which determines at which recursive path in the MLP algorithm
		the sample is drawn. It is of variable length and it is constructed
		as follows: Assume that we want to compute V_{5,5}, then in the first
		iteration our index is given by (5). In this iteration we call
		1x V_{5,4} for each i in {1,...,5}
		2x V_{5,3} for each i in {1,...,5**2}
		2x V_{5,2} for each i in {1,...,5**3}
		2x V_{5,1} for each i in {1,...,5**4}
		1x V_{5,0} for each i in {1,...,5**5}
		in the summand involving f. For example if we call V_{5,3}, then
		the new history index for this call is given by (5,i,3) if we call
		it in the left-hand term of the difference. If we would call V_{5,3}
		as the right-hand term of the difference then the index would be
		(5,i,-3). The other indicies are constructed in the same procedure.
	config: array
		An array of boolean values which determines which samples are needed
		for the respective sampling method. This array is defined as self.sampleNeeded
		in each sampling method in the module sampler.
		
	Returns
	------
	The samples in the samples dict to the corresponding index.
		
	"""
	dW = {}	
	if(config[1]): dW[(1,)] = samples[pos_index+(1,)+hist_index] # sample dW
	if(config[2]): dW[(2,)] = samples[pos_index+(2,)+hist_index] # sample Ikpw
	if(config[3]): dW[(3,)] = samples[pos_index+(3,)+hist_index] # sample Iwik
	if(config[4]): dW[(4,)] = samples[pos_index+(4,)+hist_index] # sample Imr
	if(config[5]): dW[(5,)] = samples[pos_index+(5,)+hist_index] # sample Itilde
	if(config[6]): dW[(6,)] = samples[pos_index+(6,)+hist_index] # sample Ihat
	if(config[7]): dW[(7,)] = samples[pos_index+(7,)+hist_index] # sample Xi
	
	return dW

	
def start_mlp_generateSamples(total_time, start_time, M, n, num_gridpoint, dim, active, time_dist_exponent):
	"""
	Start the mlp_generateSamples routine to pre-generate samples in the advanced/comparison mode of the mlp
	algorithm.
	
	Parameters
		---------
		total_time: float
			Total time of the interval of the PDE under consideration
		start_time: float
			Start time/Approximation time of the interval of the PDE under consideration
		M: int
			Basis of number multilevel samples
		n: int
			Iteration step
		num_gridpoint: int
			Number of gridpoints - 1 of the SDE approximation method in the MLP algorithm.
		dim: int
			Dimension of the state of the PDE under consideration
		active: float
			Array of booleans of the config file "advanced" of the PDE under consideration
			to determine which samples have to be generated.
			active[0]==True: Sample time points R
			active[1]==True: Sample Wiener increments dW
			active[2]==True: Sample iterated integrals by Ikpw
			active[3]==True: Sample iterated integrals by Iwik
			active[4]==True: Sample iterated integrals by Imr
			active[5]==True: Sample three point distribution Itildekp
			active[6]==True: Sample two point distribution Ihatkp
			active[7]==True: Sample Xi (used on certain algorithms in the module sdeint2.integrate_extension)
		time_dist_exponent: float
			Exponent of the np.random.power distribution for the time point samples R; e.g. ==1.0 is the uniform distribution.
			This parameter is specified in the config file of the PDE under consideration.
	
	Returns
	-------
	Returns a dict of samples of one realization of the MLP algorithm for the given parameters.
	
	"""
	samples = {}
	level = 0
	ind = (n,)
	samples = mlp_generateSamples(samples, start_time, total_time, start_time, M, n, ind, num_gridpoint, dim, level, active, time_dist_exponent)
	return samples
	

def find_grid(t_start, t_end, total_time, start_time, num_gridpoint):
	"""
	Calculates the grid for a given subinterval [t_start, t_end] of 
	the interval [start_time, total_time], where the main interval is
	discretized by (num_gridpoint + 1) gridpoints.
	
	Parameters
	---------
	t_start: float
		Start time of the subinterval
	t_end: float
		End time of the subinterval
	total_time: float
		End Time of the main interval
	start_time: float
		Start Time of the main interval
	num_gridpoint: int
		Number of gridpoints - 1 of the main interval.
		
	Returns
	------
	Grid of the subinterval where start and end points are
	t_start and t_end, respectively, and the interior points
	are the grid points of the main interval grid.
	
	"""
	# cast values
	t_start = float(t_start)
	t_end = float(t_end)

	h = (total_time - start_time)/num_gridpoint
	i_start = int(math.ceil((num_gridpoint/(total_time - start_time)) * t_start))
	if(t_start == i_start*h): i_start = i_start + 1
	i_end = int(math.floor((num_gridpoint/(total_time - start_time)) * t_end))
	if(t_end == i_end*h): i_end = i_end - 1
	if(i_end < i_start):
		grid = np.array([t_start, t_end])
	if(i_end == i_start):
		grid = np.array([t_start, i_start*h, t_end])
	if(i_start < i_end):
		grid = np.linspace((i_start-1)*h, (i_end+1)*h, i_end - i_start + 3)
		grid[0] = t_start
		grid[-1] = t_end
		
	return grid


def mlp_generateSamples(samples, t_approx, total_time, start_time, M, n, ind, num_gridpoint, dim, level, active, time_dist_exponent):
	"""
	Computes the samples of one realization of the MLP algorithm with respect to the given parametes.
	
	Parameters
	---------
	samples: dict
		Dict to save the drawn samples by index.
	t_approx: float
		Approximation time of the MLP iterate.
	total_time: float
		see start_mlp_generateSamples
	start_time: float
		see start_mlp_generateSamples
	M: int
		see start_mlp_generateSamples
	n: int
		see start_mlp_generateSamples
	ind: tuple
		see hist_index in load_mlp_generateSamples
	dim: int
		see start_mlp_generateSamples
	level: int
		see start_mlp_generateSamples
	active:
		see start_mlp_generateSamples
	time_dist_exponent: float
		see start_mlp_generateSamples
		
	Returns:
		Dict of the MLP samples of one realzation.
	"""
	if(n==0): return samples
	
	for i in range(M**abs(n)):
	
		grid = find_grid(t_approx, total_time, total_time, start_time, num_gridpoint)
		
		# we have at most three different step sizes
		if(len(grid) == 2):
			h1 = grid[1] - grid[0]
			if(active[1]): samples[(level,1,0,i,1)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample dW
			if(active[2]): samples[(level,1,0,i,2)+ind] = np.array([Ikpw(samples[(level,1,0,i,1)+ind][0], h1)[1]]) # sample Ikpw
			if(active[3]): samples[(level,1,0,i,3)+ind] = np.array([Iwik(samples[(level,1,0,i,1)+ind][0], h1)[1]]) # sample Iwik
			if(active[4]): samples[(level,1,0,i,4)+ind] = np.array([Imr(samples[(level,1,0,i,1)+ind][0], h1)[1]]) # sample Imr
			if(active[5]): samples[(level,1,0,i,5)+ind] = np.array([Itildekp(1, dim-1, h1)]) # sample Itilde
			if(active[6]): samples[(level,1,0,i,6)+ind] = np.array([Ihatkp(1, dim, h1)]) # sample Ihat
			if(active[7]): samples[(level,1,0,i,7)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample Xi
		else:
			h1 = grid[1] - grid[0]
			h2 = grid[2] - grid[1]
			h3 = grid[-1] - grid[-2]
			if(active[1]): samples[(level,1,0,i,1)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
														np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
														np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample dW
			if(len(grid) != 3):
				if(active[2]): samples[(level,1,0,i,2)+ind] = (np.array([Ikpw(samples[(level,1,0,i,1)+ind][0], h1)[1],
															Ikpw(samples[(level,1,0,i,1)+ind][1], h2)[1],
															Ikpw(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Ikpw
				if(active[3]): samples[(level,1,0,i,3)+ind] = (np.array([Iwik(samples[(level,1,0,i,1)+ind][0], h1)[1],
															Iwik(samples[(level,1,0,i,1)+ind][1], h2)[1],
															Iwik(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Iwik
				if(active[4]): samples[(level,1,0,i,4)+ind] = (np.array([Imr(samples[(level,1,0,i,1)+ind][0], h1)[1],
															Imr(samples[(level,1,0,i,1)+ind][1], h2)[1],
															Imr(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Imr
			else:
				if(active[2]): samples[(level,1,0,i,2)+ind] = (np.array([Ikpw(samples[(level,1,0,i,1)+ind][0], h1)[1],
															None,
															Ikpw(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Ikpw
				if(active[3]): samples[(level,1,0,i,3)+ind] = (np.array([Iwik(samples[(level,1,0,i,1)+ind][0], h1)[1],
															None,
															Iwik(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Iwik
				if(active[4]): samples[(level,1,0,i,4)+ind] = (np.array([Imr(samples[(level,1,0,i,1)+ind][0], h1)[1],
															None,
															Imr(samples[(level,1,0,i,1)+ind][2], h3)[1]])) # sample Imr
			if(active[5]): samples[(level,1,0,i,5)+ind] = (np.array([Itildekp(1, dim-1, h1),
														Itildekp(len(grid)-3, dim-1, h2),
														Itildekp(1, dim-1, h3)])) # sample Itilde
			if(active[6]): samples[(level,1,0,i,6)+ind] = (np.array([Ihatkp(1, dim, h1),
														Ihatkp(len(grid)-3, dim, h2),
														Ihatkp(1, dim, h3)])) # sample Ihat
			if(active[7]): samples[(level,1,0,i,7)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
														np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
														np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample Xi
			
			
			
	# Case l=0
	for i in range(M**abs(n)):
		if(active[0]): samples[(level,2,0,i,0)+ind] = np.random.power(time_dist_exponent,1) # sample R
		grid = find_grid(t_approx, t_approx + (total_time - t_approx)*samples[(level,2,0,i,0)+ind], total_time, start_time, num_gridpoint)
		
		# we have at most three different step sizes
		if(len(grid) == 2):
			h1 = grid[1] - grid[0]
			if(active[1]): samples[(level,2,0,i,1)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample dW
			if(active[2]): samples[(level,2,0,i,2)+ind] = np.array([Ikpw(samples[(level,2,0,i,1)+ind][0], h1)[1]]) # sample Ikpw
			if(active[3]): samples[(level,2,0,i,3)+ind] = np.array([Iwik(samples[(level,2,0,i,1)+ind][0], h1)[1]]) # sample Iwik
			if(active[4]): samples[(level,2,0,i,4)+ind] = np.array([Imr(samples[(level,2,0,i,1)+ind][0], h1)[1]]) # sample Imr
			if(active[5]): samples[(level,2,0,i,5)+ind] = np.array([Itildekp(1, dim-1, h1)]) # sample Itilde
			if(active[6]): samples[(level,2,0,i,6)+ind] = np.array([Ihatkp(1, dim, h1)]) # sample Ihat
			if(active[7]): samples[(level,2,0,i,7)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample Xi
		else:
			h1 = grid[1] - grid[0]
			h2 = grid[2] - grid[1]
			h3 = grid[-1] - grid[-2]
			if(active[1]): samples[(level,2,0,i,1)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
														np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
														np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample dW
			if(len(grid) != 3):
				if(active[2]): samples[(level,2,0,i,2)+ind] = (np.array([Ikpw(samples[(level,2,0,i,1)+ind][0], h1)[1],
															Ikpw(samples[(level,2,0,i,1)+ind][1], h2)[1],
															Ikpw(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Ikpw
				if(active[3]): samples[(level,2,0,i,3)+ind] = (np.array([Iwik(samples[(level,2,0,i,1)+ind][0], h1)[1],
															Iwik(samples[(level,2,0,i,1)+ind][1], h2)[1],
															Iwik(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Iwik
				if(active[4]): samples[(level,2,0,i,4)+ind] = (np.array([Imr(samples[(level,2,0,i,1)+ind][0], h1)[1],
															Imr(samples[(level,2,0,i,1)+ind][1], h2)[1],
															Imr(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Imr
			else:
				if(active[2]): samples[(level,2,0,i,2)+ind] = (np.array([Ikpw(samples[(level,2,0,i,1)+ind][0], h1)[1],
															None,
															Ikpw(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Ikpw
				if(active[3]): samples[(level,2,0,i,3)+ind] = (np.array([Iwik(samples[(level,2,0,i,1)+ind][0], h1)[1],
															None,
															Iwik(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Iwik
				if(active[4]): samples[(level,2,0,i,4)+ind] = (np.array([Imr(samples[(level,2,0,i,1)+ind][0], h1)[1],
															None,
															Imr(samples[(level,2,0,i,1)+ind][2], h3)[1]])) # sample Imr
			if(active[5]): samples[(level,2,0,i,5)+ind] = (np.array([Itildekp(1, dim-1, h1),
														Itildekp(len(grid)-3, dim-1, h2),
														Itildekp(1, dim-1, h3)])) # sample Itilde
			if(active[6]): samples[(level,2,0,i,6)+ind] = (np.array([Ihatkp(1, dim, h1),
														Ihatkp(len(grid)-3, dim, h2),
														Ihatkp(1, dim, h3)])) # sample Ihat
			if(active[7]): samples[(level,2,0,i,7)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
														np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
														np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample Xi

	# Case l > 0
	for l in range(1, abs(n)):
		for i in range(M**(abs(n)-l)):
			if(active[0]): samples[(level,2,l,i,0)+ind] = np.random.power(time_dist_exponent,1) # sample R
			grid = find_grid(t_approx, t_approx + (total_time - t_approx)*samples[(level,2,l,i,0)+ind], total_time, start_time, num_gridpoint)
			
			# we have at most three different step sizes
			if(len(grid) == 2):
				h1 = grid[1] - grid[0]
				if(active[1]): samples[(level,2,l,i,1)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample dW
				if(active[2]): samples[(level,2,l,i,2)+ind] = np.array([Ikpw(samples[(level,2,l,i,1)+ind][0], h1)[1]]) # sample Ikpw
				if(active[3]): samples[(level,2,l,i,3)+ind] = np.array([Iwik(samples[(level,2,l,i,1)+ind][0], h1)[1]]) # sample Iwik
				if(active[4]): samples[(level,2,l,i,4)+ind] = np.array([Imr(samples[(level,2,l,i,1)+ind][0], h1)[1]]) # sample Imr
				if(active[5]): samples[(level,2,l,i,5)+ind] = np.array([Itildekp(1, dim-1, h1)]) # sample Itilde
				if(active[6]): samples[(level,2,l,i,6)+ind] = np.array([Ihatkp(1, dim, h1)]) # sample Ihat
				if(active[7]): samples[(level,2,l,i,7)+ind] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim))]) # sample Xi
			else:
				h1 = grid[1] - grid[0]
				h2 = grid[2] - grid[1]
				h3 = grid[-1] - grid[-2]
				if(active[1]): samples[(level,2,l,i,1)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
															np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
															np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample dW
				if(len(grid) != 3):
					if(active[2]): samples[(level,2,l,i,2)+ind] = (np.array([Ikpw(samples[(level,2,l,i,1)+ind][0], h1)[1],
																Ikpw(samples[(level,2,l,i,1)+ind][1], h2)[1],
																Ikpw(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Ikpw
					if(active[3]): samples[(level,2,l,i,3)+ind] = (np.array([Iwik(samples[(level,2,l,i,1)+ind][0], h1)[1],
																Iwik(samples[(level,2,l,i,1)+ind][1], h2)[1],
																Iwik(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Iwik
					if(active[4]): samples[(level,2,l,i,4)+ind] = (np.array([Imr(samples[(level,2,l,i,1)+ind][0], h1)[1],
																Imr(samples[(level,2,l,i,1)+ind][1], h2)[1],
																Imr(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Imr
				else:
					if(active[2]): samples[(level,2,l,i,2)+ind] = (np.array([Ikpw(samples[(level,2,l,i,1)+ind][0], h1)[1],
																None,
																Ikpw(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Ikpw
					if(active[3]): samples[(level,2,l,i,3)+ind] = (np.array([Iwik(samples[(level,2,l,i,1)+ind][0], h1)[1],
																None,
																Iwik(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Iwik
					if(active[4]): samples[(level,2,l,i,4)+ind] = (np.array([Imr(samples[(level,2,l,i,1)+ind][0], h1)[1],
																None,
																Imr(samples[(level,2,l,i,1)+ind][2], h3)[1]])) # sample Imr
				if(active[5]): samples[(level,2,l,i,5)+ind] = (np.array([Itildekp(1, dim-1, h1),
															Itildekp(len(grid)-3, dim-1, h2),
															Itildekp(1, dim-1, h3)])) # sample Itilde
				if(active[6]): samples[(level,2,l,i,6)+ind] = (np.array([Ihatkp(1, dim, h1),
															Ihatkp(len(grid)-3, dim, h2),
															Ihatkp(1, dim, h3)])) # sample Ihat
				if(active[7]): samples[(level,2,l,i,7)+ind] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,dim)),
															np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,dim)),
															np.random.normal(0, np.sqrt(h3), size = (1,dim))])) # sample Xi

			samples = mlp_generateSamples(samples, t_approx + (total_time - t_approx)*samples[(level,2,l,i,0)+ind], total_time, start_time, M, l, ind + (i,l,), num_gridpoint, dim, level + 1, active, time_dist_exponent)
			samples = mlp_generateSamples(samples, t_approx + (total_time - t_approx)*samples[(level,2,l,i,0)+ind], total_time, start_time, M, -(l-1), ind + (i,-(l-1),), num_gridpoint, dim, level + 1, active, time_dist_exponent)
		
	return samples
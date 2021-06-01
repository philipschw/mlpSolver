import numpy as np
import sdeint
import sdeint2
import math

class Sampler(object):
	"""Base class for defining PDE related function."""
	
	def __init__(self, eqn_config, dimension, num_gridpoint):
		self.dim_system = eqn_config.dim_system
		self.total_time = eqn_config.total_time
		self.start_time = eqn_config.start_time
		self.dim = dimension
		self.num_gridpoint = num_gridpoint
		
class EulerMaruyama(Sampler):

	def __init__(self, eqn_config, dimension, num_gridpoint):
		super(EulerMaruyama, self).__init__(eqn_config, dimension, num_gridpoint)
	
	def dXsampling(self, mu, sigma, dx_sigma=None, t_start, t_end, x_0, dW):
		# Initialize grid and add starting value of dX at time t_start.
		grid = np.linspace(self.start_time, self.total_time, self.num_gridpoint)
		dX = np.array([x_0])
		
		i_start = int(math.ceil(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_start))
		if(t_start == grid[i_start]): 
			i_start = i_start + 1
			
		i_end = int(math.floor(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_end))
		if(t_end == grid[i_end]): 
			i_end = i_end - 1
			
		if(i_end < i_start):
			return np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (t_end - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(t_end - t_start) * dW[i_start])])
		dX = np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (grid[i_start] - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(grid[i_start] - t_start) * dW[i_start])])
		if(i_end != i_start):
			dX = np.vstack([dX, sdeint.itoEuler(mu, sigma, dX[-1], grid[i_start: (i_end + 1)], np.sqrt((self.total_time - self.start_time) / (self.num_gridpoint - 1)) * dW[(i_start + 1): (i_end + 1)])[1:,]])
		return np.vstack([dX, dX[-1] + mu(dX[-1], grid[i_end]) * (t_end - grid[i_end]) + sigma(dX[-1], grid[i_end]).dot(np.sqrt(t_end - grid[i_end]) * dW[i_end + 1])])
		
class Roessler2010KloedenPlatenWright1992(Sampler):

	def __init__(self, eqn_config, dimension, num_gridpoint):
		super(Roessler2010KloedenPlatenWright1992, self).__init__(eqn_config, dimension, num_gridpoint)
	
	def dXsampling(self, mu, sigma, dx_sigma=None, t_start, t_end, x_0, dW):
		# Initialize grid and add starting value of dX at time t_start.
		grid = np.linspace(self.start_time, self.total_time, self.num_gridpoint)
		dX = np.array([x_0])
		
		i_start = int(math.ceil(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_start))
		if(t_start == grid[i_start]): 
			i_start = i_start + 1
			
		i_end = int(math.floor(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_end))
		if(t_end == grid[i_end]): 
			i_end = i_end - 1
			
		if(i_end < i_start):
			return np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (t_end - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(t_end - t_start) * dW[i_start])])
		dX = np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (grid[i_start] - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(grid[i_start] - t_start) * dW[i_start])])
		if(i_end != i_start):
			dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], grid[i_start: (i_end + 1)], Imethod=sdeint.Ikpw, dW=np.sqrt((self.total_time - self.start_time) / (self.num_gridpoint - 1)) * dW[(i_start + 1): (i_end + 1)], I=None)[1:,]])
		return np.vstack([dX, dX[-1] + mu(dX[-1], grid[i_end]) * (t_end - grid[i_end]) + sigma(dX[-1], grid[i_end]).dot(np.sqrt(t_end - grid[i_end]) * dW[i_end + 1])])
		
class Roessler2010Wiktorsson2001(Sampler):

	def __init__(self, eqn_config, dimension, num_gridpoint):
		super(Roessler2010Wiktorsson2001, self).__init__(eqn_config, dimension, num_gridpoint)
	
	def dXsampling(self, mu, sigma, dx_sigma=None, t_start, t_end, x_0, dW):
		# Initialize grid and add starting value of dX at time t_start.
		grid = np.linspace(self.start_time, self.total_time, self.num_gridpoint)
		dX = np.array([x_0])
		
		i_start = int(math.ceil(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_start))
		if(t_start == grid[i_start]): 
			i_start = i_start + 1
			
		i_end = int(math.floor(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_end))
		if(t_end == grid[i_end]): 
			i_end = i_end - 1
			
		if(i_end < i_start):
			return np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (t_end - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(t_end - t_start) * dW[i_start])])
		dX = np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (grid[i_start] - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(grid[i_start] - t_start) * dW[i_start])])
		if(i_end != i_start):
			dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], grid[i_start: (i_end + 1)], Imethod=sdeint.Iwik, dW=np.sqrt((self.total_time - self.start_time) / (self.num_gridpoint - 1)) * dW[(i_start + 1): (i_end + 1)], I=None)[1:,]])
		return np.vstack([dX, dX[-1] + mu(dX[-1], grid[i_end]) * (t_end - grid[i_end]) + sigma(dX[-1], grid[i_end]).dot(np.sqrt(t_end - grid[i_end]) * dW[i_end + 1])])
				
		
class TamedEulerMaruyama(Sampler):

	def __init__(self, eqn_config, dimension, num_gridpoint):
		super(TamedEulerMaruyama, self).__init__(eqn_config, dimension, num_gridpoint)
	
	def dXsampling(self, mu, sigma, dx_sigma=None, t_start, t_end, x_0, dW):
		# Initialize grid and add starting value of dX at time t_start.
		grid = np.linspace(self.start_time, self.total_time, self.num_gridpoint)
		dX = np.array([x_0])
		
		i_start = int(math.ceil(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_start))
		if(t_start == grid[i_start]): 
			i_start = i_start + 1
			
		i_end = int(math.floor(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_end))
		if(t_end == grid[i_end]): 
			i_end = i_end - 1
			
		if(i_end < i_start):
			return np.vstack([dX, dX[-1] + ((mu(dX[-1], t_start) * (t_end - t_start)) / (1 + (1/(self.num_gridpoint - 1))*np.linalg.norm(mu(dX[-1], t_start)))) + sigma(dX[-1], t_start).dot(np.sqrt(t_end - t_start) * dW[i_start])])
		dX = np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (grid[i_start] - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(grid[i_start] - t_start) * dW[i_start])])
		if(i_end != i_start):
			dX = np.vstack([dX, sdeint2.itoTamedEuler(mu, sigma, dX[-1], grid[i_start: (i_end + 1)], np.sqrt((self.total_time - self.start_time) / (self.num_gridpoint - 1)) * dW[(i_start + 1): (i_end + 1)])[1:,]])
		return np.vstack([dX, dX[-1] + ((mu(dX[-1], grid[i_end]) * (t_end - grid[i_end])) / (1 + (1/(self.num_gridpoint - 1))*np.linalg.norm(mu(dX[-1], grid[i_end])))) + sigma(dX[-1], grid[i_end]).dot(np.sqrt(t_end - grid[i_end]) * dW[i_end + 1])])
		
	
		
class Milstein(Sampler):

	def __init__(self, eqn_config, dimension, num_gridpoint):
		super(Milstein, self).__init__(eqn_config, dimension, num_gridpoint)
	
	def dXsampling(self, mu, sigma, dx_sigma=None, t_start, t_end, x_0, dW):
		# Initialize grid and add starting value of dX at time t_start.
		grid = np.linspace(self.start_time, self.total_time, self.num_gridpoint)
		dX = np.array([x_0])
		
		i_start = int(math.ceil(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_start))
		if(t_start == grid[i_start]): 
			i_start = i_start + 1
			
		i_end = int(math.floor(((self.num_gridpoint - 1) / (self.total_time - self.start_time)) * t_end))
		if(t_end == grid[i_end]): 
			i_end = i_end - 1
			
		if(i_end < i_start):
			return np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (t_end - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(t_end - t_start) * dW[i_start])])
		dX = np.vstack([dX, dX[-1] + mu(dX[-1], t_start) * (grid[i_start] - t_start) + sigma(dX[-1], t_start).dot(np.sqrt(grid[i_start] - t_start) * dW[i_start])])
		if(i_end != i_start):
			dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dX[-1], grid[i_start: (i_end + 1)], np.sqrt((self.total_time - self.start_time) / (self.num_gridpoint - 1)) * dW[(i_start + 1): (i_end + 1)])[1:,]])
		return np.vstack([dX, dX[-1] + mu(dX[-1], grid[i_end]) * (t_end - grid[i_end]) + sigma(dX[-1], grid[i_end]).dot(np.sqrt(t_end - grid[i_end]) * dW[i_end + 1])])
		
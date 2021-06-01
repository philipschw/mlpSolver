import numpy as np

import sampler as spl

class Equation(object):
	"""Base class for defining PDE related function."""
	
	def __init__(self, eqn_config, dimension, num_iteration, samplingMethod, num_gridpoint):
		self.dim = dimension
		self.dim_system = eqn_config.dim_system
		self.total_time = eqn_config.total_time
		self.start_time = eqn_config.start_time
		self.num_iteration = num_iteration
		
	def mu(self, t, x, v):
		"""Drift coefficient in the SDE."""
		raise NotImplementedError
		
	def sigma(self, t, x, v):
		"""Diffusion coefficient in the SDE."""
		raise NotImplementedError
		
	def dx_sigma(self, t, x, v):
		"""Derivative of diffusion coefficient in the SDE."""
		raise NotImplementedError
		
	def dXsample(self, t_start, t_end, x_0, dW):
		"""Sample forward SDE."""
		raise NotImplementedError
		
	def dIsample(self, t_start, t_end, x, dW, Xsample):
		"""Sample derivative process of forward SDE."""
		raise NotImplementedError

	def f(self, t, x, v):
		"""Generator function in the PDE."""
		raise NotImplementedError

	def g(self, t, x):
		"""Terminal condition of the PDE."""
		raise NotImplementedError
		
		
class SemilinearBlackScholes(Equation):
	"""
	Semilinear Black-Scholes PDE from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration, samplingMethod, num_gridpoint):
		super(SemilinearBlackScholes, self).__init__(eqn_config, dimension, num_iteration, samplingMethod, num_gridpoint)
		self.x_init = np.ones(self.dim) * 50.0
		self.samplingMethod = samplingMethod
		self.num_gridpoint = num_gridpoint
		if(self.samplingMethod != "Explicit"): 
			if(self.num_gridpoint == 1):
				raise ValueError("SDE Approximation schemes with 1 gridpoint are not valid.")
			else:
				self.sampleMethod = getattr(spl, self.samplingMethod)(eqn_config, dimension, num_gridpoint)
				
		self.dx_sigmaMatrix = np.zeros((self.dim, self.dim, self.dim))
		for i in range(self.dim):
			self.dx_sigmaMatrix[i, i, i] = 1.0
				
				
	def mu(self, x, t):
		return x
	
	def sigma(self, x, t):
		return np.diag(x)
		
	def dx_sigma(self, x, t):
		return self.dx_sigmaMatrix
		
	def dXsample(self, t_start, t_end, x_0, dW):
		if(self.samplingMethod != "Explicit"):
			return self.sampleMethod.dXsampling(self.mu, self.sigma, self.dx_sigma, t_start, t_end, x_0, dW)
		else:
			return x_0 * np.exp(0.5*(t_end - t_start) +  np.sqrt(t_end - t_start) * dW)

	def f(self, t, x, v):
		return np.array([v / (1 + v**2)])

	def g(self, x):
		return np.array([np.log(0.5 * (1 + np.linalg.norm(x)**2))])
		
		
class SystemSemilinearHeatEquation(Equation):
	"""
	System of semilinear heat PDEs from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(SystemSemilinearHeatEquation, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + np.sqrt(2) * np.sqrt(t_end - t_start) * dW

	def f(self, t, x, v):
		return np.array([v[1] / (1 + v[1]**2), (2*v[0]) / 3])

	def g(self, x):
		return np.array([1 / (2 + 0.4 * np.linalg.norm(x)**2), np.log(0.5 * (1 + np.linalg.norm(x)**2))])


class AllenCahn(Equation):
	"""
	Allen-Cahn PDE from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(AllenCahn, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + np.sqrt(2)*np.sqrt(t_end - t_start)*dW

	def f(self, t, x, v):
		return np.array([v - v**3])

	def g(self, x):
		return np.array([1 / (2 + 0.4 * np.linalg.norm(x)**2)])
	
	
class SineGordon(Equation):
	"""
	Sine-Gordon type PDE from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(SineGordon, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + np.sqrt(2)*np.sqrt(t_end - t_start)*dW

	def f(self, t, x, v):
		return np.array([np.sin(v)])

	def g(self, x):
		return np.array([1 / (2 + 0.4 * np.linalg.norm(x)**2)])
	
	
class RecursivePricingDefaultRisk(Equation):
	"""
	Recursive pricing with default risk from paper https://arxiv.org/abs/1607.03295v4
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(RecursivePricingDefaultRisk, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.ones(self.dim) * 100
		self.sigma = eqn_config.sigma
		self.mu = eqn_config.mu
		self.delta = eqn_config.delta
		self.R = eqn_config.R
		self.vh = eqn_config.vh[eqn_config.dim.index(dimension)]
		self.vl = eqn_config.vl[eqn_config.dim.index(dimension)]
		self.gammah = eqn_config.gammah
		self.gammal = eqn_config.gammal

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 * np.exp((self.mu - 0.5*(self.sigma**2))*(t_end - t_start) +  self.sigma * np.sqrt(t_end - t_start) * dW)

	def f(self, t, x, v):
		return np.array([-(1-self.delta) * v * ((v < self.vh) * self.gammah + (v >= self.vl) * self.gammal + ((v >= self.vh) and (v < self.vl)) * (((self.gammah - self.gammal)/(self.vh - self.vl)) * (v - self.vh) + self.gammah)) - self.R * v])

	def g(self, x):
		return np.array([min(x)])


class PricingCreditRisk(Equation):
	"""
	Semilinear PDE for Valuing derivative contracts with counterparty credit risk from paper https://arxiv.org/abs/1607.03295v4
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(PricingCreditRisk, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.ones(self.dim) * 100
		self.sigma = eqn_config.sigma
		self.beta = eqn_config.beta
		self.K1 = eqn_config.K1[eqn_config.dim.index(dimension)]
		self.K2 = eqn_config.K2[eqn_config.dim.index(dimension)]
		self.L = eqn_config.L[eqn_config.dim.index(dimension)]

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 * np.exp(((-0.5)*(self.sigma**2))*(t_end - t_start) +  self.sigma * np.sqrt(t_end - t_start) * dW)

	def f(self, t, x, v):
		return np.array([self.beta*(max(v,0) - v)])

	def g(self, x):
		return np.array([max(min(x)-self.K1,0.0) - max(min(x)-self.K2,0.0) - self.L])


class SemilinearBlackScholesAmericanOption(Equation):
	"""
	Pricing a American Put Option with an Semilinear Black-Scholes PDE from paper https://link.springer.com/article/10.1007%2Fs007800200091
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(SemilinearBlackScholesAmericanOption, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.ones(self.dim) * 40.0
		self.sigma = eqn_config.sigma
		self.mu = eqn_config.mu
		self.R = eqn_config.R
		self.K = eqn_config.K
	
	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 * np.exp((self.mu - 0.5*(self.sigma**2))*(t_end - t_start) +  self.sigma * np.sqrt(t_end - t_start) * dW)

	def f(self, t, x, v):
		return np.array([self.R*self.K*(self.g(x) >= v)])

	def g(self, x):
		return np.array([max(self.K - max(x), 0.0)])
							
						
class PricingDifferentInterestRates(Equation):
	"""
	Pricing Problem of an European option in a financial market with different interest rates for borrowing and lending. from paper https://arxiv.org/abs/1607.03295v4
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(PricingDifferentInterestRates, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.ones(self.dim) * 100.0
		self.sigma = eqn_config.sigma
		self.mu = eqn_config.mu
		self.Rl = eqn_config.Rl
		self.Rb = eqn_config.Rb
		
	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 * np.exp((self.mu - 0.5*(self.sigma**2))*(t_end - t_start) +  self.sigma * np.sqrt(t_end - t_start) * dW)
	
	def dIsample(self, t_start, t_end, x, dW, Xsample):
		return np.concatenate([np.ones(1), (np.sqrt(t_end - t_start) * dW[-1]) / (t_end - t_start)])

	def f(self, t, x, v):
		return np.array([[-self.Rl * v[0][0] - ((self.mu - self.Rl) / self.sigma) * sum(v[0][1:]) + (self.Rb - self.Rl) * max((sum(v[0][1:]) / self.sigma) - v[0][0],0.0)]])

	def g(self, x):
		if(self.dim == 1):
			return np.array([[max(x[0]-100.0, 0.0)]])
		else:
			return np.array([[max(max(x)-120.0,0.0) - 2*max(max(x)-150.0,0.0)]])
		
		
class SystemSemilinearHeatEquationTEST(Equation):
	"""
	System of semilinear heat PDEs from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(SystemSemilinearHeatEquationTEST, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + np.sqrt(2)*np.sqrt(t_end - t_start)*dW
	
	def dIsample(self, t_start, t_end, x, dW, Xsample):
		dI = np.zeros(self.dim + 1)
		dI[0] = 1.0
		return dI

	def f(self, t, x, v):
		return np.array([[v[1][0] / (1 + v[1][0]**2)], [(2*v[0][0]) / 3]])

	def g(self, x):
		return np.array([[1 / (2 + 0.4 * np.linalg.norm(x)**2)], [np.log(0.5 * (1 + np.linalg.norm(x)**2))]])


class ExampleExplicitSolution(Equation):
	"""
	System of semilinear heat PDEs from paper arxiv.org/abs/2005.10206v2
	"""
	def __init__(self, eqn_config, dimension, num_iteration):
		super(ExampleExplicitSolution, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)
		self.sigma = eqn_config.sigma

	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + self.sigma*np.sqrt(t_end - t_start)*dW
	
	def dIsample(self, t_start, t_end, x, dW, Xsample):
		return np.concatenate([np.ones(1),(np.sqrt(t_end - t_start)*dW[-1]) / (t_end - t_start)])

	def f(self, t, x, v):
		return np.array([[self.sigma * (v[0][0] - ((2+(self.sigma**2)*self.dim) / (2*(self.sigma**2)*self.dim))) * sum(v[0][1:])]])

	def g(self, x):
		return np.array([[np.exp(self.total_time + sum(x)) / (1 + np.exp(self.total_time + sum(x)))]])	

"""		
class HJBEquation(Equation):
	
	Hamilton-Jacobi-Bellmann (HJB) Equation wich admits a explicit solution from paper https://doi.org/10.1007/s40304-017-0117-6
	
	def __init__(self, eqn_config, dimension, num_iteration):
		super(HJBEquation, self).__init__(eqn_config, dimension, num_iteration)
		self.x_init = np.zeros(self.dim)
		
	def dXsample(self, t_start, t_end, x_0, dW):
		return x_0 + np.sqrt(2)*np.sqrt(t_end - t_start)*dW
	
	def dIsample(self, t_start, t_end, x, dW, Xsample):
		return np.concatenate([np.ones(1),(np.sqrt(t_end - t_start)*dW[-1]) / (t_end - t_start)])

	def f(self, t, x, v, M):
		return np.array([[-min(np.log(np.log(M)), max(-np.log(np.log(M)), np.linalg.norm(v[0][1:])**2))]])

	def g(self, x):
		return np.array([[np.log(0.5 * (1 + np.linalg.norm(x)**2))]])
"""
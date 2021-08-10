import numpy as np
import sdeint
from sdeint.integrate import SDEValueError
import math

try:
    import sdeint2
except ImportError:
    pass

class Sampler(object):
    """Base class for SDE approximation schemes."""
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        self.dim_system = eqn_config.dim_system
        self.total_time = eqn_config.total_time
        self.start_time = eqn_config.start_time
        self.dim = dimension
        self.num_gridpoint = num_gridpoint
        
    def find_grid(self, t_start, t_end):
        """
        Calculates the grid for a given subinterval [t_start, t_end] of 
        the interval [self.start_time, self.total_time], where the main interval is
        discretized by (self.num_gridpoint + 1) gridpoints.
        
        Parameters
        ---------
        t_start: float
            Start time of the subinterval
        t_end: float
            End time of the subinterval
            
        Returns
        ------
        Grid of the subinterval where start and end points are
        t_start and t_end, respectively, and the interior points
        are the grid points of the main interval grid.
        
        """
        
        h = (self.total_time - self.start_time)/self.num_gridpoint
        i_start = int(math.ceil((self.num_gridpoint/(self.total_time - self.start_time)) * t_start))
        if(t_start == i_start*h): i_start = i_start + 1
        i_end = int(math.floor((self.num_gridpoint/(self.total_time - self.start_time)) * t_end))
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
        
    def reshapeD(self, grid, active, dW):
        """
        Reshape the samples for the derivative process D of the
        state process X from dimension d into dimension d**2
        
        Parameters
        ---------
        grid: array
            Sample time points
        active: array
            An array of boolean values which determines which samples are needed
            for the respective sampling method. This array is defined as self.sampleNeeded
            in each sampling method in the module sampler.
        dW: dict
            Samples of the current realization
            
        Returns
        ------
        Reshaped dict dW_reshape of the samples.
        
        """
        dW_reshape = {}
        if(len(grid) == 2):
            if(active[1]): 
                tmp = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(1,)][0]
                dW_reshape[(1,)] = np.array([tmp]) # sample dW
            if(active[2]):
                tmp = np.zeros((1,self.dim**2,self.dim**2))
                tmp[0, 0:self.dim, 0:self.dim] = dW[(2,)][0]
                dW_reshape[(2,)] = np.array([tmp])          # sample Ikpw
            if(active[3]): 
                tmp = np.zeros((1,self.dim**2,self.dim**2))
                tmp[0, 0:self.dim, 0:self.dim] = dW[(3,)][0]
                dW_reshape[(3,)] = np.array([tmp]) # sample Iwik
            if(active[4]): 
                tmp = np.zeros((1,self.dim**2,self.dim**2))
                tmp[0, 0:self.dim, 0:self.dim] = dW[(4,)][0]
                dW_reshape[(4,)] = np.array([tmp]) # sample Imr
            if(active[5]):
                tmp = np.zeros((1,self.dim**2))
                tmp[0, 0:(self.dim-1)] = dW[(5,)][0]
                dW_reshape[(5,)] = np.array([tmp]) # sample Itilde
            if(active[6]): 
                tmp = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(6,)][0]
                dW_reshape[(6,)] = np.array([tmp]) # sample Ihat
            if(active[7]): 
                tmp = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(7,)][0]
                dW_reshape[(7,)] = np.array([tmp]) # sample Xi
        else:
            if(active[1]): 
                tmp = np.zeros((1,self.dim**2))
                tmp2 = np.zeros((len(grid)-3,self.dim**2))
                tmp3 = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(1,)][0]
                tmp2[0:len(grid)-3, 0:self.dim] = dW[(1,)][1]
                tmp3[0, 0:self.dim] = dW[(1,)][2]
                dW_reshape[(1,)] = np.array([tmp, tmp2, tmp3]) # sample dW
                
            if(len(grid) != 3):
                if(active[2]):
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp2 = np.zeros((len(grid)-3,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(2,)][0]
                    tmp2[0:len(grid)-3, 0:self.dim, 0:self.dim] = dW[(2,)][1]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(2,)][2]
                    dW_reshape[(2,)] = np.array([tmp, tmp2, tmp3]) # sample Ikpw
                if(active[3]): 
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp2 = np.zeros((len(grid)-3,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(3,)][0]
                    tmp2[0:len(grid)-3, 0:self.dim, 0:self.dim] = dW[(3,)][1]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(3,)][2]
                    dW_reshape[(3,)] = np.array([tmp, tmp2, tmp3])# sample Iwik
                if(active[4]): 
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp2 = np.zeros((len(grid)-3,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(4,)][0]
                    tmp2[0:len(grid)-3, 0:self.dim, 0:self.dim] = dW[(4,)][1]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(4,)][2]
                    dW_reshape[(4,)] = np.array([tmp, tmp2, tmp3]) # sample Imr
            else:
                if(active[2]):
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(2,)][0]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(2,)][2]
                    dW_reshape[(2,)] = np.array([tmp, None, tmp3]) # sample Ikpw
                if(active[3]): 
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(3,)][0]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(3,)][2]
                    dW_reshape[(3,)] = np.array([tmp, None, tmp3]) # sample Iwik
                if(active[4]): 
                    tmp = np.zeros((1,self.dim**2,self.dim**2))
                    tmp3 = np.zeros((1,self.dim**2,self.dim**2))
                    tmp[0, 0:self.dim, 0:self.dim] = dW[(4,)][0]
                    tmp3[0, 0:self.dim, 0:self.dim] = dW[(4,)][2]
                    dW_reshape[(4,)] = np.array([tmp, None, tmp3])# sample Imr
            if(active[5]):
                tmp = np.zeros((1,self.dim**2))
                tmp2 = np.zeros((len(grid)-3,self.dim**2))
                tmp3 = np.zeros((1,self.dim**2))
                tmp[0, 0:(self.dim-1)] = dW[(5,)][0]
                tmp2[0:len(grid)-3, 0:(self.dim-1)] = dW[(5,)][1]
                tmp3[0, 0:(self.dim-1)] = dW[(5,)][2]
                dW_reshape[(5,)] = np.array([tmp, tmp2, tmp3]) # sample Itilde
            if(active[6]): 
                tmp = np.zeros((1,self.dim**2))
                tmp2 = np.zeros((len(grid)-3,self.dim**2))
                tmp3 = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(6,)][0]
                tmp2[0:len(grid)-3, 0:self.dim] = dW[(6,)][1]
                tmp3[0, 0:self.dim] = dW[(6,)][2]
                dW_reshape[(6,)] = np.array([tmp, tmp2, tmp3]) # sample Ihat
            if(active[7]): 
                tmp = np.zeros((1,self.dim**2))
                tmp2 = np.zeros((len(grid)-3,self.dim**2))
                tmp3 = np.zeros((1,self.dim**2))
                tmp[0, 0:self.dim] = dW[(7,)][0]
                tmp2[0:len(grid)-3, 0:self.dim] = dW[(7,)][1]
                tmp3[0, 0:self.dim] = dW[(7,)][2]
                dW_reshape[(7,)] = np.array([tmp, tmp2, tmp3]) # sample Xi
        
        return dW_reshape
        

class EulerMaruyama(Sampler):
    """
    SDE Approximation by the Euler-Maruyama scheme. For details on the approximation scheme see the documentation of the sdeint package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(EulerMaruyama, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        #f=open('test.csv','ab')
        #np.savetxt(f,np.array([grid]))
        #f.close()
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim))]) # sample dW
            return (np.vstack([dX, sdeint.itoEuler(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[2] - grid[1]), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[-1] - grid[-2]), size = (1,self.dim))])) # sample dW
            # Calculate first step
            dX = np.vstack([dX, sdeint.itoEuler(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint.itoEuler(mu, sigma, dX[-1], grid[1:-1], dW[(1,)][1])[1:,]])
                
            # Calculate last step
            return (np.vstack([dX, sdeint.itoEuler(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), dW[(1,)][2])[1:,]]), dW)
        
            
class TamedEulerMaruyama(Sampler):
    """
    SDE Approximation by the drift-tamed Euler-Maruyama scheme. For details on the approximation scheme see the documentation of the sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(TamedEulerMaruyama, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim))]) # sample dW
            return (np.vstack([dX, sdeint2.itoTamedEuler(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[2] - grid[1]), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[-1] - grid[-2]), size = (1,self.dim))])) # sample dW
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoTamedEuler(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoTamedEuler(mu, sigma, dX[-1], grid[1:-1], dW[(1,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoTamedEuler(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), dW[(1,)][2])[1:,]]), dW)
    
        
class itoSRI2Ikpw(Sampler):
    """
    SDE Approximation by the Roessler2010 SRI2 scheme with iterated integral approximation by Kloden, Platen and Wright (1992).
    For details on the approximation scheme see the documentation of the sdeint package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoSRI2Ikpw, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,True,False,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1]]) # sample Ikpw
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][0], I=dW[(2,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1], sdeint.Ikpw(dW[(1,)][1], h2)[1], sdeint.Ikpw(dW[(1,)][2], h3)[1]]) # sample Ikpw
                else:
                    dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1], None, sdeint.Ikpw(dW[(1,)][2], h3)[1]]) # sample Ikpw             
                
            # Calculate first step
            dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][0], I=dW[(2,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], grid[1:-1], Imethod=sdeint.Ikpw, dW=dW[(1,)][1], I=dW[(2,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][2], I=dW[(2,)][2])[1:,]]), dW)
        
            
class itoSRI2Iwik(Sampler):
    """
    SDE Approximation by the Roessler2010 SRI2 scheme with iterated integral approximation by Wiktorsson (2001).
    For details on the approximation scheme see the documentation of the sdeint package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoSRI2Iwik, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,True,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1]]) # sample Iwik
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][0], I=dW[(3,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1], sdeint.Iwik(dW[(1,)][1], h2)[1], sdeint.Iwik(dW[(1,)][2], h3)[1]]) # sample Iwik
                else:
                    dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1], None, sdeint.Iwik(dW[(1,)][2], h3)[1]]) # sample Iwik             
                
            # Calculate first step
            dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][0], I=dW[(3,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], grid[1:-1], Imethod=sdeint.Iwik, dW=dW[(1,)][1], I=dW[(3,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][2], I=dW[(3,)][2])[1:,]]), dW)


class itoSRI2Imr(Sampler):
    """
    SDE Approximation by the Roessler2010 SRI2 scheme with iterated integral approximation by Mrongowius and Roessler (2021).
    For details on the approximation scheme see the documentation of the sdeint and sdeint2 package.
    """
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoSRI2Imr, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,True,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1]]) # sample Iwik
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][0], I=dW[(4,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1], sdeint2.Imr(dW[(1,)][1], h2)[1], sdeint2.Imr(dW[(1,)][2], h3)[1]]) # sample Iwik
                else:
                    dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1], None, sdeint2.Imr(dW[(1,)][2], h3)[1]]) # sample Iwik             
                
            # Calculate first step
            dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][0], I=dW[(4,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], grid[1:-1], Imethod=sdeint2.Imr, dW=dW[(1,)][1], I=dW[(4,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint.itoSRI2(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][2], I=dW[(4,)][2])[1:,]]), dW)

        
class MilsteinIkpw(Sampler):
    """
    SDE Approximation by the Milstein scheme with iterated integral approximation by Kloden, Platen and Wright (1992).
    For details on the approximation scheme see the documentation of the sdeint and sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(MilsteinIkpw, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,True,False,False,False,False,False])
        
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1]]) # sample Ikpw
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][0], I=dW[(2,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1], sdeint.Ikpw(dW[(1,)][1], h2)[1], sdeint.Ikpw(dW[(1,)][2], h3)[1]]) # sample Ikpw
                else:
                    dW[(2,)] = np.array([sdeint.Ikpw(dW[(1,)][0], h1)[1], None, sdeint.Ikpw(dW[(1,)][2], h3)[1]]) # sample Ikpw             
                
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][0], I=dW[(2,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], grid[1:-1], Imethod=sdeint.Ikpw, dW=dW[(1,)][1], I=dW[(2,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint.Ikpw, dW=dW[(1,)][2], I=dW[(2,)][2])[1:,]]), dW)
    

class MilsteinIwik(Sampler):
    """
    SDE Approximation by the Milstein scheme with iterated integral approximation by Wiktorsson (2001).
    For details on the approximation scheme see the documentation of the sdeint and sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(MilsteinIwik, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,True,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1]]) # sample Iwik
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][0], I=dW[(3,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1], sdeint.Iwik(dW[(1,)][1], h2)[1], sdeint.Iwik(dW[(1,)][2], h3)[1]]) # sample Iwik
                else:
                    dW[(3,)] = np.array([sdeint.Iwik(dW[(1,)][0], h1)[1], None, sdeint.Iwik(dW[(1,)][2], h3)[1]]) # sample Iwik             
                
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][0], I=dW[(3,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], grid[1:-1], Imethod=sdeint.Iwik, dW=dW[(1,)][1], I=dW[(3,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint.Iwik, dW=dW[(1,)][2], I=dW[(3,)][2])[1:,]]), dW)


class MilsteinImr(Sampler):
    """
    SDE Approximation by the Milstein scheme with iterated integral approximation by Mrongowius and Roessler (2021).
    For details on the approximation scheme see the documentation of the sdeint and sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(MilsteinImr, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,True,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1]]) # sample Iwik
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][0], I=dW[(4,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                if(len(grid) != 3):
                    dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1], sdeint2.Imr(dW[(1,)][1], h2)[1], sdeint2.Imr(dW[(1,)][2], h3)[1]]) # sample Imr
                else:
                    dW[(4,)] = np.array([sdeint2.Imr(dW[(1,)][0], h1)[1], None, sdeint2.Imr(dW[(1,)][2], h3)[1]]) # sample Imr          
                
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[0], grid[1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][0], I=dW[(4,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], grid[1:-1], Imethod=sdeint2.Imr, dW=dW[(1,)][1], I=dW[(4,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoMilstein(mu, sigma, dx_sigma, dX[-1], np.array([grid[-2], grid[-1]]), Imethod=sdeint2.Imr, dW=dW[(1,)][2], I=dW[(4,)][2])[1:,]]), dW)


class itoSRIC2(Sampler):
    """
    SDE Approximation by the Roessler2010 SRIC2 scheme. For details on the approximation scheme see the documentation of the sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoSRIC2, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,False,False,False,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim))]) # sample dW
            return (np.vstack([dX, sdeint2.itoSRIC2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(grid[1] - grid[0]), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[2] - grid[1]), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(grid[-1] - grid[-2]), size = (1,self.dim))])) # sample dW
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoSRIC2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW[(1,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoSRIC2(mu, sigma, dX[-1], grid[1:-1], dW[(1,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoSRIC2(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), dW[(1,)][2])[1:,]]), dW)
        

class itoSRID2(Sampler):
    """
    SDE Approximation by the Roessler2010 SRID2 scheme. For details on the approximation scheme see the documentation of the sdeint2 package.
    """
    
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoSRID2, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,False,False,False,True])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(7,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample Xi
            return (np.vstack([dX, sdeint2.itoSRID2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW=dW[(1,)][0], Xi=dW[(7,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                dW[(7,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample Xi            
                
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoSRID2(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), dW=dW[(1,)][0], Xi=dW[(7,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoSRID2(mu, sigma, dX[-1], grid[1:-1], dW=dW[(1,)][1], Xi=dW[(7,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoSRID2(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), dW=dW[(1,)][2], Xi=dW[(7,)][2])[1:,]]), dW)
        
        
class itoRI5(Sampler):
    """
    SDE (Weak) Approximation by the Roessler2009 RI5 scheme. For details on the approximation scheme see the documentation of the sdeint2 package.
    """
    def __init__(self, eqn_config, dimension, num_gridpoint):
        super(itoRI5, self).__init__(eqn_config, dimension, num_gridpoint)
        self.sampleNeeded = np.array([True,True,False,False,False,True,True,False])
    
    def dXsampling(self, mu, sigma, dx_sigma, t_start, t_end, x_0, dW=None):
        """
        The sdeint and sdeint2 packages does not allow for non-equidistant discretization grids. Therefore we have to split
        the calculations depending on the grid.
        """
        
        # Initialize grid and add starting value of dX at time t_start.
        dX = np.array([x_0])
        
        # Cast values
        t_start = float(t_start)
        t_end = float(t_end)
        
        # Find gridvalues
        grid = self.find_grid(t_start, t_end)
        
        # Sample or read random variables
        if(len(grid) == 2):
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                dW[(1,)] = np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim))]) # sample dW
                dW[(5,)] = np.array([sdeint2.Itildekp(1, self.dim-1, h1)]) # sample Itilde
                dW[(6,)] = np.array([sdeint2.Ihatkp(1, self.dim, h1)]) # sample Ihat        
            return (np.vstack([dX, sdeint2.itoRI5(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Itilde=dW[(5,)][0], Ihat=dW[(6,)][0])[1:,]]), dW)
        else:
            if dW is None:
                dW = {}
                h1 = grid[1] - grid[0]
                h2 = grid[2] - grid[1]
                h3 = grid[-1] - grid[-2]
                dW[(1,)] = (np.array([np.random.normal(0, np.sqrt(h1), size = (1,self.dim)),
                                                            np.random.normal(0, np.sqrt(h2), size = (len(grid)-3,self.dim)),
                                                            np.random.normal(0, np.sqrt(h3), size = (1,self.dim))])) # sample dW
                dW[(5,)] = np.array([sdeint2.Itildekp(1, self.dim-1, h1), sdeint2.Itildekp(len(grid)-3, self.dim-1, h2), sdeint2.Itildekp(1, self.dim-1, h3)]) # sample Itilde
                dW[(6,)] = np.array([sdeint2.Ihatkp(1, self.dim, h1), sdeint2.Ihatkp(len(grid)-3, self.dim, h2), sdeint2.Ihatkp(1, self.dim, h3)]) # sample Ihat            
                
            # Calculate first step
            dX = np.vstack([dX, sdeint2.itoRI5(mu, sigma, dX[-1], np.array([grid[0], grid[1]]), Itilde=dW[(5,)][0], Ihat=dW[(6,)][0])[1:,]])
            # Calculate all steps between first and last
            if(len(grid) != 3):
                dX = np.vstack([dX, sdeint2.itoRI5(mu, sigma, dX[-1], grid[1:-1], Itilde=dW[(5,)][1], Ihat=dW[(6,)][1])[1:,]])
            # Calculate last step
            return (np.vstack([dX, sdeint2.itoRI5(mu, sigma, dX[-1], np.array([grid[-2], grid[-1]]), Itilde=dW[(5,)][2],Ihat=dW[(6,)][2])[1:,]]), dW)
        
        
        
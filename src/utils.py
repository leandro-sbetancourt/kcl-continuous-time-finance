from random import random
import numpy as np
from scipy import integrate
from tqdm import tqdm


class BrownianMotion:
    """
    Defines and simulates a standard Brownian motion
    """
    def __init__(self, T, Nt):
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        
    def simulate_BM(self, nsims = 1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = 0.
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + np.sqrt(dt) * errs[t,:]
        return x


class ArithmeticBrownianMotion:
    """
    Model parameters for the environment.
    """
    def __init__(self, x0 = 0, mu = 0.5, sigma = 1.0, T = 1.0, Nt = 100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))

    def simulate_ABM(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + dt * self.mu + np.sqrt(dt) * self.sigma * errs[t,:]
        return x


class GeometricBrownianMotion:
    """
    Model parameters for the environment.
    """
    def __init__(self, x0 = 100., mu = 0.05, sigma = 0.1, T = 1.0, Nt = 100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))

    def simulate_GBM(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] * np.exp( dt * (self.mu - 0.5*self.sigma**2) + np.sqrt(dt) * self.sigma * errs[t,:] ) 
        return x

    
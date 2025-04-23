import numpy as np

class linear:
    def __init__(self, score, totalCustomers):
        self.str = 'linear'
        self.totalCustomers = totalCustomers
        self.score = score

    def eval(self, n):
        return (self.score / self.totalCustomers) * n

class exponential:
    def __init__(self, score, totalCustomers):
        self.str = 'exponential'
        self.totalCustomers = totalCustomers
        self.score = score
    
    def eval(self, n):
        if n == 0: return 0
        return np.exp((np.log(self.score) / self.totalCustomers) * n)

class exponential_with_initial:
    def __init__(self, score, totalCustomers, initialPercentage):
        self.str = 'exponential_with_initial'
        self.score = score
        self.totalCustomers = totalCustomers
        self.initialPercentage = initialPercentage
    
    def eval(self,n):
        if n == 0: return 0
        return self.initialPercentage * self.score + np.exp((np.log((1 - self.initialPercentage) * self.score) / self.totalCustomers) * n)

class logarithmic:
    def __init__(self, score, totalCustomers):
        self.str = 'logarithmic'
        self.totalCustomers = totalCustomers
        self.score = score
    
    def eval(self, n):
        if n == 0: return 0
        if self.totalCustomers == 1: return self.score
        return np.emath.logn(self.totalCustomers**(1/self.score), n)

class quadratic:
    def __init__(self, score, totalCustomers):
        self.str = 'quadratic'
        self.totalCustomers = totalCustomers
        self.score = score
    
    def eval(self, n):
        #Makes so f(totalCustomers/2) = score
        return (-4 * self.score / self.totalCustomers**2) * n**2 + (4 * self.score / self.totalCustomers) * n

import numpy as np

class ClusterFunction():
    '''Base class for representing cluster functions.'''

    def __init__(self, score: float, totalCustomers: int, name: str):
        self.str = name
        self.totalCustomers = totalCustomers
        self.score = score
    
    def eval(self, n:int):
        raise NotImplementedError

class linear(ClusterFunction):
    def __init__(self, score, totalCustomers):
        super().__init__(score, totalCustomers, 'linear')

    def eval(self, n):
        return (self.score / self.totalCustomers) * n

class exponential(ClusterFunction):
    def __init__(self, score, totalCustomers):
        super().__init__(score, totalCustomers, 'exponential')
    
    def eval(self, n):
        if n == 0: return 0
        return np.exp((np.log(self.score) / self.totalCustomers) * n)

class exponential_with_initial(ClusterFunction):
    def __init__(self, score, totalCustomers, initialPercentage):
        super().__init__(score, totalCustomers, 'exponential_with_initial')
        self.initialPercentage = initialPercentage
    
    def eval(self,n):
        if n == 0: return 0
        return self.initialPercentage * self.score + np.exp((np.log((1 - self.initialPercentage) * self.score) / self.totalCustomers) * n)

class logarithmic(ClusterFunction):
    def __init__(self, score, totalCustomers):
        super().__init__(score, totalCustomers, 'logarithmic')

    def eval(self, n):
        if n == 0: return 0
        if self.totalCustomers == 1: return self.score
        return np.emath.logn(self.totalCustomers**(1/self.score), n)

class quadratic(ClusterFunction):
    def __init__(self, score, totalCustomers):
        super().__init__(score, totalCustomers, 'quadratic')
    
    def eval(self, n):
        #Makes so f(totalCustomers/2) = score
        return (-4 * self.score / self.totalCustomers**2) * n**2 + (4 * self.score / self.totalCustomers) * n

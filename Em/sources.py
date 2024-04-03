import numpy as np
import matplotlib.pyplot as plt
class GaussianSource:
    def __init__(self, tc, sigma) -> None:
        self.tc = tc
        self.sigma = sigma
    
    def __call__(self, t):
        return np.exp(-((t-self.tc)/self.sigma)**2)
    
    def __repr__(self) -> str:
        return "MySource(t0={}, sigma={})".format(self.tc, self.sigma)
    
    def plot(self, t):
        plt.plot(t, self(t))
        plt.title("Source")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    def checks(self):
        print("Symmetric Gaussian Source tc >= 5 sigma: {}".format(self.tc >= 5*self.sigma))

    
## test the source
# source = MySource(t0=30, omega=2*np.pi, sigma=5)
# t = np.linspace(0, 100, 100)
# print(source)
# source.plot(t)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class Monte_Carlo:

    def __init__(self, spot, strike, r, sigma, horizon, timesteps, n, seed):
        self.spot = spot
        self.strike = strike
        self.r = r
        self.sigma = sigma
        self.horizon = horizon
        self.timesteps = timesteps
        self.n = n
        self.seed = seed

        self.euler_maruyama = self.Euler_Maruyama(
            spot, strike, r, sigma, horizon, timesteps, n, seed
        )

        self.milstein = self.Milstein(
            spot, strike, r, sigma, horizon, timesteps, n, seed
        )

        self.gbm = self.GBM(
            spot, strike, r, sigma, horizon, timesteps, n, seed
        )

    
    class Euler_Maruyama:

        def __init__(self, spot, strike, r, sigma, horizon, timesteps, n, seed):
            self.spot = spot
            self.strike = strike
            self.r = r
            self.sigma = sigma
            self.horizon = horizon
            self.timesteps = timesteps
            self.n = n
            self.seed = seed


        def Simulate(self, explicit_seed=None):
        
            self.dt = self.horizon/self.timesteps
            self.S = np.zeros((self.timesteps,self.n))
            self.S[0] = self.spot
    
            # np.random.seed(self.seed)  
            np.random.seed(explicit_seed if explicit_seed is not None else self.seed)

            for i in range(0, self.timesteps-1):
                w = np.random.standard_normal(self.n)
                # Euler-Maruyama Scheme
                self.S[i+1] = self.S[i] * (1 + self.r*self.dt + self.sigma * np.sqrt(self.dt) * w)
    
            self.price_path = pd.DataFrame(self.S)
    
            C0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.S[-1]-self.strike))
            P0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.strike-self.S[-1]))
            BC0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] > self.strike).astype(int))
            BP0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] < self.strike).astype(int))
    
    
            # Print the values
            #print(f"\rEuropean Call Option Value is {C0:0.4f}")
            #print(f"\rEuropean Put Option Value is {P0:0.4f}")
            #print(f"\rEuropean Binary Call Option Value is {BC0:0.4f}")
            #print(f"\rEuropean Binary Put Option Value is {BP0:0.4f}")
    
            # Modified to return only call values
            # return C0, P0, BC0, BP0 
            return C0, BC0
        
        def Plot_Underlying_Path(self):
            # Plot initial 100 simulated path using matplotlib
            plt.plot(self.price_path.iloc[:,:1000])
            plt.xlabel('time steps')
            plt.xlim(0,252)
            plt.ylabel('underlying levels')
            plt.title('Monte Carlo Simulated Euler-Maruyama Method');
    
    
    class Milstein:
        
        def __init__(self, spot, strike, r, sigma, horizon, timesteps, n, seed):
            self.spot = spot
            self.strike = strike
            self.r = r
            self.sigma = sigma
            self.horizon = horizon
            self.timesteps = timesteps
            self.n = n
            self.seed = seed

        def Simulate(self, explicit_seed=None):


            self.dt = self.horizon/self.timesteps
            self.S = np.zeros((self.timesteps,self.n))
            self.S[0] = self.spot

            # np.random.seed(self.seed)  
            np.random.seed(explicit_seed if explicit_seed is not None else self.seed)

            for i in range(0, self.timesteps-1):
                w = np.random.standard_normal(self.n)
                # Milstein Scheme
                self.S[i+1] = self.S[i] * (1 + self.r*self.dt + self.sigma * np.sqrt(self.dt) * w
                                            + 0.5*self.sigma**2 * self.dt * (w**2 - 1))

            self.price_path = pd.DataFrame(self.S)

            C0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.S[-1]-self.strike))
            P0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.strike-self.S[-1]))
            BC0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] > self.strike).astype(int))
            BP0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] < self.strike).astype(int))


            # Print the values
            #print(f"\rEuropean Call Option Value is {C0:0.4f}")
            #print(f"\rEuropean Put Option Value is {P0:0.4f}")
            #print(f"\rEuropean Binary Call Option Value is {BC0:0.4f}")
            #print(f"\rEuropean Binary Put Option Value is {BP0:0.4f}")

            # Modified to return only call values
            # return C0, P0, BC0, BP0 
            return C0, BC0
        
        def Plot_Underlying_Path(self):
            # Plot initial 100 simulated path using matplotlib
            plt.plot(self.price_path.iloc[:,:1000])
            plt.xlabel('time steps')
            plt.xlim(0,252)
            plt.ylabel('underlying levels')
            plt.title('Monte Carlo Simulated Milstein Method');


    class GBM:
        
        def __init__(self, spot, strike, r, sigma, horizon, timesteps, n, seed):
            self.spot = spot
            self.strike = strike
            self.r = r
            self.sigma = sigma
            self.horizon = horizon
            self.timesteps = timesteps
            self.n = n
            self.seed = seed

        def Simulate(self, explicit_seed=None):


            self.dt = self.horizon/self.timesteps
            self.S = np.zeros((self.timesteps,self.n))
            self.S[0] = self.spot

            # np.random.seed(self.seed)  
            np.random.seed(explicit_seed if explicit_seed is not None else self.seed)
            for i in range(0, self.timesteps-1):
                w = np.random.standard_normal(self.n)
                # GBM
                self.S[i+1] = self.S[i] * np.exp((self.r - self.sigma**2/2)
                                            * self.dt + self.sigma * np.sqrt(self.dt) * w) 

            self.price_path = pd.DataFrame(self.S)

            C0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.S[-1]-self.strike))
            P0  = np.exp(-(self.r)*self.horizon) * np.mean(np.maximum(0, self.strike-self.S[-1]))
            BC0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] > self.strike).astype(int))
            BP0 = np.exp(-(self.r)*self.horizon) * np.mean((self.S[-1] < self.strike).astype(int))


            # Print the values
            #print(f"\rEuropean Call Option Value is {C0:0.4f}")
            #print(f"\rEuropean Put Option Value is {P0:0.4f}")
            #print(f"\rEuropean Binary Call Option Value is {BC0:0.4f}")
            #print(f"\rEuropean Binary Put Option Value is {BP0:0.4f}")

            # Modified to return only call values
            # return C0, P0, BC0, BP0 
            return C0, BC0
        
        def Plot_Underlying_Path(self):
            # Plot initial 100 simulated path using matplotlib
            plt.plot(self.price_path.iloc[:,:1000])
            plt.xlabel('time steps')
            plt.xlim(0,252)
            plt.ylabel('underlying levels')
            plt.title('Monte Carlo Simulated GBM');  

    
    
    
    def Generate_Single_Random_Path(self):
        # Generate a single random path for the underlying asset using GBM
        dt = self.horizon / self.timesteps
        S = np.zeros(self.timesteps)
        S[0] = self.spot

        np.random.seed(self.seed)  # Set the random seed for reproducibility
        for i in range(1, self.timesteps):
            w = np.random.standard_normal()
            S[i] = S[i-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * w)

        return S
    
    
    def Plot_All_Methods(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Euler-Maruyama plot
        axes[0].plot(self.euler_maruyama.price_path.iloc[:, :1000])
        axes[0].set_title("Euler-Maruyama")
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Underlying Levels")

        # Milstein plot
        axes[1].plot(self.milstein.price_path.iloc[:, :1000])
        axes[1].set_title("Milstein")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Underlying Levels")

        # GBM plot
        axes[2].plot(self.gbm.price_path.iloc[:, :1000])
        axes[2].set_title("GBM Closed Form")
        axes[2].set_xlabel("Time Steps")
        axes[2].set_ylabel("Underlying Levels")

        plt.tight_layout()
        plt.show()
     
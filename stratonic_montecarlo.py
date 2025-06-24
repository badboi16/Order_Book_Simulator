import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

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



class BlackScholes:
    
    def __init__(self, spot, strike, rate, volatility, dte):
        self.S = spot
        self.K = strike
        self.r = rate
        self.t = dte
        self.sigma = volatility

        self.sqrt_time = np.sqrt(self.t)
        self.discount_factor = np.exp(-self.r * self.t)

        self.d1 = self.calculate_d1()
        self.d2 = self.d1 - (self.sigma * self.sqrt_time)

        self.call_price, self.binary_call_price = self.price_option()
        
    def calculate_d1(self) -> float:
        return (np.log(self.S/self.K)+(self.r + (self.sigma**2)/2)*self.t)/(self.sigma * self.sqrt_time)

    def price_option(self) -> Tuple[float, float]:

        call = self.S * norm.cdf(self.d1) - self.K * self.discount_factor * norm.cdf(self.d2)
        put = self.K * self.discount_factor * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

        binary_call = self.discount_factor * norm.cdf(self.d2)
        binary_put = self.discount_factor * norm.cdf(-self.d2)

        return call, binary_call # modified to return only call values
    
    # greeks can be added subsequantly if and when required, I am not going to use them in this project




class Underlying:

    def __init__(self, spot, strike, r, sigma, horizon, timesteps, n, seed=None, samples=None):
        self.spot = spot
        self.strike = strike
        self.r = r
        self.sigma = sigma
        self.horizon = horizon
        self.timesteps = timesteps
        self.n = n
        self.seed = seed
        self.samples = samples

        np.random.seed(self.seed if self.seed is not None else int(datetime.now().timestamp()))

        self.monte_carlo = Monte_Carlo(self.spot, self.strike, self.r, self.sigma,
                               self.horizon, self.timesteps, self.n, self.seed)
        
        self.black_scholes = BlackScholes(self.spot, self.strike, self.r, self.sigma,
                               self.horizon)
        
        # Comparison Functions
        self.method_comparison = self.Method_Comparison(self.monte_carlo, self.black_scholes,
                                self.seed, self.samples)
    
    
    
    
    # The comparison will compare the results of the Monte Carlo methods with
    # the Black-Scholes closed-form solution.
    class Method_Comparison:
        
        def __init__(self, monte_carlo, black_scholes, seed, samples):
            self.monte_carlo = monte_carlo
            self.black_scholes = black_scholes
            self.seed = seed
            self.samples = samples

            if self.samples is not None: # We will skip the comparison if samples is None
                # Setting up random seed for uniformity across all methods in this class
                np.random.seed(self.seed) # Note: Black-Scholes unaffected by random seed
                self.random_numbers = np.random.uniform(1, 10000, self.samples) 
                
                self.CE_EM_error, self.BCE_EM_error = self.BlackScholes_VS_EulerMaruyamaMC()
                self.CE_Mil_error, self.BCE_Mil_error = self.BlackScholes_VS_Milstein()
                self.CE_GBM_error, self.BCE_GBM_error = self.BlackScholes_VS_GBM()

    
        def BlackScholes_VS_EulerMaruyamaMC(self):

            if self.samples is not None:

                error_CE = []
                error_BCE = []

                for i in range(len(self.random_numbers)):
                    # np.random.seed(int(self.random_numbers[i]))

                    C0, BC0 = self.monte_carlo.euler_maruyama.Simulate(explicit_seed = int(self.random_numbers[i])) # Simulate with random seed
                    error_CE.append(C0 - self.black_scholes.call_price)
                    error_BCE.append(BC0 - self.black_scholes.binary_call_price)

                error_CE = pd.Series(error_CE)
                error_BCE = pd.Series(error_BCE)

                return error_CE, error_BCE


        def BlackScholes_VS_Milstein(self):

            if self.samples is not None:

                error_CE = []
                error_BCE = []

                for i in range(len(self.random_numbers)):
                    # np.random.seed(int(self.random_numbers[i]))

                    C0, BC0 = self.monte_carlo.milstein.Simulate(explicit_seed = int(self.random_numbers[i])) # Simulate with random seed
                    error_CE.append(C0 - self.black_scholes.call_price)
                    error_BCE.append(BC0 - self.black_scholes.binary_call_price)

                error_CE = pd.Series(error_CE)
                error_BCE = pd.Series(error_BCE)

                return error_CE, error_BCE

        def BlackScholes_VS_GBM(self):

            if self.samples is not None:
                
                error_CE = []
                error_BCE = []

                for i in range(len(self.random_numbers)):
                    # np.random.seed(int(self.random_numbers[i]))

                    C0, BC0 = self.monte_carlo.gbm.Simulate(explicit_seed = int(self.random_numbers[i])) # Simulate with random seed
                    error_CE.append(C0 - self.black_scholes.call_price)
                    error_BCE.append(BC0 - self.black_scholes.binary_call_price)

                error_CE = pd.Series(error_CE)
                error_BCE = pd.Series(error_BCE)

                return error_CE, error_BCE
            
        def Summary(self, return_df = False):

            # EM_MAD  = (self.CE_EM_error.abs().sum())/len(self.CE_EM_error) # Mean Absolute Deviation
            # Mil_MAD = (self.CE_Mil_error.abs().sum())/len(self.CE_Mil_error) # Mean Absolute Deviation
            # GBM_MAD = (self.CE_GBM_error.abs().sum())/len(self.CE_GBM_error) # Mean Absolute Deviation

            # Dataframe with mean error and stddev
            errors = pd.DataFrame(
                {
                    'Euler-Maruyama': [self.CE_EM_error.mean(), self.CE_EM_error.std(), 
                                        self.BCE_EM_error.mean(), self.BCE_EM_error.std()],
                    'Milstein': [self.CE_Mil_error.mean(), self.CE_Mil_error.std(), 
                                        self.BCE_Mil_error.mean(), self.BCE_Mil_error.std()],
                    'GBM': [self.CE_GBM_error.mean(), self.CE_GBM_error.std(), 
                                        self.BCE_GBM_error.mean(), self.BCE_GBM_error.std()],
                },
                index=['Mean Error (Call)', 'Standard Deviation (Call)', 'Mean (Binary Call)',
                                        'Standard Deviation (Binary Call)']
            )

            # display the errors dataframe
            if return_df==False:
                print(errors)
            else:
                return errors
            
        def SummaryPerc(self, return_df=False):

            # Calculate percentage errors relative to Black-Scholes call and binary call prices
            errors_perc = pd.DataFrame(
            {
            'Euler-Maruyama': [
            (self.CE_EM_error.abs().mean() / self.black_scholes.call_price) * 100,
            (self.CE_EM_error.std() / self.black_scholes.call_price) * 100,
            (self.BCE_EM_error.abs().mean() / self.black_scholes.binary_call_price) * 100,
            (self.BCE_EM_error.std() / self.black_scholes.binary_call_price) * 100,
            ],
            'Milstein': [
            (self.CE_Mil_error.abs().mean() / self.black_scholes.call_price) * 100,
            (self.CE_Mil_error.std() / self.black_scholes.call_price) * 100,
            (self.BCE_Mil_error.abs().mean() / self.black_scholes.binary_call_price) * 100,
            (self.BCE_Mil_error.std() / self.black_scholes.binary_call_price) * 100,
            ],
            'GBM': [
            (self.CE_GBM_error.abs().mean() / self.black_scholes.call_price) * 100,
            (self.CE_GBM_error.std() / self.black_scholes.call_price) * 100,
            (self.BCE_GBM_error.abs().mean() / self.black_scholes.binary_call_price) * 100,
            (self.BCE_GBM_error.std() / self.black_scholes.binary_call_price) * 100,
            ],
            },
            index=[
            'Mean Absolute Error % (Call)',
            'Standard Deviation % (Call)',
            'Mean Absolute Error % (Binary Call)',
            'Standard Deviation % (Binary Call)',
            ],
            )

            # Display the errors dataframe
            if return_df == False:
                print(errors_perc)
            else:
                return errors_perc


        
        def Plot_Error_Distributions(self):
            
            # Create a figure with 3 subplots side by side
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)

            # Plot histogram for error_EM_CE
            axes[0,0].hist(pd.Series(self.CE_EM_error), bins=50, alpha=1, color='blue')
            axes[0,0].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[0,0].set_title('Error Distribution: Euler-Maruyama')
            axes[0,0].set_xlabel('Error')
            axes[0,0].set_ylabel('Frequency')

            # Plot histogram for error_Mil_CE
            axes[0,1].hist(self.CE_Mil_error, bins=50, alpha=1, color='green')
            axes[0,1].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[0,1].set_title('Error Distribution: Milstein')
            axes[0,1].set_xlabel('Error')

            # Plot histogram for error_GBM_CE
            axes[0,2].hist(self.CE_GBM_error, bins=50, alpha=1, color='orange')
            axes[0,2].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[0,2].set_title('Error Distribution: GBM')
            axes[0,2].set_xlabel('Error')

            axes[1,0].hist(self.BCE_EM_error, bins=50, alpha=1, color='blue')
            axes[1,0].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[1,0].set_title('Error Distribution: Euler-Maruyama (Binary)')
            axes[1,0].set_xlabel('Error')
            axes[1,0].set_ylabel('Frequency')

            # Plot histogram for error_Mil_CE
            axes[1,1].hist(self.BCE_Mil_error, bins=50, alpha=1, color='green')
            axes[1,1].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[1,1].set_title('Error Distribution: Milstein (Binary)')
            axes[1,1].set_xlabel('Error')

            # Plot histogram for error_GBM_CE
            axes[1,2].hist(self.BCE_GBM_error, bins=50, alpha=1, color='orange')
            axes[1,2].axvline(x=0, color='red', linestyle='dashed', linewidth=1)
            axes[1,2].set_title('Error Distribution: GBM (Binary)')
            axes[1,2].set_xlabel('Error')

            # Adjust layout
            plt.tight_layout()
            plt.show()
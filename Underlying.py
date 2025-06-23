import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

from Monte_Carlo import Monte_Carlo
from BlackScholes import BlackScholes

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


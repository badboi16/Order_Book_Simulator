import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple

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
    
        

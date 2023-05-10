import numpy as np
from scipy.stats import norm

# Function to calculate the dollar volatility of a calendar spread option
def calendar_spread_dollar_volatility(F1, F2, T, sigma):
    d = (F2 - F1) / (sigma * np.sqrt(T))
    nd = norm.pdf(d)
    N_d = norm.cdf(d)
    dn_d = -d * nd
    C = (F2 - F1) * N_d + (sigma * np.sqrt(T) * (norm.pdf(d) - dn_d))
    partial_deriv_F1 = -N_d + sigma * np.sqrt(T) * dn_d / (F2 - F1)
    partial_deriv_F2 = N_d
    dollar_vol_F1 = partial_deriv_F1 * sigma * F1
    dollar_vol_F2 = partial_deriv_F2 * sigma * F2
    return dollar_vol_F1, dollar_vol_F2, C

# Example usage:
F1 = 100
F2 = 105
T = 0.5
sigma = 0.2
dollar_vol_F1, dollar_vol_F2, option_price = calendar_spread_dollar_volatility(F1, F2, T, sigma)
print('Dollar volatility of F1:', dollar_vol_F1)
print('Dollar volatility of F2:', dollar_vol_F2)
print('Option price:', option_price)

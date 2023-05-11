import yfinance as yf
import numpy as np
from numpy import std, log
from scipy.stats import norm
import datetime
from dateutil.relativedelta import relativedelta




def calendar_spread_dollar_volatility(symbol1, symbol2, T1, T2):
    data1 = yf.download(symbol1, period='1d', start=(datetime.date.today()-relativedelta(years=20)).strftime('%Y-%m-%d'), end=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    data2 = yf.download(symbol2, period='1d', start=(datetime.date.today()-relativedelta(years=20)).strftime('%Y-%m-%d'), end=(datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    prices1 = data1['Close']
    prices2 = data2['Close']

    # Last Available price
    F1 = prices1.iloc[-1]
    F2 = prices2.iloc[-1]
    sigma1=np.log(prices1/prices1.shift(1)).std()*np.sqrt(252)
    # Calculate the time ratio, which is the ratio of T2 to T1 raised to the power of 0.5
    time_ratio = np.sqrt(T2 / T1)
    # Calculate the volatility of the back-month contract by scaling sigma1 by the time ratio
    sigma2 = sigma1 / time_ratio
    # Calculate the overall volatility of the spread as the square root of sigma1 squared plus sigma2 squared
    sigma = np.sqrt(sigma1 ** 2 + sigma2 ** 2)
    # Standardized distance between F1 and F2
    d = (F2 - F1) / (sigma * np.sqrt(T2))
    nd = norm.pdf(d)
    N_d = norm.cdf(d)
    dn_d = -d * nd
    # Bachelier formula
    C = (F2 - F1) * N_d + (sigma * np.sqrt(T2) * (norm.pdf(d) - dn_d))
    # Partial derivatives of C with respect to F1 and F2
    partial_deriv_F1 = -N_d + sigma * np.sqrt(T2) * dn_d / (F2 - F1)
    partial_deriv_F2 = N_d
    dollar_vol_F1 = partial_deriv_F1 * sigma * F1
    dollar_vol_F2 = partial_deriv_F2 * sigma * F2
    overall_dollar_volatility = np.sqrt(dollar_vol_F1 ** 2 + dollar_vol_F2 ** 2)
    return dollar_vol_F1, dollar_vol_F2, C, sigma, overall_dollar_volatility


def extract_month_year(symbol):
    # Mapping for months
    month_mapping = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    month = symbol[2]
    year = int(symbol[3:5])
    # Convert the month to a number
    month_number=month_mapping[month]
    # Compute the last day of the month
    last_day_of_month = datetime.datetime(year + 2000, month_number, 1) + datetime.timedelta(days=32)
    last_day_of_month = last_day_of_month.replace(day=1)
    last_day_of_month -= datetime.timedelta(days=1)
    return last_day_of_month.date()

# Calculate with:
symbol1 = 'NGM23.NYM'
symbol2 = 'NGN23.NYM'

today = datetime.datetime.now().date()

T1 = (datetime.datetime.combine(extract_month_year(symbol1), datetime.time()) - datetime.datetime.combine((today - datetime.timedelta(days=1)), datetime.time())).days / 365
T2 = (datetime.datetime.combine(extract_month_year(symbol2), datetime.time()) - datetime.datetime.combine((today - datetime.timedelta(days=1)), datetime.time())).days / 365


dollar_vol_F1, dollar_vol_F2, option_price, sigma, overall_dollar_volatility = calendar_spread_dollar_volatility(symbol1, symbol2, T1, T2)
print('Dollar volatility of', symbol1, ':', dollar_vol_F1)
print('Dollar volatility of', symbol2, ':', dollar_vol_F2)
print('Overall volatility of the spread', sigma)
print('Option price:', option_price )

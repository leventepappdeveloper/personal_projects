import numpy as np
from scipy.stats import norm
import math

# Function which takes in drift, volatility, strike price, days of expiry, and returns probability
def call_probability(stock_price, call_strike_price, daily_drift, daily_volatility, days_expiry):
    expected_price = stock_price* math.exp(daily_drift*days_expiry)
    print("Duration Drift", daily_drift*days_expiry)
    expected_growth_rate = np.log (expected_price/stock_price)
    standard_deviation = math.sqrt(days_expiry) * daily_volatility
    print("Duration Volatility", standard_deviation)
    z_value = (np.log(call_strike_price/stock_price)-expected_growth_rate)/standard_deviation
    print("Probability ITM:", 1-norm.cdf(z_value))

def put_probability(stock_price, put_strike_price, daily_drift, daily_volatility, days_expiry):
    expected_price = stock_price* math.exp(daily_drift*days_expiry)
    print("Duration Drift", daily_drift*days_expiry)
    expected_growth_rate = np.log (expected_price/stock_price)
    standard_deviation = math.sqrt(days_expiry) * daily_volatility
    print("Duration Volatility", standard_deviation)
    z_value = (np.log(put_strike_price/stock_price)-expected_growth_rate)/standard_deviation
    print("Probability ITM:", norm.cdf(z_value))
    
StockPrice = 102.34
StrikePrice = 105.00
DailyDrift = 0
DailyVolatility = 0.02240
DaysToExpiry = 12

StockPrice = 101.68
StrikePrice = 95.00
DailyDrift = 0
DailyVolatility = 0.02240
DaysToExpiry = 12




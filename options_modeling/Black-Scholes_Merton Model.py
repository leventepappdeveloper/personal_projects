# Levente Papp
import math
from scipy.stats import norm


def csnd(point):
   return (1.0 + math.erf(point/math.sqrt(2.0)))/2.0

# Function which takes in drift, volatility, call prices x2, days of expiry, and returns probability

def durvol_(sigma, days):
  return sigma*math.sqrt(days)

def log_shift_adj_(sigma):
  return ((sigma**2)/2)

def discount_(r, days):
  return math.exp(-r*days/365)

def d1_(sto_pr,str_pr,r,log_shift_adj,days,dur_vol):
  return (math.log(sto_pr/str_pr)+((r/365)+log_shift_adj)*days)/dur_vol

def d2_(d1,dur_vol):
  return d1 - dur_vol

def cum1_(d1):
  return csnd(d1)

def cum2_(d2):
  return csnd(d2)

def call_price(sto_pr,str_pr,r,days,sigma):
  durvol=durvol_(sigma,days)
  log_shift_adj=log_shift_adj_(sigma)
  discount=discount_(r,days)
  d1=d1_(sto_pr,str_pr,r, log_shift_adj,days,durvol)
  d2=d2_(d1,durvol)
  cum1=cum1_(d1)
  cum2=cum2_(d2)
  #print("Duration Volatility:", durvol)
  #print("Call Price:", sto_pr*cum1 - str_pr*discount*cum2)
  return (sto_pr*cum1 - str_pr*discount*cum2)

def put_price(sto_pr,str_pr,r,days,sigma):
  durvol=durvol_(sigma,days)
  log_shift_adj=log_shift_adj_(sigma)
  discount=discount_(r,days)
  d1=d1_(sto_pr,str_pr,r, log_shift_adj,days,durvol)
  d2=d2_(d1,durvol)
  d1=-d1
  d2=-d2
  cum1=cum1_(d1)
  cum2=cum2_(d2)
  #print("Duration Volatility:", durvol)
  #print("Put Price:", -sto_pr*cum1 +str_pr*discount*cum2)
  return -sto_pr*cum1 +str_pr*discount*cum2

# Delta Calculations
def call_delta(stock_price, strike_price, risk_free_rate, days_to_expiry, daily_volatility):
    log_shift_adj = log_shift_adj_(daily_volatility)
    duration_volatility = durvol_(daily_volatility, days_to_expiry)
    
    d1 = d1_(stock_price, strike_price, risk_free_rate, log_shift_adj, days_to_expiry, duration_volatility)
    delta = csnd(d1)
    
    return delta

def put_delta(stock_price, strike_price, risk_free_rate, days_to_expiry, daily_volatility):
    return 1-call_delta(stock_price, strike_price, risk_free_rate, days_to_expiry, daily_volatility)

# Implied Volatitily Calculations

def call_implied_volatility(stock_price, strike_price, call_price, risk_free_rate, days_to_expiry):
    tempcp = float(0)
    cipd = float (0.0)
    while tempcp < call_price:
        cipd = cipd + 0.00001
        d1 = math.log(stock_price/strike_price)+((risk_free_rate/365)+(cipd**2)/2)*days_to_expiry
        durvol = cipd*math.sqrt(days_to_expiry)
        cumd1 = csnd(d1/durvol)
        cumd2 = csnd((d1/durvol) - durvol)
        discount = math.exp(-risk_free_rate*days_to_expiry/365)
        tempcp = (stock_price*cumd1)-(strike_price*discount*cumd2)
    return cipd

def put_implied_volatility(stock_price, strike_price, put_price, risk_free_rate, days_to_expiry):
    tempcp = float(0)
    cipd = float (0.0)
    while tempcp < put_price:
        cipd = cipd + 0.00001
        d1 = math.log(stock_price/strike_price)+((risk_free_rate/365)+(cipd**2)/2)*days_to_expiry
        durvol = cipd*math.sqrt(days_to_expiry)
        cumd1 = csnd(-d1/durvol)
        cumd2 = csnd(-(d1/durvol - durvol))
        discount = math.exp(-risk_free_rate*days_to_expiry/365)
        tempcp = -(stock_price*cumd1)+(strike_price*discount*cumd2)
    return cipd

# Time Decay Calculations
# Note: daystd = 1 if one-day time decay
def call_time_decay(stock_price, strike_price, call_price, risk_free_rate, days_to_expiry, daystd):
    days = days_to_expiry - daystd
    cipd = call_implied_volatility(stock_price, strike_price, call_price, risk_free_rate, days_to_expiry)
    d1 = math.log(stock_price/strike_price)+((risk_free_rate/365)+(cipd**2)/2)*days
    durvol = cipd*math.sqrt(days)
    cumd1 = csnd(d1/durvol)
    cumd2 = csnd((d1/durvol) - durvol)
    discount = math.exp(-risk_free_rate*days/365)
    newcp = (stock_price*cumd1)-(strike_price*discount*cumd2)
    timedecay = call_price - newcp
    return timedecay

def put_time_decay(stock_price, strike_price, put_price, risk_free_rate, days_to_expiry, daystd):
    days = days_to_expiry - daystd
    cipd = call_implied_volatility(stock_price, strike_price, put_price, risk_free_rate, days_to_expiry)
    d1 = math.log(stock_price/strike_price)+((risk_free_rate/365)+(cipd**2)/2)*days
    durvol = cipd*math.sqrt(days)
    cumd1 = csnd(-d1/durvol)
    cumd2 = csnd(-(d1/durvol - durvol))
    discount = math.exp(-risk_free_rate*days/365)
    newcp = (stock_price*cumd1)-(strike_price*discount*cumd2)
    timedecay = put_price - newcp
    return timedecay

# Note: DON'T forget to change last input for time decay according to questio
def call_BSM_IV_Model(stock_price, strike_price, risk_free_rate, days_to_expiry, call_price):
    IV = call_implied_volatility(stock_price, strike_price, call_price, risk_free_rate, days_to_expiry)
    delta = call_delta(stock_price, strike_price, risk_free_rate, days_to_expiry, IV)
    time_decay = call_time_decay(stock_price, strike_price, call_price, risk_free_rate, days_to_expiry, 1)
    
    print("Call Delta:", delta)
    print("Call Implied Volatility:", IV)
    print("Call Time Decay", time_decay)

def put_BSM_IV_Model(stock_price, strike_price, risk_free_rate, days_to_expiry, put_price):
    IV = put_implied_volatility(stock_price, strike_price, put_price, risk_free_rate, days_to_expiry)
    delta = put_delta(stock_price, strike_price, risk_free_rate, days_to_expiry, IV)
    time_decay = put_time_decay(stock_price, strike_price, put_price, risk_free_rate, days_to_expiry, 1)
    
    print("Put Delta:", delta)
    print("Put Implied Daily Volatility:", IV)
    print("Put Time Decay", time_decay)
    
StockPrice = 101.24
StrikePrice = 105.00
RiskFreeRate = 0.000
Days = 25
DailyVolatility = 0.01820

call_price(StockPrice, StrikePrice, RiskFreeRate, Days, DailyVolatility)

StockPrice = 101.24
StrikePrice = 95.00
RiskFreeRate = 0.000
Days = 25
DailyVolatility = 0.01820

put_price(StockPrice, StrikePrice, RiskFreeRate, Days, DailyVolatility)

StockPrice = 236.68
StrikePrice = 240.00
CallValue = 8.150
RiskFreeRate = 0.000
Days = 16

call_BSM_IV_Model(StockPrice, StrikePrice, RiskFreeRate, Days, CallValue)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import numpy as np

import seaborn as sns

def MC(sims,days,drift,stock_pr,sigma):
    setup = np.arange(sims*days)   #note here that we are creating one very long 1D array
    setup = setup.reshape((sims,days),order='C')  #for contiguous columns, order = 'F'
    draw = np.zeros_like(setup, dtype="float32")
    price = np.zeros_like(setup, dtype="float32")
    draw = np.random.standard_normal([sims,days])
    #mean adjustment
    adj_drift = drift - ((sigma**2)/2)


    #set the starting point of every simulation to be stock price
    for i in range(0,sims):
        price[i,0] = stock_pr
    for i in range(0,sims):  # rows (each a different simulation)
        for j in range(1,days):    # cols (each a new day!) 
            price[i,j] = price[i,j-1]*math.exp(adj_drift+sigma*draw[i,j])

    #reference drift for plotting
    ref = np.zeros(days)
    ref[0] = stock_pr
    for m in range(1,days):
        ref[m] = ref[m-1]*math.exp(drift)
    
    return (price, ref)


def plot(sims,price,days,ref,call=None,put=None,callpr=None,putpr=None):
    sns.set_style("darkgrid")
    sns.palplot(sns.color_palette("coolwarm",sims))
    sns.set_palette("coolwarm", sims)
    sns.set_context("poster")
    fig, ax = plt.subplots()
    fig.set_size_inches(14,10)
    for k in range(0,sims):
        plt.plot(price[k,...])
    plt.plot(ref, color="black")
    
    if call:
        plt.plot([call for i in range(0,days)], color="green")
    if put:
        plt.plot([put for i in range(0,days)], color="green")
    if callpr:
        plt.plot([callpr for i in range(0,days)], linestyle=':',color="green")
        plt.plot([call+callpr for i in range(0,days)], color = "red")
    if putpr:
        plt.plot([putpr for i in range(0,days)], linestyle=':',color="green")
        plt.plot([put-putpr for i in range(0,days)], color = "red")
    plt.title("Monte Carlo Simulation")
    plt.show() 

def main():
    days = 252
    sims = 12
    stock_sym = "HMC"
    stock_pr = 100.0
    drift = 0.000238   # our mean, and we could call it that, but it is drift in our model
    sigma = 0.020
    call=130
    put=70
    price,ref=MC(sims,days,drift,stock_pr,sigma)
    plot(sims,price,days,ref,130,70, 2.5, 3.5)
    #TODO count the number of sims in the money and profitable
if __name__ == "__main__":
   main()


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import numpy as np

import seaborn as sns

def non_mapping_call(sims,days,drift,stock_pr,sigma, call_strike, call_price):
    setup = np.arange(sims*days)   #note here that we are creating one very long 1D array
    setup = setup.reshape((sims,days),order='C')  #for contiguous columns, order = 'F'
    draw = np.zeros_like(setup, dtype="float32")
    price = np.zeros_like(setup, dtype="float32")
    np.random.seed(404)
    draw = np.random.standard_normal([sims,days])
    #draw = np.random.seed(642)
    #mean adjustment
    adj_drift = drift - ((sigma**2)/2)

    #set the starting point of every simulation to be stock price
    for i in range(0,sims):
        price[i,0] = stock_pr
    for i in range(0,sims):  # rows (each a different simulation)
        for j in range(1,days):    # cols (each a new day!) 
            price[i,j] = price[i,j-1]*math.exp(adj_drift+sigma*draw[i,j])

    #reference drift for plotting
    ref = np.zeros(days)
    ref[0] = stock_pr
    for m in range(1,days):
        ref[m] = ref[m-1]*math.exp(drift)
    
    # calculate the frequency that the option is in the money on expiry
    option_ITM_frequency = 0
    for simulation in price:
        if simulation[-1] > call_strike:
            option_ITM_frequency = option_ITM_frequency + 1
    print("Call ITM at expiry frequency: " + str(option_ITM_frequency) + "/" + str(sims))
    print("Call ITM at expiry frequency percentage: " + str(option_ITM_frequency/sims))
    
    # calculate the frequency that the option is in the money at all
    option_ITM_atall_frequency = 0
    for simulation in price:
        for day in simulation:
            if day > call_strike:
                option_ITM_atall_frequency = option_ITM_atall_frequency + 1
    print("Call ITM at all frequency: " + str(option_ITM_atall_frequency) + "/" + str(sims*days))
    print("Call ITM at all frequency percentage: " + str(option_ITM_atall_frequency / (sims*days)))
    
    # profitable at expiry
    option_profitable_frequency = 0
    for simulation in price:
        if simulation[-1] > call_strike + call_price:
            option_profitable_frequency = option_profitable_frequency + 1
    print("Call profitable at expiry frequency: " + str(option_profitable_frequency) + "/" + str(sims))
    print("Call profitable at expiry frequency percentage: " + str(option_profitable_frequency/sims))
    
    # profitable before expiry
    option_profitable_atall_frequency = 0
    for simulation in price:
        for day in simulation:
            if day > call_strike + call_price:
                option_profitable_atall_frequency = option_profitable_atall_frequency + 1
    print("Call profitable at all frequency: " + str(option_profitable_atall_frequency) + "/" + str(sims*days))
    print("Call profitable at all frequency percentage: " + str(option_profitable_atall_frequency / (sims*days)))
    
def non_mapping_put(sims,days,drift,stock_pr,sigma, put_strike, put_price):
    setup = np.arange(sims*days)   #note here that we are creating one very long 1D array
    setup = setup.reshape((sims,days),order='C')  #for contiguous columns, order = 'F'
    draw = np.zeros_like(setup, dtype="float32")
    price = np.zeros_like(setup, dtype="float32")
    np.random.seed(592)
    draw = np.random.standard_normal([sims,days])
    #print(draw)
    #draw = np.random.seed(642)
    #mean adjustment
    adj_drift = drift - ((sigma**2)/2)

    #set the starting point of every simulation to be stock price
    for i in range(0,sims):
        price[i,0] = stock_pr
    for i in range(0,sims):  # rows (each a different simulation)
        for j in range(1,days):    # cols (each a new day!) 
            price[i,j] = price[i,j-1]*math.exp(adj_drift+sigma*draw[i,j])

    #reference drift for plotting
    ref = np.zeros(days)
    ref[0] = stock_pr
    for m in range(1,days):
        ref[m] = ref[m-1]*math.exp(drift)
    
    # calculate the frequency that the option is in the money on expiry
    option_ITM_frequency = 0
    for simulation in price:
        if simulation[-1] < put_strike:
            option_ITM_frequency = option_ITM_frequency + 1
    print("Put ITM at expiry frequency: " + str(option_ITM_frequency) + "/" + str(sims))
    print("Put ITM at expiry frequency percentage: " + str(option_ITM_frequency/sims))
    
    # calculate the frequency that the option is in the money at all
    option_ITM_atall_frequency = 0
    for simulation in price:
        for day in simulation:
            if day < put_strike:
                option_ITM_atall_frequency = option_ITM_atall_frequency + 1
    print("Put ITM at all frequency: " + str(option_ITM_atall_frequency) + "/" + str(sims*days))
    print("Put ITM at all frequency percentage: " + str(option_ITM_atall_frequency / (sims*days)))
    
    # profitable at expiry
    option_profitable_frequency = 0
    for simulation in price:
        if simulation[-1] < put_strike - put_price:
            option_profitable_frequency = option_profitable_frequency + 1
    print("Put profitable at expiry frequency: " + str(option_profitable_frequency) + "/" + str(sims))
    print("Put profitable at expiry frequency percentage: " + str(option_profitable_frequency/sims))
    
    # profitable before expiry
    option_profitable_atall_frequency = 0
    for simulation in price:
        for day in simulation:
            if day < put_strike - put_price:
                option_profitable_atall_frequency = option_profitable_atall_frequency + 1
    print("Put profitable at all frequency: " + str(option_profitable_atall_frequency) + "/" + str(sims*days))
    print("Put profitable at all frequency percentage: " + str(option_profitable_atall_frequency / (sims*days)))

Simulations = 1000
Days = 20
DailyDrift = 0.000348
StockPrice = 53.25
DailyVolatility = 0.02400
StrikePrice = 55.00
CallPrice = 1.32





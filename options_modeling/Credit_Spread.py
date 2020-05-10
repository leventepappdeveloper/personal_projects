# Levente Papp
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

call = True
stosym = "SPY"
expiry = "20190517"
days = int(16)

stock_price = 266.27

short_strike = 267.00
short_price = 2.150

hedge_strike = 269.00
hedge_price = 1.210 

short_delta = 0.4570
hedge_delta = 0.3335

spread = hedge_strike - short_strike
max_gain = short_price - hedge_price
max_loss = max_gain - spread

prob_max_gain = 1 - short_delta
prob_max_loss = hedge_delta
prob_between = 1 - prob_max_gain - prob_max_loss

value_max_gain = max_gain * prob_max_gain
value_max_loss = max_loss * prob_max_loss
value_between = (0.6*max_gain + 0.4*max_loss) * prob_between

expected_value = value_max_gain + value_max_loss + value_between

print ("Spread: {:.2f}, Maximum gain: {:.3f}, and Maximum Loss: {:.3f}.".format(spread,max_gain,max_loss))

num_strikes = 19  #default 19 - must be an odd number
sides = (num_strikes - 1)/2
#  center = int(stock_price)
center = int(short_strike + (spread/2))
left_offset = 0  # this shifts the center of the mapping left (pos) or right (neg)
low_str = center - sides + left_offset
hi_str = center + sides + left_offset

strikes = np.linspace(low_str, hi_str, num = num_strikes, dtype = int)

gains = np.ones(num_strikes)
gains = gains*max_gain
print(gains)

element1 = np.where(strikes == short_strike)
startit = int(element1[0])

element2 = np.where(strikes == hedge_strike)
endit = int(element2[0])


for i in range(startit+1,endit):
    print(strikes[i])
    gains[i] = gains[i] - (strikes[i] - short_strike)
print (gains)


x = strikes
y = gains
upper_x = int(x.max())
lower_x = int(x.min())
upper_y = int(y.max()) + 2
lower_y = int(y.min()) - 2

fig = plt.figure()
sns.set()
sns.set_style(style="darkgrid")
sns.set_context("poster", font_scale=1.2)
fig.set_size_inches(14,10)
plt.axis([lower_x,upper_x,lower_y,upper_y])
plt.plot(x,y, color="red")
plt.axhline(0, color="black")
plt.axvline(hedge_strike, color="blue", linestyle="--", label="Hedge")
plt.axvline(short_strike, color="green", linestyle="--", label="Short")
plt.axvline(stock_price, color="orange", label="Stock price")
plt.title(stosym + " Credit Spread Payoff")
plt.ylabel("Payoff")
plt.xlabel("Strike prices")
plt.legend()

print ("Underlying: ", stosym)
print ("Stock price: {:.2f}.".format(stock_price))
print ("Expiry: ", expiry)
print ("Days: ", days)
print ("Short strike: {:.2f}.".format(short_strike))
print ("Short strike price: {:.2f}.".format(short_price))
print ("Hedge strike: {:.2f}.".format(hedge_strike))
print ("Hedge strike price: {:.2f}.".format(hedge_price))
print ("Spread: {:.2f}.".format(spread))
print ("Max gain: {:.2f}.".format(max_gain))
print ("Max loss: {:.2f}.".format(max_loss))
print ("Probability of max gain: {:.4f}.".format(prob_max_gain))
print ("Probability of max loss: {:.4f}.".format(prob_max_loss))
print ("Probability of between maxes: {:.4f}.".format(prob_between))
print (("Value of max gain tranche: {:.4f}.".format(value_max_gain)))
print (("Value of max loss tranche: {:.4f}.".format(value_max_loss)))
print (("Value of middle tranche: {:.4f}.".format(value_between)))
print (("Value of bet: {:.4f}.".format(expected_value)))






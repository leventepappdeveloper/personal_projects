# Levente Papp
# Homework 1, Problem 2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as sk

# Air Pollution vs. Life Satisfaction
oecd_bli = pd.read_excel("oecd_bli.xlsx", thousands = ",")

x = np.c_[oecd_bli["Air Pollution"]]
y = np.c_[oecd_bli["Value"]]

oecd_bli.plot(kind = 'scatter', x = 'Air Pollution', y = 'Value')
plt.show()

model = sk.LinearRegression()
model.fit(x, y)
model.coef_

""" Yes, based on the data it seems to be true that smaller air pollution leads to a better life. One way to check 
this is to look at the scatter plot above. It is clear that there is a negative correlation between the two variables.
Another way to check it is by looking at the regression coefficient on the Air Pollution variable: -0.07203. 
This means that a decrease in Air Pollution leads to an increase in Life Satisfaction, according to our model"""
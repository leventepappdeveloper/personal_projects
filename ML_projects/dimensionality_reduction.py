# Levente Papp, 11/5/2019
# In order to make this task computationally less expensive, the models are only run of datasets of size 4,000 as 
# opposed to the whole dataset.
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version = 1)

y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

X_10000 = X.loc[0:4000, :]
y_10000 = y[:4001]


# run t-SNE on first 1000 datapoints
from sklearn.manifold import TSNE

tsne_X_embedded = TSNE(n_components=2).fit_transform(X_10000)

# for plotting purposes, separate first features from second ones
first_features = []
second_features = []
for elt in tsne_X_embedded:
    first_features.append(elt[0])
    second_features.append(elt[1])

# give different colors to different labels
colors = []
for i in range(len(y_10000)):
    if y_10000[i] == 0:
        colors.append("black")
    elif y_10000[i] == 1:
        colors.append("red")
    elif y_10000[i] == 2:
        colors.append("chocolate")
    elif y_10000[i] == 3:
        colors.append("blue")
    elif y_10000[i] == 4:
        colors.append("yellow")
    elif y_10000[i] == 5:
        colors.append("purple")
    elif y_10000[i] == 6:
        colors.append("gray")
    elif y_10000[i] == 7:
        colors.append("orange")
    elif y_10000[i] == 8:
        colors.append("lightsteelblue")
    elif y_10000[i] == 9:
        colors.append("magenta")

# Plot features after t_SNE dimensionality reduction
plt.scatter(first_features, second_features, color = colors)

# run MDS dimensionality reduction algorithm
from sklearn.manifold import MDS

MDS_X_embedded = MDS(n_components=2).fit_transform(X_10000)

# for plotting purposes, separate first features from second ones
first_features = []
second_features = []
for elt in MDS_X_embedded:
    first_features.append(elt[0])
    second_features.append(elt[1])

# give different colors to different labels
colors = []
for i in range(len(y_10000)):
    if y_10000[i] == 0:
        colors.append("black")
    elif y_10000[i] == 1:
        colors.append("red")
    elif y_10000[i] == 2:
        colors.append("chocolate")
    elif y_10000[i] == 3:
        colors.append("blue")
    elif y_10000[i] == 4:
        colors.append("yellow")
    elif y_10000[i] == 5:
        colors.append("purple")
    elif y_10000[i] == 6:
        colors.append("gray")
    elif y_10000[i] == 7:
        colors.append("orange")
    elif y_10000[i] == 8:
        colors.append("lightsteelblue")
    elif y_10000[i] == 9:
        colors.append("magenta")

# Plot features after t_SNE dimensionality reduction
plt.scatter(first_features, second_features, color = colors)





# Levente Papp, 11/5/2019
# For each "shape" it seems that the optimal number of neighbors DEPENDS on the size of the dataset. However, 
# since my sampling is not perfectly uniform, the results I get do not align with my intuition that the number of 
# optimal neighbors should increase as the number of instances in the dataset increases. 
import numpy as np
import pandas as pd
import random as random
import math
import matplotlib.pyplot as plt

# Generate Klein Bottle Data
def klein_bottle(num_obs):
    data = pd.DataFrame(columns=["x", "y", "z"])
    u_list = np.linspace(0, math.pi, int(math.sqrt(num_obs)))
    v_list = np.linspace(0, 2*math.pi, int(math.sqrt(num_obs)))
    for i in range(int(math.sqrt(num_obs))):
        for j in range(int(math.sqrt(num_obs))):
            u = u_list[i]
            v = v_list[j]
            x = -2.0/15*np.cos(u)*(3*np.cos(v)-30*np.sin(u)+90*(np.cos(u))**4*np.sin(u)-60*(np.cos(u))**6*np.sin(u)
                              +5*np.cos(u)*np.cos(v)*np.sin(u))
            y = -1.0/15*np.sin(u)*(3*np.cos(v)-3*(np.cos(u))**2*np.cos(v)-48*(np.cos(u))**4*np.cos(v)
                              +48*(np.cos(u))**6*np.cos(v)-60*np.sin(u)+5*np.cos(u)*np.cos(v)*np.sin(u)
                              -5*(np.cos(u))**3*np.cos(v)*np.sin(u)-80*(np.cos(u))**5*np.cos(v)*np.sin(u)
                              +80*(np.cos(u))**7*np.cos(v)*np.sin(u))
            z = 2.0/15*(3+5*np.cos(u)*np.sin(u))*np.sin(v)
            row = {"x": [x], "y": [y], "z": [z]}
            df = pd.DataFrame(data=row)
            data = data.append(df)
            
    return data

klein_bottle_1024 = klein_bottle(1024)
klein_bottle_2048 = klein_bottle(2048)
klein_bottle_4096 = klein_bottle(4096)


# Part b)
# Figure 2
def circular_helicoid(a, b, c, r, num_obs):
    data = pd.DataFrame(columns=["x", "y", "z"])
    u_list = np.linspace(0, r, int(math.sqrt(num_obs)))
    v_list = np.linspace(float(a)/c, float(b)/c, int(math.sqrt(num_obs)))
    for i in range(int(math.sqrt(num_obs))):
        for j in range(int(math.sqrt(num_obs))):
            u = u_list[i]
            v = v_list[j]
            x = u * np.cos(v)
            y = u * np.sin(v)
            z = c * v
            row = {"x": [x], "y": [y], "z": [z]}
            df = pd.DataFrame(data=row)
            data = data.append(df)
    return data

circular_helicoid_1024 = circular_helicoid(2, 3, 4, 3, 1024)
circular_helicoid_2048 = circular_helicoid(2, 3, 4, 3, 2048)
circular_helicoid_4096 = circular_helicoid(2, 3, 4, 3, 4096)

# Part b)
# Figure 23
def spherical_helicoid(c, r, num_obs):
    data = pd.DataFrame(columns=["x", "y", "z"])
    u_list = np.linspace(0, 1, int(math.sqrt(num_obs)))
    v_list = np.linspace(-r, r, int(math.sqrt(num_obs)))
    for i in range(int(math.sqrt(num_obs))):
        for j in range(int(math.sqrt(num_obs))):
            u = u_list[i]
            v = v_list[j]
            x = math.sqrt(r*r - v*v)*np.cos(float(v)/c)
            y = math.sqrt(r*r - v*v)*np.sin(float(v)/c)
            z = v
            row = {"x": [x], "y": [y], "z": [z]}
            df = pd.DataFrame(data=row)
            data = data.append(df)
    return data

spherical_helicoid_1024 = spherical_helicoid(0.5, 3, 1024)
spherical_helicoid_2048 = spherical_helicoid(0.5, 3, 2048)
spherical_helicoid_4096 = spherical_helicoid(0.5, 3, 4096)


# a) Klein Bottle
# From the data below, we can see that the optimal number of neighbors depends on the size of the dataset. 
from sklearn.manifold import LocallyLinearEmbedding

klein_bottle_data = [klein_bottle_1024, klein_bottle_2048, klein_bottle_4096]
optimal_neighbors = []
for data_set in klein_bottle_data:
    minimum_error = float("inf")
    optimal_k = 0
    for k in range(3, 10):
        embedding = LocallyLinearEmbedding(n_neighbors = k, n_components = 2)
        X_transformed = embedding.fit_transform(data_set)
        reconstruction_error = embedding.reconstruction_error_
        if reconstruction_error < minimum_error:
            optimal_k = k
            minimum_error = reconstruction_error
    optimal_neighbors.append(optimal_k)
    
print("Optimal Number of Neighbors for N=1024: " + str(optimal_neighbors[0]))
print("Optimal Number of Neighbors for N=2048: " + str(optimal_neighbors[1]))
print("Optimal Number of Neighbors for N=4096: " + str(optimal_neighbors[2]))


# b) Circular Helicoid
# From the data below, we can see that the optimal number of neighbors depends on the size of the dataset. 
from sklearn.manifold import LocallyLinearEmbedding

circular_helicoid_data = [circular_helicoid_1024, circular_helicoid_2048, circular_helicoid_4096]
optimal_neighbors = []
for data_set in circular_helicoid_data:
    minimum_error = float("inf")
    optimal_k = 0
    for k in range(2, 10):
        embedding = LocallyLinearEmbedding(n_neighbors = k, n_components = 2)
        X_transformed = embedding.fit_transform(data_set)
        reconstruction_error = embedding.reconstruction_error_
        if reconstruction_error < minimum_error:
            optimal_k = k
            minimum_error = reconstruction_error
    optimal_neighbors.append(optimal_k)
    
print("Optimal Number of Neighbors for N=1024: " + str(optimal_neighbors[0]))
print("Optimal Number of Neighbors for N=2048: " + str(optimal_neighbors[1]))
print("Optimal Number of Neighbors for N=4096: " + str(optimal_neighbors[2]))


# c) Spherical Helicoid
# From the data below, we can see that the optimal number of neighbors depends on the size of the dataset. 
from sklearn.manifold import LocallyLinearEmbedding

spherical_helicoid_data = [spherical_helicoid_1024, spherical_helicoid_2048, spherical_helicoid_4096]
optimal_neighbors = []
for data_set in spherical_helicoid_data:
    minimum_error = float("inf")
    optimal_k = 0
    for k in range(5, 10):
        embedding = LocallyLinearEmbedding(n_neighbors = k, n_components = 2)
        X_transformed = embedding.fit_transform(data_set)
        reconstruction_error = embedding.reconstruction_error_
        if reconstruction_error < minimum_error:
            optimal_k = k
            minimum_error = reconstruction_error
    optimal_neighbors.append(optimal_k)
    
print("Optimal Number of Neighbors for N=1024: " + str(optimal_neighbors[0]))
print("Optimal Number of Neighbors for N=2048: " + str(optimal_neighbors[1]))
print("Optimal Number of Neighbors for N=4096: " + str(optimal_neighbors[2]))





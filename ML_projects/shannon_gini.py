# Levente Papp
import math 
import matplotlib.pyplot as plt

def shannon_entropy(data, split_index):
    left_subtree = data[:split_index]
    right_subtree = data[split_index:]
    left_classes = set(left_subtree)
    right_classes = set(right_subtree)
    
    left_sum = 0
    for my_class in left_classes:
        count = left_subtree.count(my_class)
        prob = float(count) / len(left_subtree)
        left_sum = left_sum + (-prob*math.log(prob, 2))
        
    right_sum = 0
    for my_class in right_classes:
        count = right_subtree.count(my_class)
        prob = float(count) / len(right_subtree)
        right_sum = right_sum + (-prob*math.log(prob, 2))
    
    left_weight = len(left_subtree) / len(data)
    right_weight = len(right_subtree) / len(data)
    
    shannon_entropy = left_weight * left_sum + right_weight * right_sum
    
    return shannon_entropy

def gini_index(data, split_index):
    left_subtree = data[:split_index]
    right_subtree = data[split_index:]
    left_classes = set(left_subtree)
    right_classes = set(right_subtree)
    left_weight = len(left_subtree) / len(data)
    right_weight = len(right_subtree) / len(data)
    
    left_sum = 0
    for my_class in left_classes:
        count = left_subtree.count(my_class)
        prob = float(count) / len(left_subtree)
        left_sum = left_sum + prob * prob
    gini_index_left = 1 - left_sum
    
    right_sum = 0
    for my_class in right_classes:
        count = right_subtree.count(my_class)
        prob = float(count) / len(right_subtree)
        right_sum = right_sum + prob * prob
    gini_index_right = 1 - right_sum
    
    gini_index = left_weight * gini_index_left + right_weight * gini_index_right
    
    return gini_index

def misclassification_index(data, split_index):
    left_subtree = data[:split_index]
    right_subtree = data[split_index:]
    left_classes = set(left_subtree)
    right_classes = set(right_subtree)
    left_weight = len(left_subtree) / len(data)
    right_weight = len(right_subtree) / len(data)
    
    max_prob = 0
    for my_class in left_classes:
        count = left_subtree.count(my_class)
        prob = float(count) / len(left_subtree)
        max_prob = max(max_prob, prob)
    left_error = 1 - max_prob
    
    max_prob = 0
    for my_class in right_classes:
        count = right_subtree.count(my_class)
        prob = float(count) / len(right_subtree)
        max_prob = max(max_prob, prob)
    right_error = 1 - max_prob
    
    miss_specification_index = left_weight * left_error + right_weight * right_error
    
    return miss_specification_index


input_data = ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A", "B", "B", "C", "C"]

# Plot Shannon Entropy
entropies = []
for split in range(len(input_data) + 1):
    entropies.append(shannon_entropy(input_data, split))
    
x = range(len(input_data) + 1)
y = entropies
plt.plot(x, y)
plt.xlabel("Index")
plt.ylabel("Shannon Entropy")

# Plot Gini Index
gini_indices = []
for split in range(len(input_data) + 1):
    gini_indices.append(2*gini_index(input_data, split))

x = range(len(input_data) + 1)
y = gini_indices
plt.plot(x, y)
plt.xlabel("Index")
plt.ylabel("2 * Gini Index")

# Misclassification Index
indices = []
for split in range(len(input_data) + 1):
    indices.append(2*misclassification_index(input_data, split))

x = range(len(input_data) + 1)
y = gini_indices
plt.plot(x, y)
plt.xlabel("Index")
plt.ylabel("2 * Misclassification Index")



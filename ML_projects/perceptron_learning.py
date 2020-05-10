# Levente Papp, 11/12/2019
import numpy as np
import random as random
import matplotlib.pyplot as plt

def prediction(dot_product, threshold):
    if dot_product < threshold:
        return 0
    else:
        return 1

def compute_metric(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.dot(np.linalg.norm(vector1), np.linalg.norm(vector2))
    return numerator/denominator

def perceptron_learning(teacher_weight_vector, student_weight_vector, iterations):
    y_values = [0, compute_metric(student_weight_vector, teacher_weight_vector)]
    for i in range(iterations-2):
        x_vector = [random.choice([-1, 1]) for i in range(len(teacher_weight_vector))]
        teacher_product = np.dot(teacher_weight_vector, x_vector)
        student_product = np.dot(student_weight_vector, x_vector)
        teacher_prediction = prediction(teacher_product, 0)
        student_prediction = prediction(student_product, 0)
    
        #update weight vector
        student_weight_vector = list(np.add(student_weight_vector, 
                                            np.multiply([teacher_prediction-student_prediction], x_vector)))
    
        y_values.append(compute_metric(student_weight_vector, teacher_weight_vector))
    return y_values


teacher_weights = [random.choice([-1, 1]) for i in range(10)]
for i in range(4):
    student_weights = [random.choice([-1, 1]) for i in range(10)]
    result = perceptron_learning(teacher_weights, student_weights, 200)
    plt.plot(range(200), result)


teacher_weights = [random.choice([-1, 1]) for i in range(100)]
for i in range(4):
    student_weights = [random.choice([-1, 1]) for i in range(100)]
    result = perceptron_learning(teacher_weights, student_weights, 2000)
    plt.plot(range(2000), result)


teacher_weights = [random.choice([-1, 1]) for i in range(1000)]
for i in range(4):
    student_weights = [random.choice([-1, 1]) for i in range(1000)]
    result = perceptron_learning(teacher_weights, student_weights, 20000)
    plt.plot(range(20000), result)




# Levente Papp
# Homework 1, Problem 3
import random


# this function computes the equilibrium Q-matrix
def q_star(reward_matrix, goal_state, learning_rate):
    # initialize Q-matrix
    q_matrix = []
    for i in range(len(reward_matrix)):
        new_row = []
        for j in range(len(reward_matrix[i])):
            new_row.append(0)
        q_matrix.append(new_row)

    for i in range(1000):
        current_state = random.randint(0, 5)
        while current_state != goal_state:

            # find a random next state
            row = reward_matrix[current_state]
            possible_next_states = []
            for i in range(len(row)):
                if row[i] != -1:
                    possible_next_states.append(i)
            next_state = random.choice(possible_next_states)

            # update Q-matrix
            reward = reward_matrix[current_state][next_state]
            max_q = max(q_matrix[next_state])
            q_matrix[current_state][next_state] = reward + learning_rate * max_q
            current_state = next_state

    return q_matrix

""" Note that the output is actually scaled down by a factor of 5 compared to the output from class"""
# Q-star function: given a reward matrix, a goal state index and a learning rate, it computes the converged Q-values
learning_rate = 0.8
reward_matrix = [[-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 1, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100]]
goal_state = 5

q = q_star(reward_matrix, goal_state, learning_rate)
print(q)


# given a Q-matrix, this function computes the shortest path between two states
def shortest_path(q_matrix, start_state, goal_state):
    current_state = start_state
    path = [current_state]

    while current_state != goal_state:
        current_row = q_matrix[current_state]
        next_state = current_row.index(max(current_row))
        path.append(next_state)
        current_state = next_state
    return path

""" Find shortest path within the graph"""
path = shortest_path(q, 3, goal_state)
print(path)
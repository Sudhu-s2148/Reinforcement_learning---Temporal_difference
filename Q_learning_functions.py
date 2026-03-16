import math
import random
import json
import ast
import math
R_wall  = -20
R_cell  = -1    # drop from -5 → -1, less noise in value propagation
R_goal  = 500
def softmax(list_values, T):
    # Safeguard T to avoid division by zero or extreme sensitivity
    T = max(T, 0.05)

    # 1. Scale values by temperature
    scaled_values = [i / T for i in list_values]

    # 2. Subtract the maximum value (The Stability Trick)
    # This prevents math.exp from receiving huge positive numbers
    max_val = max(scaled_values)
    exp_values = [math.exp(v - max_val) for v in scaled_values]

    # 3. Calculate sum and normalize
    sum_of_exp_values = sum(exp_values)
    probabilities = [round(ev / sum_of_exp_values, 3) for ev in exp_values]

    return probabilities

def state_updater(state, action, rewards):
    i, j = state
    if action == 0:
        new_state = (i, j-1)
    elif action == 1:
        new_state = (i, j+1)
    elif action == 2:
        new_state = (i+1, j)
    else:
        new_state = (i-1, j)

    # bounce back if OOB or wall
    if new_state not in rewards or rewards[new_state] == -20:
        return state
    return new_state

def bellmans_update(rewards, Q_values, state, action, gamma, alpha, goal):
    new_state = state_updater(state, action, rewards)

    reward = rewards[new_state]
    max_Q = 0.0 if new_state == goal else max(Q_values[new_state])

    error = (reward + gamma * max_Q) - Q_values[state][action]
    Q_values[state][action] += alpha * error

    return new_state

#this function is specifically created only for 4 probabilites
def weighted_random(probabilities):
    random_num = random.random()
    cumulative = 0.0
    for i, prob in enumerate(probabilities):
        cumulative += prob
        if random_num < cumulative:
            return i
    return len(probabilities) - 1

#saving and retreiving Q_value data from a json file
def json_save(data, filename):
    data_serializable = {str(k): v for k, v in data.items()}
    with open(filename, "w") as fh:
        json.dump(data_serializable, fh)
def json_load(filename):
    with open(filename, "r") as fh:
        loaded = json.load(fh)
        new_data = {ast.literal_eval(k): v for k, v in loaded.items()}
    return new_data

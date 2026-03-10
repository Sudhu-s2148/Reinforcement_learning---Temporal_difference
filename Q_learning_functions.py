import math
import random
def softmax(list_values, T):
    sum_of_exp_values = 0
    for i in list_values:
        sum_of_exp_values += math.exp(i/T)
    probabilities = [round(math.exp(i/T)/sum_of_exp_values,3) for i in list_values]
    return probabilities

def state_updater(state, action):
    i, j= state
    if action == 0:
        return (i, j+1)
    elif action == 1:
        return (i, j-1)
    elif action == 2:
        return (i+1, j)
    else:
        return (i-1,j)

def bellmans_update(rewards, Q_values, state, action, gamma, alpha, goal):
    new_state = state_updater(state,action)
    if new_state not in Q_values:
        reward = -5
        max_Q = 0.0
        new_state = state
    elif new_state == goal:
        reward = rewards[new_state]
        max_Q = 0.0
    else:
        reward = rewards[new_state]
        max_Q = max(Q_values[new_state])
    target = reward + gamma*max_Q
    error = target - Q_values[state][action]
    Q_values[state][action] = Q_values[state][action]+alpha*error
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
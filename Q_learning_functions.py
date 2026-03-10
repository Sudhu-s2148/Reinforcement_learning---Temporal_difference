import math
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

def bellmans_update(rewards, Q_values, state, action, gamma, alpha):
    new_state = state_updater(state,action)
    if new_state not in rewards.keys():
        reward = -5
    else:
        reward = rewards[new_state]
    target = reward + gamma*max(Q_values[new_state])
    error = target - Q_values[state][action]
    Q_values[state][action] = Q_values[state][action]+alpha*error
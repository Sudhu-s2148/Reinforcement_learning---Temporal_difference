import math
def softmax(index, list_values, T):
    sum_of_exp_values = 0
    for i in list_values:
        sum_of_exp_values += math.exp(i/T)
    prob = math.exp(list_values[index]/T)/sum_of_exp_values
    return prob


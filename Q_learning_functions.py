import math
def softmax(list_values, T):
    sum_of_exp_values = 0
    for i in list_values:
        sum_of_exp_values += math.exp(i/T)
    probabilities = [round(math.exp(i/T)/sum_of_exp_values,3) for i in list_values]
    return probabilities


import math
def softmax(list_values, T):
    sum_of_exp_values = 0
    for i in list_values:
        sum_of_exp_values += math.exp(i/T)
    for i in range(len(list_values)):
        list_values[i] = round(math.exp(list_values[i]/T)/sum_of_exp_values,3)
    return list_values


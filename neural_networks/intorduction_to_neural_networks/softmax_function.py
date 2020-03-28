import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    result = []
    exp_tot = np.exp(L).sum()
    [result.append(np.exp(L[i]) / exp_tot) for i in range(len(L))]
    return result

print(softmax([2, 4, 6]))
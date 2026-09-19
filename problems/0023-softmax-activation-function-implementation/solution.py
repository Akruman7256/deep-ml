import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_scores = np.max(scores)
    
    den = 0
    for j in range(len(scores)):
        den += np.exp(scores[j] - max_scores)

    softmax_list = []
    for i in range(len(scores)):
        softmax_i = (np.exp(scores[i] - max_scores))/(den)
        softmax_list.append(softmax_i)
    
    return softmax_list
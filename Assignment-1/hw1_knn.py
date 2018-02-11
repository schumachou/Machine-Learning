from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function
        self.trains = None
        self.labels = None

    def train(self, features: List[List[float]], labels: List[int]):
        #raise NotImplementedError
        self.trains = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError
        distances = []
        prediction = []
        
        for feature in features:
            distances.append([self.distance_function(feature, train) for train in self.trains])
        
        for distance in distances:
            pos, neg = 0, 0
            sorted_dist = numpy.argsort(distance)
            for i in range(self.k):
                if self.labels[sorted_dist[i]] == 1:
                    pos = pos + 1
                else:
                    neg = neg + 1
            prediction.append(1) if pos > neg else prediction.append(0)
                
        return prediction
            

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)

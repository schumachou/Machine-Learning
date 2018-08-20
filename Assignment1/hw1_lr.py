from __future__ import division, print_function

from typing import List

import numpy 
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################
from copy import deepcopy

class LinearRegression:
    
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.w = [None] * (nb_features +1)

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        #raise NotImplementedError
        t_features = deepcopy(features)
        for feature in t_features:
            feature.append(1)
            
        self.w = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(t_features), t_features)),numpy.dot(numpy.transpose(t_features), values))
      
    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        #raise NotImplementedError
        t_features = deepcopy(features)
        for feature in t_features:
            feature.append(1)
        return numpy.dot(self.w, numpy.transpose(t_features))
    
    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        #raise NotImplementedError
        return self.w
    

class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.w = [None] * (nb_features +1)

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        #raise NotImplementedError     
        t_features = deepcopy(features)
        for feature in t_features:
            feature.append(1)

        identity = numpy.eye(self.nb_features + 1)
        self.w = numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(t_features), t_features) + self.alpha * identity),numpy.dot(numpy.transpose(t_features), values))
        
    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        #raise NotImplementedError       
        t_features = deepcopy(features)
        for feature in t_features:
            feature.append(1)   
        return numpy.dot(self.w, numpy.transpose(t_features))
    
    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        #raise NotImplementedError       
        return self.w


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)

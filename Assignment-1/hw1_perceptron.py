from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. Your algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        
        #raise NotImplementedError
        converge = True
        for i in range(self.max_iteration):
            converge = True
            for feature, label in zip(features, labels):
                if np.dot(self.w, feature) < self.margin/2 and np.dot(self.w, feature) > -self.margin/2:
                    self.w = (np.asarray(self.w) + label*np.asarray(feature)/(np.linalg.norm(feature) + np.nextafter(0,1))).tolist()
                    converge = False        
                elif np.dot(self.w, feature)*label <= -self.margin/2:
                    self.w = (np.asarray(self.w) + label*np.asarray(feature)/np.linalg.norm(feature)).tolist()
                    converge = False
                else:
                    continue
            if converge:
                break
        #print("w:", self.w)
        return converge   
                
        
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        
        #raise NotImplementedError
        results = np.dot(self.w,np.transpose(features))
        return [1 if result > 0 else -1 for result in results]

    def get_weights(self) -> List[float]:
        return self.w
    
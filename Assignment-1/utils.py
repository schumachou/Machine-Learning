from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)

    # raise NotImplementedError    
    return np.mean(np.square(y_pred - y_true))


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    #raise NotImplementedError
    t_pos, f_pos, t_neg, f_neg = (0 for i in range(4))
    
    for i, j in zip(real_labels, predicted_labels):
        if i > j:
            f_neg = f_neg + 1
        elif i < j:
            f_pos = f_pos + 1
        elif i == 1:
            t_pos = t_pos + 1
        else:
            t_neg = t_neg + 1
    precision = t_pos/(t_pos + f_pos)
    recall = t_pos/(t_pos + f_neg)
    return 2*precision*recall/(precision + recall)
    

def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    #raise NotImplementedError
    p_features = []    
    for feature in features:
        t_feature = []
        for i in range(2, k+1):
            t_feature = t_feature + (np.asarray(feature)**i).tolist()
        p_features.append(feature + t_feature)
    return p_features
    

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return np.linalg.norm(np.asarray(point1) - np.asarray(point2))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return np.dot(point1,point2)


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    #raise NotImplementedError
    return -np.exp(-np.linalg.norm(np.asarray(point1) - np.asarray(point2))**2/2)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        #raise NotImplementedError
        n_features = []
        for feature in features:
            n = np.linalg.norm(np.asarray(feature))
            if n == 0:     
                n_features.append(feature)
            else:
                n_features.append((np.asarray(feature)/n).tolist())
        return n_features


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #raise NotImplementedError
        np_features = np.asarray(features)
        minmax = np.vstack((np.amin(np_features, 0),np.amax(np_features, 0)))
        n_features = []
        for feature in np_features:
            n_features.append(((feature - minmax[0])/(minmax[1] - minmax[0])).tolist())
        return n_features
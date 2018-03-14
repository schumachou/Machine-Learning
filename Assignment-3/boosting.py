import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T

		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		predictions = np.zeros(len(features))
		for beta, clf in zip(self.betas, self.clfs_picked):
			predictions += beta * np.array(clf.predict(features))
		return np.where(predictions >= 0, 1, -1).tolist()


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		w = np.full(N, 1/N)
		for t in range(self.T):
			min_error = np.inf
			h = None
			for clf in self.clfs:
				error = np.sum((np.array(clf.predict(features)) != np.array(labels)).astype(int) * w)
				if error < min_error:
					min_error = error
					h = clf

			self.clfs_picked.append(h)
			beta = 1 / 2 * np.log((1 - min_error) / min_error)
			self.betas.append(beta)
			w = np.where((np.array(h.predict(features)) == np.array(labels)), w * np.exp(-beta), w * np.exp(beta))
			w /= np.sum(w)


	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		pi = np.full(N, 0.5)
		f = np.zeros(N)
		for t in range(self.T):
			z = ((np.array(labels) + 1) / 2 - pi) / (pi * (1 - pi))
			w = pi * (1 - pi)
			min_error = np.inf
			h = None
			for clf in self.clfs:
				error = np.sum(w * (z - np.array(clf.predict(features))) ** 2)
				if error < min_error:
					min_error = error
					h = clf
			self.clfs_picked.append(h)
			f += 0.5 * np.array(h.predict(features))
			self.betas.append(0.5)
			pi = 1 / (1 + np.exp(-2 * f))

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	
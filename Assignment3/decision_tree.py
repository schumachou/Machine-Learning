import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of
					  corresponding training samples
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################

			branches = np.asarray(branches)
			B = branches.shape[1]
			tot_samples = np.sum(branches)
			cond_entropy = 0
			for b in range(B):
				samples = np.sum(branches[:, b])
				probabilities = branches[:, b] / samples
				entropies = -probabilities[probabilities.nonzero()] * np.log2(probabilities[probabilities.nonzero()])
				cond_entropy += samples / tot_samples * np.sum(entropies)
			return cond_entropy


		uniq_classes = np.unique(self.labels)
		min_entropy = np.inf
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################

			values = np.asarray(self.features)[:, idx_dim]
			uniq_values = np.unique(values)
			num_branch = uniq_values.size
			branches = np.empty((self.num_cls, num_branch), dtype = np.int32)

			for b in range(num_branch):
				labels = np.array(self.labels)[(values == uniq_values[b]).nonzero()] # get corresponding labels
				for c in range(self.num_cls):
					branches[c, b] = np.sum((labels == uniq_classes[c]).astype(int))

			if conditional_entropy(branches) < min_entropy:
				min_entropy = conditional_entropy(branches)
				self.dim_split = idx_dim
				self.feature_uniq_split = uniq_values.tolist()

		############################################################
		# TODO: split the node, add child nodes
		############################################################

		for value in self.feature_uniq_split:
			indices = (np.array(self.features)[:, self.dim_split] == value).nonzero()
			features = np.delete(self.features, self.dim_split, 1)[indices].tolist()
			labels = np.array(self.labels)[indices].tolist()
			self.children.append(TreeNode(features, labels, np.unique(labels).size))

		# split the child nodes
		for child in self.children:
			if len(child.features[0]) == 0:
				child.splittable = False
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max


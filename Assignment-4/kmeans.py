import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py')

        means = x[np.random.choice(N, size=self.n_cluster, replace=False)]
        distances = np.zeros((N, self.n_cluster))
        r = None
        J = np.inf
        iteration = 0

        for i in range(self.max_iter):

            # assignment step
            for membership, mean in enumerate(means):
                distances[:,membership] = np.linalg.norm(x - mean, axis=1)
            r = np.argmin(distances, axis=1)

            # check for continuing
            J_new = 0
            for membership, mean in enumerate(means):
                J_new += np.sum(np.linalg.norm((x[r == membership] - mean), axis=1))
            J_new /= N
            if abs(J - J_new) <= self.e:
                break
            J = J_new

            # update step
            for membership in range(self.n_cluster):
                if len(x[r == membership]) == 0:
                    continue
                # means[membership] = np.mean(x[r == membership], axis=0)
                means[membership] = np.sum(x[r == membership], axis=0) / len(x[r == membership])

            iteration = i + 1

        return (means, r, iteration)


        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        kmeans = KMeans(self.n_cluster, self.max_iter,self.e)
        centroids, memberships, _ = kmeans.fit(x)

        centroid_labels = np.empty((self.n_cluster))
        for k in range(self.n_cluster):
            if len(y[memberships == k]) == 0:
                centroid_labels[k] == 0
                continue
            centroid_labels[k] = np.bincount(y[memberships == k]).argmax()

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py')

        distances = np.zeros((N, self.n_cluster))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(x - centroid, axis=1)

        return self.centroid_labels[np.argmin(distances, axis=1)]

        # DONOT CHANGE CODE BELOW THIS LINE

import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')

            kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
            centroids, memberships, _ = kmeans.fit(x)
            self.means = centroids

            cov = np.empty((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                cov_tmp = np.zeros((D, D))
                for term in (x[memberships == k] - centroids[k]):
                    cov_tmp += np.matmul(term.reshape((D,1)), term.reshape((D,1)).T)
                cov[k] = cov_tmp / len(x[memberships == k])
            self.variances = cov

            pi = np.empty(self.n_cluster)
            for k in range(self.n_cluster):
                pi[k] = len(x[memberships == k]) / N
            self.pi_k = pi

            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')

            # self.means = np.random.randint(1001, size=(self.n_cluster, D)) / 1000
            self.means = np.random.rand(self.n_cluster, D)

            cov = np.empty((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                cov[k] = np.eye(D)
            self.variances = cov

            self.pi_k = np.repeat(1 / self.n_cluster, self.n_cluster)

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')



        # p_x_z = np.zeros((N, self.n_cluster))
        # for i, point in enumerate(x):
        #     for k in range(self.n_cluster):
        #         p_x_z[i,k] = self.pi_k[k] * (1 / np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(self.variances[k]))) * \
        #                    np.exp(-0.5 * np.matmul(np.matmul((point - self.means[k]).T, np.linalg.inv(self.variances[k])), point - self.means[k]))
        # p_x = np.sum(p_x_z, axis=1)
        # l = np.sum(np.log(p_x))

        # raise Exception('********* DEBUG **********')

        l = self.compute_log_likelihood(x)

        for itr in range(self.max_iter):

            # E step
            p_x_z = np.empty((N, self.n_cluster))
            for i, point in enumerate(x):
                for k in range(self.n_cluster):
                    p_x_z[i,k] = self.pi_k[k] * (1 / np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(self.variances[k]))) * \
                               np.exp(-0.5 * np.matmul(np.matmul((point - self.means[k]).T, np.linalg.inv(self.variances[k])), point - self.means[k]))

            gamma = p_x_z / np.sum(p_x_z, axis=1).reshape((N,1))

            # M step
            for k in range(self.n_cluster):
                self.means[k] = np.sum(gamma[:,k].reshape(N, 1) * x, axis=0) / np.sum(gamma[:,k])

                cov_tmp = np.zeros((D, D))
                for i in range(N):
                    cov_tmp += gamma[i, k] * np.matmul((x[i] - self.means[k]).reshape((D,1)), (x[i] - self.means[k]).reshape((D,1)).T)
                self.variances[k] = cov_tmp / np.sum(gamma[:,k])

                self.pi_k[k] = np.sum(gamma[:,k]) / N

            l_new = self.compute_log_likelihood(x)
            if abs(l - l_new) <= self.e:
                break
            l = l_new

        return itr + 1

        # DONOT MODIFY CODE BELOW THIS LINE


    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')

        samples = np.empty((N, self.means.shape[1]))
        numbers = np.random.multinomial(N, self.pi_k)
        pos = 0
        for k, num in enumerate(numbers):
            samples[pos:pos + num] = np.random.multivariate_normal(self.means[k], self.variances[k], num)
            pos += num
        return samples

# DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')

        N, D = x.shape
        p_x_z = np.empty((N, self.n_cluster))
        for i, point in enumerate(x):
            for k in range(self.n_cluster):
                while np.linalg.matrix_rank(self.variances[k]) != self.variances[k].shape[0]:
                    self.variances[k] += 0.001 * np.eye(D)
                p_x_z[i, k] = self.pi_k[k] * (1 / np.sqrt(np.power(2 * np.pi, D) * np.linalg.det(self.variances[k]))) \
                              * np.exp(-0.5 * np.matmul(np.matmul((point - self.means[k]).T, np.linalg.inv(self.variances[k])),point - self.means[k]))

        p_x = np.sum(p_x_z, axis=1)
        return np.sum(np.log(p_x)).item()

        # DONOT MODIFY CODE BELOW THIS LINE

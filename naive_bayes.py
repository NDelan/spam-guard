'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Delanyo Nutakor
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np

from classifier import Classifier

class NaiveBayes(Classifier):
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Call superclass constructor
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        super().__init__(num_classes)

        self.log_priors = None       
        self.log_likelihoods = None  

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.log_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.log_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samples, num_features = data.shape

        class_counts = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_counts[c] = np.sum(y == c)

        self.log_priors = np.log(class_counts / num_samples)

        self.log_likelihoods = np.zeros((self.num_classes, num_features))

        # Calculate class likelihoods using smoothing:
        # L_c,w = (T_c,w + 1) / (T_c + M)
        for c in range(self.num_classes):
            class_samples = data[y == c]

            T_c = np.sum(class_samples)

            if T_c == 0:
                T_c = 1

            for w in range(num_features):
                T_c_w = np.sum(class_samples[:, w])

                likelihood = (T_c_w + 1) / (T_c + num_features)
                self.log_likelihoods[c, w] = np.log(likelihood)

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        if self.log_priors is None or self.log_likelihoods is None:
            raise ValueError("Model has not been trained yet.")

        num_test_samples = data.shape[0]

        log_posterior = np.zeros((num_test_samples, self.num_classes))

        for c in range(self.num_classes):
            log_posterior[:, c] = self.log_priors[c]

            for i in range(num_test_samples):
                for j in range(data.shape[1]):
                    if data[i, j] > 0:  # If the word appears
                        log_posterior[i, c] += data[i, j] * self.log_likelihoods[c, j]

        return np.argmax(log_posterior, axis=1)
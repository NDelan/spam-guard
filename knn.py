'''knn.py
K-Nearest Neighbors algorithm for classification
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classifier import Classifier

class KNN(Classifier):
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Call superclass constructor
        '''
        super().__init__(num_classes)

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        num_test_samps = data.shape[0]
        predictions = np.zeros(num_test_samps, dtype=int)

        for i in range(num_test_samps):
            test_sample = data[i]
            distances = np.sqrt(np.sum((self.exemplars - test_sample)**2, axis=1))

            k_nearest_indices = np.argsort(distances)[:k]

            k_nearest_classes = self.classes[k_nearest_indices]

            unique_classes, class_counts = np.unique(k_nearest_classes, return_counts=True)
            predictions[i] = unique_classes[np.argmax(class_counts)]

        return predictions


    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
        cmap = ListedColormap(colors[:self.num_classes])

        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x_grid, y_grid = np.meshgrid(samp_vec, samp_vec)

        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        y_pred = self.predict(grid_points, k)

        y_pred_grid = y_pred.reshape(n_sample_pts, n_sample_pts)

        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_grid, y_grid, y_pred_grid, cmap=cmap, shading='auto')

        for c in range(self.num_classes):
            idx = self.classes == c
            plt.scatter(self.exemplars[idx, 0], self.exemplars[idx, 1],
                       c=[colors[c]], edgecolor='k', s=50, label=f'Class {c}')

        plt.colorbar(ticks=range(self.num_classes), label='Class')
        plt.title(f'KNN Decision Boundaries (k={k})')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()


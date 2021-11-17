#############################################
# From Sklearn's t-SNE
# Custom callable class
#############################################

import numpy as np
from sklearn.manifold import TSNE

class tsne:
    def __init__(self, n_components):
        self.name = 'tSNE'
        self.n_components = n_components
        
    def __call__(self, X):
        if(self.n_components > np.min(X.shape)):
            raise ValueError('n_components must be between 0 and min(n_samples, n_features)')

        X_new = TSNE(n_components=self.n_components).fit_transform(X)

        return X_new
#############################################
# From Sklearn's PCA
# Custom callable class
#############################################

import numpy as np
from sklearn.decomposition import PCA

class pca:
    def __init__(self, n_components):
        self.name = 'PCA'
        self.n_components = n_components
        
    def __call__(self, X):
        if(self.n_components > np.min(X.shape)):
            raise ValueError('n_components must be between 0 and min(n_samples, n_features)')

        X_new = self.model.transform(X)

        return X_new

    def fit(self, X):
        if(self.n_components > np.min(X.shape)):
            raise ValueError('n_components must be between 0 and min(n_samples, n_features)')

        self.model = PCA(n_components=self.n_components)
        X_new = self.model.fit_transform(X)

        return X_new
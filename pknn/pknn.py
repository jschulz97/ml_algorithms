import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import random
import csv




#######################################
# Just the PKNN Metric
# x : 1 sample
# P : n prototypes
# Y : n class labels 
# N : n eta bandwidths
# PKNN m : the m param
# PKNN k : knn neighbors
#######################################
def PKNN(x, P, Y, N, m=2, k=5):
    
    # Get distances
    prototypes = np.array(P,dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k).fit(prototypes)
    dists, results = nn.kneighbors(x)

    num_classes = len(set(sorted(Y)))
    uniqs = set(sorted(Y))

    # For each case, classify using mean/sd
    # Get class for each
    # typicalities  = [0] * num_classes
    typicalities  = dict()
    # typ_true_pos  = [0] * num_classes
    # typ_false_pos = [0] * num_classes
    # i_true_pos    = [0] * num_classes
    # i_false_pos   = [0] * num_classes

    for cls in uniqs:
        cls_typ = []
        for res,dis in zip(results,dists):
            typ = 0
            # Calculate Possibilistic score
            for r,d in zip(res,dis):
                # Calc typicality
                n = N[r]
                y = Y[r]
                
                if(y == cls):
                    # try:
                    typ_temp = ( 1/(1+np.power(np.maximum(0,d - n), (2 / (m-1)) )) )
                    # except Exception as e:
                    #     print(e)
                    #     print(1+np.power(np.maximum(0,d - n), (2 / (m-1)) ))
                    typ += typ_temp

                # # Increase class typicality if class matches
                # if(uniqs[r] == cls and typ_temp >= .01):
                #     typ += typ_temp
                #     # typ_true_pos[cls] += typ_temp
                #     # i_true_pos[cls]   += 1
                # # Increase false pos typicality if not
                # else:
                #     _=0
                #     # typ_false_pos[cls] += typ_temp
                #     # i_false_pos[cls]   += 1

            cls_typ.append(round(typ/k,4))
        typicalities[cls] = cls_typ

    # # get mean of typicalities
    # for cls in uniq_labels:
    #     if(i_true_pos[cls] != 0):
    #         typ_true_pos[cls] = typ_true_pos[cls] / i_true_pos[cls]
    #     if(i_false_pos[cls] != 0):
    #         typ_false_pos[cls] = typ_false_pos[cls] / i_false_pos[cls]

    return typicalities











##
# Possibilistic KNN Classifier
# Needs to be fit with all known classes
class PKNN_Model():
    def __init__(self, k, m=2):
        random.seed(0)
        self.k      = k
        self.mean   = dict()
        self.sd     = dict()
        self.nn = NearestNeighbors()


    
    #################################
    # Fit PKNN to training data
    #################################
    def train(self, train_feats, train_labels, etas=None):
        self.train_feats = train_feats
        self.train_labels = train_labels

        uniq_labels = list(set(sorted(train_labels)))

        ## Nearest Neighbor
        # Fit to nearest neighbor index
        self.nn.fit(self.train_feats)
        
        ## Get distances for init
        init_bound = dict()
        self.mx = 0
        self.mn = 100000

        if(not etas):
            # For each class, find dists from each training case to every other training case
            for cls in uniq_labels:
                class_dists = []

                # Compute dists to every other training case, add one K to exclude distance to self
                dists, results = self.nn.kneighbors(self.train_feats, self.k+1)

                # Only add dists from own class
                for i,d in enumerate(dists):
                    if(self.train_labels[i] == cls):
                        class_dists.append(np.array(d).ravel())

                # Chop off first distance (to self)
                # Compute mean/sd
                class_dists     = [v[1:] for v in class_dists]
                class_dists     = np.ravel(np.array(class_dists))
                class_mean      = np.mean(class_dists)
                class_sd        = np.std(class_dists)
                self.mean[cls]  = class_mean
                self.sd[cls]    = class_sd
                init_bound[cls] = class_mean / (3* class_sd)

                tmx = np.max(class_dists)
                if(tmx > self.mx):
                    self.mx = tmx
                tmn = np.min(class_dists)
                if(tmn < self.mn):
                    self.mn = tmn

                self.bounds = init_bound
        
        else:
            for cls,eta in zip(uniq_labels,etas):
                init_bound[cls] = eta
            self.bounds = init_bound

        return self.bounds


    ####################################
    # Predict Possibilistic KNN
    ####################################
    def predict(self, test_set):

        # Get distances
        test_set = np.array(test_set,dtype=np.float32)
        dists, results = self.nn.kneighbors(test_set, self.k)

        # For each case, classify using mean/sd
        # Get class for each
        uniq_labels = list(set(sorted(self.train_labels)))
        typicalities  = [0] * len(uniq_labels)
        typ_true_pos  = [0] * len(uniq_labels)
        typ_false_pos = [0] * len(uniq_labels)
        i_true_pos    = [0] * len(uniq_labels)
        i_false_pos   = [0] * len(uniq_labels)

        for cls in uniq_labels:
            cls_typ = []
            for res,dis in zip(results,dists):
                typ = 0
                # Calculate Possibilistic score
                for r,d in zip(res,dis):
                    # Calc typicality
                    bound = self.bounds[cls]
                    typ_temp = (1/(1+np.power(np.maximum(0,d - bound),2)))

                    # Increase class typicality if class matches
                    if(self.train_labels[r] == cls and typ_temp >= .01):
                        typ += typ_temp
                        typ_true_pos[cls] += typ_temp
                        i_true_pos[cls]   += 1
                    # Increase false pos typicality if not
                    else:
                        typ_false_pos[cls] += typ_temp
                        i_false_pos[cls]   += 1

                cls_typ.append(round(typ/self.k,4))
            typicalities[cls] = cls_typ

        # get mean of typicalities
        for cls in uniq_labels:
            if(i_true_pos[cls] != 0):
                typ_true_pos[cls] = typ_true_pos[cls] / i_true_pos[cls]
            if(i_false_pos[cls] != 0):
                typ_false_pos[cls] = typ_false_pos[cls] / i_false_pos[cls]

        return typicalities

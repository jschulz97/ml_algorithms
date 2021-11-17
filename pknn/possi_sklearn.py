from pyflann import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from progressbar import progressbar
import time
import matplotlib.pyplot as plt
import random
import csv

##
# Possibilistic KNN Classifier
# Needs to be fit with all known classes
class PossiKNN():
    def __init__(self,k,kfit,train_features,train_labels,valid_features,valid_labels, test_features, test_labels, pca=False,typ_threshold='.01'):
        random.seed(0)
        self.k      = k
        self.kfit   = kfit
        self.mean   = dict()
        self.sd     = dict()
        self.nn = NearestNeighbors()
        self.typ_threshold = typ_threshold
        self.pca = pca

        # Get dimensions for training and testing features for each class
        self.train_dims = [s.shape[0] for s in train_features]
        self.valid_dims  = [s.shape[0] for s in valid_features]
        self.test_dims  = [s.shape[0] for s in test_features]

        # Unravel training data
        train_features_ravel = []
        train_labels_ravel = []
        for c, l, d in zip(train_features,train_labels, self.train_dims):
            train_features_ravel.extend(c)
            train_labels_ravel.extend([l] * d)
            
        self.train_features_ravel   = np.array(train_features_ravel)
        self.train_labels_ravel     = np.array(train_labels_ravel)
        self.train_features         = np.array(train_features)
        self.train_labels           = np.array(train_labels)

        # Unravel validation data
        valid_features_ravel = []
        valid_labels_ravel = []
        for c, l, d in zip(valid_features,valid_labels, self.valid_dims):
            valid_features_ravel.extend(c)
            valid_labels_ravel.extend([l] * d)
            
        self.valid_features_ravel   = np.array(valid_features_ravel)
        self.valid_labels_ravel     = np.array(valid_labels_ravel)
        self.valid_features         = np.array(valid_features)
        self.valid_labels           = np.array(valid_labels)

        # Unravel test data
        test_features_ravel = []
        test_labels_ravel = []
        for c, l, d in zip(test_features,test_labels, self.test_dims):
            test_features_ravel.extend(c)
            test_labels_ravel.extend([l] * d)
            
        self.test_features_ravel   = np.array(test_features_ravel)
        self.test_labels_ravel     = np.array(test_labels_ravel)
        self.test_features         = np.array(test_features)
        self.test_labels           = np.array(test_labels)


    def valid(self, ind):

        # Get bounds from individual
        bounds = dict()
        for i,cls in enumerate(self.train_labels):
            bounds[cls] = ind[i]

        # # Pick 50 random samples for validation
        # num_samples   = 0
        # valid_features = []
        # valid_labels   = []

        # for c, l, d in zip(self.valid_features.tolist(), self.valid_labels.tolist(), self.valid_dims):
        #     if(l == 'none'):
        #         # Add all N+1 classes
        #         valid_features.extend(c)
        #         valid_labels.extend([l] * d)
        #     elif(num_samples == 0):
        #         valid_features.extend(c)
        #         valid_labels.extend([l] * d)
        #     else:
        #         # Pick 50 random samples from each 
        #         for i in range(num_samples):
        #             valid_features.append(c[random.randint(0,d-1)])

        #         valid_labels.extend([l] * num_samples)

        # Predict on test data
        
        #print('Predicting...')
        t1           = time.time()
        possi_scores, typ_true_pos, typ_false_pos = self.predict(self.valid_features_ravel,bounds)
        t2           = time.time()
        #print('Average time per prediction:',(t2-t1)/test_features.shape[0])

        #################################
        # build confusion matrix
        # (0,0) True class1
        # (0,1) False class2
        # (1,0) False class1
        # (1,1) True class2
        #################################
        valid_dim_all = len(self.valid_features_ravel)
        classification = [''] * valid_dim_all

        # Use scoring to assign classes
        for i in range(valid_dim_all):
            score = 'none'
            max = 0
            for cls in self.valid_labels:
                if(possi_scores[cls][i] == max):
                    score = 'none'
                elif(possi_scores[cls][i] > max):
                    score = cls
                    max = possi_scores[cls][i]

            classification[i] = score
        
        # Build confusion matrix
        # Score on classification
        classes = self.valid_labels.tolist()
        classes.append('none')
        cm = np.zeros((len(classes),len(classes)))
        score = 0.0
        for i,fit in enumerate(classes):
            for j,test in enumerate(classes):
                for k,c in enumerate(classification):
                    if(fit == self.valid_labels_ravel[k] and test == c):
                        cm[i,j] += 1
                        if(i == j):
                            score += 1.0

        score = np.array(score/valid_dim_all).reshape((1,1))
        #print(score)
        #input()
        #print(bounds)
        return score , list(bounds.values()), list(typ_true_pos.values()), list(typ_false_pos.values())


    def test(self,ind,batch=0,disp_cm=False,alpha=0,desc='test_result'):
        #Get bounds from individual
        bounds = dict()
        for i,cls in enumerate(self.train_labels):
            bounds[cls] = ind[i]

        # num_samples   = batch
        # test_features = []
        # test_labels   = []

        # for c, l, d in zip(self.test_features.tolist(), self.test_labels.tolist(), self.test_dims):
        #     # Add all N+1 classes
        #     if(l == 'none'):
        #         test_features.extend(c)
        #         test_labels.extend([l] * d)
        #     # If no batching, add all images
        #     elif(num_samples == 0):
        #         test_features.extend(c)
        #         test_labels.extend([l] * d)
        #     # Batching
        #     else:
        #         # Pick random samples from each 
        #         for i in range(num_samples):
        #             test_features.append(c[random.randint(0,d-1)])

        #         test_labels.extend([l] * num_samples)

        # Predict on test data

        #print('Predicting...')
        t1           = time.time()
        possi_scores, _, _ = self.predict(self.test_features_ravel,bounds)
        t2           = time.time()
        #print('Average time per prediction:',(t2-t1)/test_features.shape[0])

        # Display best avg typicality between classes
        # avg_typ = 0
        # for cls in self.train_labels:
        #     if(np.mean(possi_scores[cls]) > avg_typ):
        #         avg_typ = np.mean(possi_scores[cls])
        # print('Best typ:',avg_typ)

        #################################
        # build confusion matrix
        # (0,0) True class1
        # (0,1) False class2
        # (1,0) False class1
        # (1,1) True class2
        #################################
        test_dim_all = len(self.test_features_ravel)
        test_dim_np  = self.test_dims[-1]
        classification = [''] * test_dim_all

        # Use scoring to assign classes
        for i in range(test_dim_all):
            score = 'none'
            max = self.typ_threshold
            for cls in self.train_labels:
                if(possi_scores[cls][i] == max):
                    score = 'none'
                elif(possi_scores[cls][i] > max):
                    score = cls
                    max = possi_scores[cls][i]

            classification[i] = score
        
        # Build confusion matrix
        # Score on classification
        classes = self.test_labels.tolist()
        cm = np.zeros((len(classes),len(classes)))
        score = 0.0
        np_score = 0.0
        for i,fit in enumerate(classes):
            for j,test in enumerate(classes):
                for k,c in enumerate(classification):
                    if(fit == self.test_labels_ravel[k] and test == c):
                        cm[i,j] += 1
                        if(i == j):
                            score += 1.0
                            if(c == 'none'):
                                np_score += 1.0

        score    = round(score/test_dim_all,4)
        np_score = round(np_score/test_dim_np,4)
        print('Testing score on',test_dim_all,'images:',score)
        print('N+1 score on',test_dim_np,'images:',np_score)

        # Display Confusion Matrix
        if(disp_cm):
            ## Stem plot of typicalities
            fig = plt.figure()
            ax = fig.add_subplot()
            #fig.suptitle('Typicalities')

            colors = ['b','r','g']
            for cls,clr in zip(self.train_labels,colors):
                ax.stem(possi_scores[cls],linefmt=clr,markerfmt=clr+'o',use_line_collection=True)

            # Build x axis label
            str_x_classes = ''
            for i,s in enumerate(classes):
                str_x_classes += s
                if(i < len(classes)-1):
                    str_x_classes += '                  '

            plt.xlabel(str_x_classes)
            ax.tick_params(
                axis='x',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False) # labels along the bottom edge are off
            #plt.show()
            fig.savefig('./export/cm/'+desc+'_stem'+'.png')
            plt.clf()

            ## Confusion Matrix
            mx = np.max(cm)

            round_ind = []
            for i in ind:
                round_ind.append(round(i,1))

            fig = plt.figure()
            fig.suptitle('Score: '+str(score)+'\n'+str(round_ind))

            ax = fig.add_subplot()
            im = ax.imshow(cm,vmax=mx,vmin=0)

            for j in range(len(classes)):
                for k in range(len(classes)):
                    text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=16)

            # Build y & x axis labels
            str_x_classes = ''
            str_y_classes = ''
            reversed_classes = classes[::-1]
            for i,(s,r) in enumerate(zip(classes,reversed_classes)):
                str_x_classes += s
                str_y_classes += r
                if(i < len(classes)-1):
                    str_x_classes += '           '
                    str_y_classes += '           '

            plt.colorbar(im)
            plt.xlabel(str_x_classes)
            plt.ylabel(str_y_classes)
            ax.tick_params(
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False) # labels along the bottom edge are off
            #plt.show()
            fig.savefig('./export/cm/'+desc+'_cm'+'.png')
            plt.close()

        return score


    #################################
    # Fit PKNN to training data
    #################################
    def fit(self):

        # ## Do manual PCA to pick components
        # # Find num components
        # print(self.train_features_ravel.shape)
        # cov = np.cov(self.train_features_ravel.T)
        # print('Done cov')
        # cov = np.real(cov)
        # print('Done real')
        # eva, eve = np.linalg.eig(cov)
        # print('Done eig')
        # tup = [(i,eva[i]) for i in range(len(eva))]
        # print('Done tup')
        # s = sorted(tup, key=lambda x: x[0])
        # print('Done sort')
        # score_sum = 0
        # total = sum([si[1] for si in s])
        # print('Done total')
        # with open('results.csv','w') as csvfile:
        #     spamwriter = csv.writer(csvfile, delimiter=' ',
        #                     quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     for si in s:
        #         score_sum += si[1]
        #         #print(si[0],score_sum/total)
        #         spamwriter.writerow([si[0]] + [score_sum/total])
        # print('Done csv')
        # input()

        # Reduce by num components
        if(self.pca):
            self.do_pca = PCA(self.pca)
            self.do_pca.fit(self.train_features_ravel)
            self.train_features_ravel = self.do_pca.transform(self.train_features_ravel)

        ## Nearest Neighbor
        # Fit to nearest neighbor index
        self.nn.fit(self.train_features_ravel)
        
        ## Get distances for init
        init_bound = dict()
        self.mx = 0
        self.mn = 100000

        calc_init_vars = True
        if(calc_init_vars):
            # For each class, find dists from each training case to every other training case
            for cls in self.train_labels:
                class_dists = []

                # Compute dists to every other training case, add one K to exclude distance to self
                dists, results = self.nn.kneighbors(self.train_features_ravel, self.kfit+1)

                # Only add dists from own class
                for i,d in enumerate(dists):
                    if(self.train_labels_ravel[i] == cls):
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

        return init_bound


    #################################
    # Predict Possibilistic KNNif(i_true_pos[cls] != 0):
    def predict(self,test_set,bounds):
        # Perform PCA - init by fit()
        if(self.pca):
            test_set = self.do_pca.transform(test_set)

        # Get distances
        test_set = np.array(test_set,dtype=np.float32)
        dists, results = self.nn.kneighbors(test_set, self.k)

        # For each case, classify using mean/sd
        # Get class for each
        typicalities  = dict()
        typ_true_pos  = dict()
        typ_false_pos = dict()
        i_true_pos    = dict()
        i_false_pos   = dict()
        for cls in self.train_labels:
            cls_typ = []
            typ_true_pos[cls]  = 0
            typ_false_pos[cls] = 0
            i_true_pos[cls]    = 0
            i_false_pos[cls]   = 0
            for res,dis in zip(results,dists):
                typ = 0
                # Calculate Possibilistic score
                for r,d in zip(res,dis):
                    # Calc typicality
                    bound = bounds[cls]
                    typ_temp = (1/(1+np.power(np.maximum(0,d - bound),2)))

                    # Increase class typicality if class matches
                    if(self.train_labels_ravel[r] == cls and typ_temp >= .01):
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
        for cls in self.train_labels:
            if(i_true_pos[cls] != 0):
                typ_true_pos[cls] = typ_true_pos[cls] / i_true_pos[cls]
            if(i_false_pos[cls] != 0):
                typ_false_pos[cls] = typ_false_pos[cls] / i_false_pos[cls]

        return typicalities, typ_true_pos, typ_false_pos

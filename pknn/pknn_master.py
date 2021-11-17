import numpy as np
import sys
sys.path.append('./utils/')
from possi_sklearn import *

def pick_pknn(data='food',exp='known',dimred='none',pca=False,typ_threshold=.34):

    # Load Data
    if(data == 'food'):
        if(dimred == 'mp'):
            ora_train_features  = np.load('./export/features/orange_train_mp_resnet101.npy')
            ban_train_features  = np.load('./export/features/banana_train_mp_resnet101.npy')
            don_train_features  = np.load('./export/features/donut_train_mp_resnet101.npy')
        elif(dimred == 'none'):
            ora_train_features  = np.load('./export/features/orange_train_resnet101.npy')
            ban_train_features  = np.load('./export/features/banana_train_resnet101.npy')
            don_train_features  = np.load('./export/features/donut_train_resnet101.npy')

        train_labels = ['orange','banana']
        test_labels  = ['orange','banana']
        train_features = [
            ora_train_features[:100],
            ban_train_features[:100]
        ]
        valid_features = [
            ora_train_features[100:200],
            ban_train_features[100:200]
        ]

        if(exp=='known'):
            test_features = [
                ora_train_features[500:550],
                ban_train_features[500:550],
            ]
        if(exp=='nplusone'):
            test_features = [
                ora_train_features[500:550],
                ban_train_features[500:550],
                don_train_features[100:150]
            ]

    # Load Data
    if(data == 'animals'):
        if(dimred == 'mp'):
            mon_train_features  = np.load('./export/features/monkey_train_mp_resnet101.npy')
            tig_train_features  = np.load('./export/features/tiger_train_mp_resnet101.npy')
            che_train_features  = np.load('./export/features/cheetah_train_mp_resnet101.npy')
            hor_train_features  = np.load('./export/features/horse_test_mp_resnet101.npy')
        elif(dimred == 'none'):
            mon_train_features  = np.load('./export/features/monkey_train_resnet101.npy')
            tig_train_features  = np.load('./export/features/tiger_train_resnet101.npy')
            che_train_features  = np.load('./export/features/cheetah_train_resnet101.npy')
            hor_train_features  = np.load('./export/features/horse_test_resnet101.npy')

        train_labels = ['monkey','tiger','cheetah']
        test_labels  = ['monkey','tiger','cheetah']
        train_features = [
            mon_train_features[:100],
            tig_train_features[:100],
            che_train_features[:100]
        ]
        valid_features = [
            mon_train_features[100:200],
            tig_train_features[100:200],
            che_train_features[100:200]
        ]

        if(exp=='known'):
            test_features = [
                mon_train_features[500:550],
                tig_train_features[500:550],
                che_train_features[500:550],
            ]
        if(exp=='nplusone'):
            test_features = [
                mon_train_features[500:550],
                tig_train_features[500:550],
                che_train_features[500:550],
                hor_train_features
            ]

    # Experiment
    if(exp == 'known'):

        # Create PKNN, then fit
        
        pknn = PossiKNN(3,5, \
            train_features, \
            train_labels, \
            valid_features, \
            train_labels, \
            test_features, \
            test_labels, pca=pca)
        pknn.fit()
    
    # Experiment
    if(exp == 'nplusone'):

        # Create PKNN, then fit
        test_labels.append('none')
        pknn = PossiKNN(3,5, \
            train_features, \
            train_labels, \
            valid_features, \
            train_labels, \
            test_features, \
            test_labels, pca=pca, typ_threshold=typ_threshold)
        init_bound = pknn.fit()
        print(init_bound)

    return pknn
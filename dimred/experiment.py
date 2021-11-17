from pca import *
from tsne import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

tig_train = np.load('../data/tigers/tiger_train.npy')
mon_train = np.load('../data/monkeys/monkey_train.npy')
che_train = np.load('../data/cheetahs/cheetah_train.npy')
tig_test = np.load('../data/tigers/tiger_test.npy')
mon_test = np.load('../data/monkeys/monkey_test.npy')
che_test = np.load('../data/cheetahs/cheetah_test.npy')
hor = np.load('../data/horses/horses.npy')

experiment = 2
print('Running experiment #',experiment)

if(experiment == 1):
    dims        = [0, 100, 200, 300, 310]
    colors      = ['green','red','blue','orange','purple','yellow']
    components  = 2

    data = np.append(tig_train[:dims[1]],mon_train[:dims[1]],0)
    data = np.append(data,che_train[:dims[1]],0)
    data = np.append(data,hor,0)

    reductions = [pca(50), tsne(components)]
    #reductions = [pca(components)]

    print('Data loaded...',end='')
    print(data.shape)

    for red in reductions:
        data = red(data)
        print(red.name,'completed...',end='')
        print(data.shape)

    fig = plt.figure()

    if(components == 3):
        ax = plt.axes(projection='3d')
        for i,dim in enumerate(dims):
            if(i != 0):
                ax.scatter3D(data[dims[i-1]:dims[i],0], data[dims[i-1]:dims[i],1], data[dims[i-1]:dims[i],2], c=colors[i-1])
        plt.show()

    elif(components == 2):
        ax = plt.axes()
        for i,dim in enumerate(dims):
            if(i != 0):
                ax.scatter(data[dims[i-1]:dims[i],0], data[dims[i-1]:dims[i],1], c=colors[i-1])
        plt.show()

    print('Finished')

elif(experiment == 2):
    components  = 2

    dims = []

    data = [tig_train,mon_train,che_train,tig_test,mon_test,che_test,hor]

    ravel = []
    for d in data:
        dims.append(d.shape[0])
        ravel.extend(d.tolist())
    ravel = np.array(ravel)

    #reductions = [pca(50), tsne(components)]
    reductions = [pca(components)]

    print('Data loaded...',end='')
    print(ravel.shape)

    for red in reductions:
        ravel = red(ravel)
        print(red.name,'completed...',end='')
        print(ravel.shape)

    data_new = []

    for d in dims:
        data_new.append(ravel[:d].tolist())
        ravel = ravel[d:]
    data_new = np.array(data_new)

    print(data_new.shape)

    names = ['tig_train','mon_train','che_train','tig_test','mon_test','che_test','hor']

    for i,n in enumerate(names):
        np.save('../data/pca2/'+n+'_pca_2',data[i])
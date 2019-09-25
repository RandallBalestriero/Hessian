import os
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import tabulate
import matplotlib as mpl
#reds = mpl.cmap.get_cmap('Reds')
plt.style.use('ggplot')

label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


path = '/mnt/drive1/rbalSpace/Hessian/actnoise8_{}_{}_{}_{}_{}_{}.h5'
N = 10000
DATASET = 'cifar10'
DATA_AUGMENTATION = 0
run = 0
MODEL = 'simpleresnet'
LR = 0.0001
DATA = list()

for run in [0]:
    for GAMMA in [0., 0.001, 0.01, 0.1]:
        filename = path.format(MODEL, DATASET, N, str(GAMMA),
                               LR, run)
        f = h5py.File(filename, 'r', swmr=True)

        arg = np.argmax(f['valid_set/accu'][...])
        DATA.append(f['train_set/hessian'][...].mean(1))
        DATA.append(f['valid_set/hessian'][...].mean(1))
        DATA.append(f['train_set/accu'][...] * 100)
        DATA.append(f['test_set/accu'][...] * 100)
        f.close()

for k, name in enumerate(['hessian_train_resnet.pdf', 'hessian_test_resnet.pdf',
                          'accu_train_resnet.pdf', 'accu_test_resnet.pdf']):
    plt.figure(figsize=(7, 3))
    for i, (c, gamma) in enumerate(zip(['k', 'b', 'c', 'r'],
                                       [0., 0.001, 0.01, 0.1])):
        plt.plot(DATA[i*4+k], c=c, label=str(gamma))
    plt.legend(fontsize=12, loc='lower right')
    if 'accu' in name:
        plt.ylim([70, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

MODEL = 'cnn'
LR = 0.0001
DATA = list()

for run in [0]:
    for GAMMA in [0., 0.00001, 0.0001, 0.001]:
        filename = path.format(MODEL, DATASET, N, str(GAMMA),
                               LR, run)
        f = h5py.File(filename, 'r', swmr=True)

        arg = np.argmax(f['valid_set/accu'][...])
        DATA.append(f['train_set/hessian'][...].mean(1))
        DATA.append(f['valid_set/hessian'][...].mean(1))
        DATA.append(f['train_set/accu'][...] * 100)
        DATA.append(f['test_set/accu'][...] * 100)
        f.close()

for k, name in enumerate(['hessian_train_cnn.pdf', 'hessian_test_cnn.pdf',
                          'accu_train_cnn.pdf', 'accu_test_cnn.pdf']):
    plt.figure(figsize=(7,3))
    for i, (c, gamma) in enumerate(zip(['k', 'b', 'c', 'r'],
                                       [0., 0.00001, 0.0001, 0.001])):
        plt.plot(DATA[i*4+k], c=c, label=str(gamma))
    plt.legend(fontsize=12, loc='lower right')
    if 'accu' in name:
        plt.ylim([70, plt.ylim()[1]])                            
    plt.tight_layout()
    plt.savefig(name)
    plt.close()




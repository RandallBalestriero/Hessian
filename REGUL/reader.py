import os
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import tabulate
import matplotlib as mpl
#reds = mpl.cmap.get_cmap('Reds')


path = '/mnt/drive1/rbalSpace/Hessian/actnoise_{}_{}_{}_{}_{}_{}.h5'
N = 10000
DATASET = 'cifar10'
DATA_AUGMENTATION = 0
run = 0
MODEL = 'simpleresnet'
LR = 0.0001
DATA = list()
plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)


plt.figure()
ax5 = plt.subplot(231)
ax6 = plt.subplot(232)
ax7 = plt.subplot(233)
ax8 = plt.subplot(234)
ax9 = plt.subplot(235)
ax10 = plt.subplot(236)



for run in [0]:
#    for ls, EPSILON in zip(['--', '-', '-.'], [0.1, 0.001, 0.0001]):
    for c, GAMMA in zip(['k', 'r', 'm'], [0., 0.001, 0.1]):
        filename = path.format(MODEL, DATASET, N, str(GAMMA),
                               LR, run)
        if not os.path.isfile(filename):
            print('NOT A FILE :',filename)
            continue
        f = h5py.File(filename,'r',swmr=True)

        arg = np.argmax(f['valid_set/accu'][...])
        ax1.plot(f['train_set/hessian'][...].mean(1), c=c)
        ax2.plot(f['valid_set/hessian'][...].mean(1), c=c)
        ax3.plot(f['train_set/accu'][...], c=c, label=str(GAMMA))
        ax4.plot(f['test_set/accu'][...], c=c)
        print([i for i in f['interp'].keys()])
        vv = f['interp/inter'][...][-1][:, :32].reshape((-1, 10))

        print(f['interp/inter'][...][-1].shape)
        p1 = 0
        p2 = 0
        if c == 'k':
            a1 = ax5
            a2 = ax8
        elif c == 'r':
            a1 = ax6
            a2 = ax9
        elif c == 'm':
            a1 = ax7
            a2 = ax10
        for cla in range(10):
            a1.plot(vv[p1*100:(p1+1)*100, cla], c=c)
            a2.plot(vv[p2*100+2000:(p2+1)*100+2000, cla], c=c)

#        print('eps:', EPSILON, 'gamma:',GAMMA, f['test_set/accu'][...][arg], len(f['test_set/accu'][...]))
        #DATA.append(f['test_set/hessian'][...][arg])
        f.close()

ax1.set_title('Hessian on train set')
ax3.set_title('Train accuracy')
ax4.set_title('Test accuracy')
ax3.legend()
plt.show()

#print(tabulate.tabulate(np.array(DATA).reshape((2, 3, 4)).mean(0),
#                        headers=["gamma=0.", "gamma=1.", "gamma=0.01",
#                                 "gamma=0.0001"]))



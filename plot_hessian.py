import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py



def plotit(NAME):
    print(NAME)
    filename = '/mnt/drive1/rbalSpace/HESSIAN/'+NAME+'.h5'
    if not os.path.isfile(filename):
        print('NOT A FILE :',filename)
        return
    f = h5py.File(filename,'r',swmr=True)

    loss = f['train_set/loss/1'][...].flatten()
    train_labels = f['train_set/loss/3'][...].flatten()
    train_A = f['train_set/loss/2'][...]
    print(train_A.shape)
    TRAIN_A = np.log(1e-7+train_A.mean((-3,-2,-1)))
    TRAIN_a = train_A.mean((-3,-2,-1))
    TRAIN_A-=TRAIN_A.min()
    TRAIN_A/=TRAIN_A.max()
    train_A = train_A.reshape((-1,10))

    accu = f['test_set/accu/0'][...].flatten()
    test_labels = f['test_set/accu/2'][...].flatten()
    test_A = f['test_set/accu/1'][...]
    TEST_A = np.log(1e-7+test_A.mean((-3,-2,-1)))
    TEST_a = test_A.mean((-3,-2,-1))
    TEST_A-=TEST_A.min()
    TEST_A/=TEST_A.max()
    test_A = test_A.reshape((-1,10))

    plt.figure(figsize=(6,3))
    plt.plot(TRAIN_A/TEST_A.max(),'b')
    plt.plot(TEST_A/TEST_A.max(),'k')
    plt.plot(accu,'--k')
    plt.tight_layout()
    plt.savefig('images/'+NAME+'_comparison.pdf')
    plt.close()
    print('SCORES:',TRAIN_a[-1],TEST_a[-1],accu[-1])



    plt.figure(figsize=(10,6))
    plt.subplot(321)
    plt.plot(loss)
    plt.title('Train set, every 100 batch')
    plt.xticks([])
    plt.ylabel('cross entropy')

    plt.subplot(323)
    rows = range(len(train_labels))
    plt.semilogy(train_A[rows,train_labels],alpha=0.3)
    plt.semilogy((train_A.sum(1)-train_A[rows,train_labels])/9,
                                                         alpha=0.3)
    plt.title('Correct and Wrong class rows')
    plt.xticks([])

    plt.subplot(325)
    for k in range(10):
        plt.semilogy(train_A[:,k],alpha=0.5)
    plt.title('Individual rows')
    plt.suptitle(NAME)

    plt.subplot(322)
    plt.plot(accu*100)
    plt.title('Test set, every epoch')
    plt.xticks([])
    plt.ylabel(r'accuracy in $\%$')

    plt.subplot(324)
    rows = range(len(test_labels))
    plt.semilogy(test_A[rows,test_labels], alpha=0.3)
    plt.semilogy((test_A.sum(1)-test_A[rows,test_labels])/9,
                                                        alpha=0.3)
    plt.title('Correct and Wrong class rows')
    plt.legend(['True class','Other classes'],loc='upper left')
    plt.xticks([])

    plt.subplot(326)
    for k in range(10):
        plt.semilogy(test_A[:,k],alpha=0.5)
    plt.title('Individual rows')
    plt.suptitle(NAME)
    plt.savefig('images/'+NAME+'_summary.pdf')
    plt.close()
    f.close()



for dataset in ['mnist','cifar10']:
    for model in ['cnn','resnetsmall','resnetlarge']:
        for data_augmentation in [0,1]:
            NAME = 'hessian_{}_{}_{}'.format(dataset,
                                            model, data_augmentation)
            plotit(NAME)




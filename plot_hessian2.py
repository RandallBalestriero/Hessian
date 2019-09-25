import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
plt.style.use('ggplot')
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

cmap = plt.get_cmap('Reds')


def getit(name):
    filename = '/mnt/drive1/rbalSpace/HESSIAN/'+name+'.h5'
    f = h5py.File(filename, 'r', swmr=True)

    loss = f['train_set/loss/1'][...].flatten()
    train_A = f['train_set/loss/2'][...].flatten()

    test_accu = f['test_set/accu/0'][...].flatten()
    test_A = f['test_set/accu/1'][...].mean(1).flatten()

    valid_accu = f['valid_set/accu/0'][...]

    f.close()

    return loss, train_A/100, test_accu, test_A/100, valid_accu



for model in ['cnn', 'resnetlarge']:
    loss, train_A, test_accu, test_A, valid_accu = getit('hessian_cifar100_{}_0_0.0'.format(model))
    loss2, train_A2, test_accu2, test_A2, valid_accu2 = getit('hessian_cifar100_{}_1_0.0'.format(model))

    plt.figure(figsize=(6,3))
    plt.plot(loss, 'k')
    plt.plot(loss2, 'b')
    plt.tight_layout()
    plt.savefig('images2/{}_loss_comparison.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(test_accu*100, 'k')
    plt.plot(test_accu2*100, 'b')
    plt.tight_layout()
    plt.savefig('images2/{}_accu_comparison.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(train_A, 'k')
    plt.plot(train_A2, 'b')
    plt.tight_layout()
    plt.savefig('images2/{}_trainA_comparison.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(test_A, 'k')
    plt.plot(test_A2, 'b')
    plt.tight_layout()
    plt.savefig('images2/{}_testA_comparison.pdf'.format(model))
    plt.close()

    print(model, ' SCORES:', train_A[-30:].mean(),train_A2[-30:].mean())
    print(test_A[-1],test_A2[-1])
    print(test_accu[valid_accu.argmax()],test_accu2[valid_accu2.argmax()])

exit()

for model in ['cnn', 'resnetlarge']:
    loss, train_A, test_accu, test_A, valid_accu = getit('hessian_cifar10_{}_0_0.0'.format(model))
    loss2, train_A2, test_accu2, test_A2, valid_accu2 = getit('hessian_cifar10_{}_0_0.01'.format(model))
    loss3, train_A3, test_accu3, test_A3, valid_accu3 = getit('hessian_cifar10_{}_0_0.1'.format(model))
#    loss4, train_A4, test_accu4, test_A4, valid_accu4 = getit('hessian_cifar10_{}_0_1.0'.format(model))


    plt.figure(figsize=(6,3))
    plt.plot(loss, 'k')
    plt.plot(loss2, color=cmap(0.3))
    plt.plot(loss3, color=cmap(0.6))
#    plt.plot(loss4, cmap(0.9))
    plt.tight_layout()
    plt.savefig('images2/{}_loss_regularization.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(test_accu*100, 'k')
    plt.plot(test_accu2*100, color=cmap(0.3))
    plt.plot(test_accu3*100, color=cmap(0.6))
#    plt.plot(test_accu4*100, cmap(0.9))

    plt.tight_layout()
    plt.savefig('images2/{}_accu_regularization.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(train_A, 'k')
    plt.plot(train_A2, color=cmap(0.3))
    plt.plot(train_A3, color=cmap(0.6))
#    plt.plot(train_A4, cmap(0.9))
    plt.tight_layout()
    plt.savefig('images2/{}_trainA_regularization.pdf'.format(model))
    plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(test_A, 'k')
    plt.plot(test_A2, color=cmap(0.3))
    plt.plot(test_A3, color=cmap(0.6))
#    plt.plot(test_A4, cmap(0.9))

    plt.tight_layout()
    plt.savefig('images2/{}_testA_regularization.pdf'.format(model))
    plt.close()

    print(model, ' SCORES:', train_A[-30:].mean(),train_A2[-30:].mean(),train_A3[-30:].mean())#,train_A4[-30:].mean())
    print(test_A[-1],test_A2[-1],test_A3[-1])#,test_A4[-1])
    print(test_accu[valid_accu.argmax()],test_accu2[valid_accu2.argmax()],test_accu3[valid_accu3.argmax()])#,test_accu4[valid_accu4.argmax()])









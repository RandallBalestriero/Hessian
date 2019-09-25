import sys
sys.path.insert(0, "../../Sknet/")

import sknet
import os
import numpy as np
import time
import tensorflow as tf
from sknet import ops,layers
import argparse


parser = argparse.ArgumentParser()
#parser.add_argument('--data_augmentation', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
#parser.add_argument('--epsilon', type=float)
parser.add_argument('-n', type=int)
parser.add_argument('--gamma', type=float)
parser.add_argument('--lr', type=float)
args = parser.parse_args()

#DATA_AUGMENTATION = args.data_augmentation
#EPSILON = args.epsilon
DATASET = args.dataset
MODEL = args.model
GAMMA = args.gamma
N = args.n
LR = args.lr
# Data Loading
#-------------

if DATASET=='cifar10':
    dataset = sknet.datasets.load_cifar10()
elif DATASET=='mnist':
    dataset = sknet.datasets.load_mnist()
elif DATASET=='svhn':
    dataset = sknet.datasets.load_svhn()
elif DATASET=='cifar100':
    dataset = sknet.datasets.load_cifar100()

if "valid_set" not in dataset.sets:
    dataset.split_set("train_set","valid_set",0.15)

preprocess = sknet.datasets.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = preprocess.transform(dataset['images/train_set'])
dataset['images/test_set'] = preprocess.transform(dataset['images/test_set'])
dataset['images/valid_set'] = preprocess.transform(dataset['images/valid_set'])
time = np.linspace(0, 1, 100).reshape((-1, 1, 1, 1))
dataset['images/interpolation'] = np.concatenate([np.expand_dims(dataset['images/train_set'][i], 0)*time+\
                                   np.expand_dims(dataset['images/train_set'][i+1], 0)*(1-time) for i in range(20)] +\
                                  [np.expand_dims(dataset['images/test_set'][i], 0)*time+\
                                   np.expand_dims(dataset['images/test_set'][i+1], 0)*(1-time) for i in range(20)], 0).astype('float32')


#perm = np.random.permutation(N)
#dataset['images/train_set'] = dataset['images/train_set'][perm]
#dataset['labels/train_set'] = dataset['labels/train_set'][perm]


options = {'train_set': "random_see_all",
           'valid_set': 'continuous',
           'interpolation' : 'continuous',
           'test_set': 'continuous'}

dataset.create_placeholders(32, options, device="/cpu:0")
const = 1.#(2*EPSILON)**(1./2)
# Create Network
#---------------
dnn = sknet.Network()
#if DATA_AUGMENTATION:
start = 1
#    dnn.append(sknet.ops.RandomAxisReverse(dataset.images, axis=[-1]))
if DATASET == 'fashion':
    dnn.append(sknet.ops.RandomCrop(dataset.images, (28, 28), pad=(4, 4), seed=10))
elif DATASET in ['cifar10', 'cifar100', 'svhn']:
    dnn.append(sknet.ops.RandomCrop(dataset.images, (32, 32), pad=(4, 4), seed=10))
#else:
#    dnn.append(dataset.images)
#    start = 1


if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn)
elif MODEL == 'simpleresnet':
    sknet.networks.Resnet(dnn, D=4, W=1, block=sknet.layers.ResBlockV2)
elif MODEL == 'resnet':
    sknet.networks.Resnet(dnn, D=10, W=1, block=sknet.layers.ResBlockV2)
elif MODEL == 'wideresnet':
    sknet.networks.Resnet(dnn, D=6, W=2, block=sknet.layers.ResBlockV2)

dnn.append(sknet.ops.Dense(dnn[-1], dataset.n_classes))
# accuracy and loss


prediction = dnn[-1]

accu = sknet.losses.streaming_mean(sknet.losses.accuracy(dataset.labels,
                                                         dnn[-1]))
loss = sknet.losses.crossentropy_logits(dataset.labels, dnn[-1])

# optimizer and updates
B = dataset.N_BATCH('train_set')
lr = sknet.schedules.PiecewiseConstant(LR, {70*B: LR/3, 120*B: LR/9})
optimizer = sknet.optimizers.Adam(loss,
                                  dnn.variables(trainable=True), lr)
minimizer = tf.group(*optimizer.updates, *dnn.updates)
reset = tf.group(optimizer.reset_variables_op, dnn.reset_variables_op)

# Workers
train = sknet.Worker(minimizer, loss=loss, accu=accu,
                     context='train_set', to_print=loss,
                     feed_dict=dnn.deter_dict(False))
test = sknet.Worker(loss=loss, accu=accu,
                    context='test_set', to_print=accu,
                    feed_dict=dnn.deter_dict(True))


# Pipeline
workplace = sknet.utils.Workplace(dataset=dataset)
path = '/mnt/drive1/rbalSpace/Hessian/naked_{}_{}_{}_{}.h5'
#path = '/mnt/project2/rb42Data/BatchNorm/pretrain_{}_{}_{}_{}_{}.h5'
#for run in range(5):
workplace.init_file(path.format(MODEL, DATASET, N, LR))
#    workplace.execute_worker(inter)
workplace.execute_worker((train, test), repeat=150)
#    workplace.execute_worker(inter)
#    workplace.session.run(reset)




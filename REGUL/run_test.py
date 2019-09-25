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
parser.add_argument('--data_augmentation', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--epsilon', type=float)
parser.add_argument('-n', type=int)
parser.add_argument('--gamma', type=float)
parser.add_argument('--lr', type=float)
args = parser.parse_args()

DATA_AUGMENTATION = args.data_augmentation
EPSILON = args.epsilon
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

dataset['indicator/train_set'] = np.concatenate([np.ones(len(dataset['images/train_set'])),
                                                 np.zeros(len(dataset['images/test_set']))])
dataset['indicator/test_set'] = np.zeros(4000)
dataset['images/train_set'] = np.concatenate([dataset['images/train_set'],
                                              dataset['images/test_set']],0)
dataset['labels/train_set'] = np.concatenate([dataset['labels/train_set'],
                                                dataset['labels/test_set']],0)

if "valid_set" not in dataset.sets:
    dataset.split_set("train_set","valid_set",0.15)

preprocess = sknet.datasets.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = preprocess.transform(dataset['images/train_set'])
dataset['images/test_set'] = preprocess.transform(dataset['images/test_set'])
dataset['images/valid_set'] = preprocess.transform(dataset['images/valid_set'])


options = {'train_set': "random_see_all",
           'valid_set': 'continuous',
           'test_set': 'continuous'}

dataset.create_placeholders(32, options, device="/cpu:0")
const = (2*EPSILON)**(1./2)
# Create Network
#---------------
dnn = sknet.Network()
if DATA_AUGMENTATION:
    start = 2
    dnn.append(sknet.ops.RandomAxisReverse(dataset.images, axis=[-1]))
    if DATASET == 'fashion':
        dnn.append(sknet.ops.RandomCrop(dnn[-1], (28, 28), pad=(6, 6), seed=10))
    elif DATASET in ['cifar10', 'cifar100', 'svhn']:
        dnn.append(sknet.ops.RandomCrop(dnn[-1], (32, 32), pad=(8, 8), seed=10))
else:
    dnn.append(dataset.images)
    start = 1

noise = tf.nn.l2_normalize(tf.random_normal(dnn[-1].get_shape().as_list()),
                           (1, 2, 3))*EPSILON
dnn.append(ops.Concat([dnn[-1],dnn[-1]+noise],axis=0))


if MODEL == 'cnn':
    sknet.networks.ConvLarge(dnn, noise=NOISE)
elif MODEL == 'simpleresnet':
    sknet.networks.Resnet(dnn, D=4, W=1, block=sknet.layers.ResBlockV2)
elif MODEL == 'resnet':
    sknet.networks.Resnet(dnn, D=10, W=1, block=sknet.layers.ResBlockV2)
elif MODEL == 'wideresnet':
    sknet.networks.Resnet(dnn, D=6, W=2, block=sknet.layers.ResBlockV2)

dnn.append(sknet.ops.Dense(dnn[-1], dataset.n_classes))
# accuracy and loss
vvv = tf.reshape(tf.cast(dataset.indicator, tf.float32), (-1, 1, 1, 1))
def compute_row(i):
    onehot = tf.ones((64 ,1))*tf.expand_dims(tf.one_hot(i, dataset.n_classes), 0)
    grad = tf.gradients(dnn[-1], dnn[start], onehot)[0]
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((1-vvv)*tf.square((grad[:32]-grad[32:])/EPSILON),
                          [1, 2, 3])+0.0001))

prediction = dnn[-1]
hessian = tf.sqrt(tf.reduce_sum(tf.map_fn(compute_row, tf.range(dataset.n_classes),
                                dtype=tf.float32))+0.0001)
accu = sknet.losses.streaming_mean(sknet.losses.accuracy(dataset.labels,
                                                         dnn[-1][:32]))
vvv = tf.cast(tf.reshape(dataset.indicator, (-1, 1)), tf.float32)
loss = sknet.losses.crossentropy_logits(dataset.labels, vvv*dnn[-1][:32]) +\
                        GAMMA*hessian

# optimizer and updates
B = dataset.N_BATCH('train_set')
lr = sknet.schedules.PiecewiseConstant(LR, {70*B: LR/3, 120*B: LR/9})
optimizer = sknet.optimizers.Adam(loss, dnn.variables(trainable=True), lr)
minimizer = tf.group(*optimizer.updates, *dnn.updates)
reset = tf.group(optimizer.reset_variables_op, dnn.reset_variables_op)

# Workers
train = sknet.Worker(minimizer, loss=loss, accu=accu, hessian=hessian,
                     context='train_set', to_print=loss,
                     feed_dict=dnn.deter_dict(False))
test = sknet.Worker(loss=loss, accu=accu, hessian=hessian,
                    context='test_set', to_print=accu,
                    feed_dict=dnn.deter_dict(True))
valid = sknet.Worker(loss=loss, accu=accu, hessian=hessian,
                     context='valid_set', to_print=accu,
                     feed_dict=dnn.deter_dict(True))

# Pipeline
workplace = sknet.utils.Workplace(dataset=dataset)
path = '/mnt/drive1/rbalSpace/Hessian/acttest_{}_{}_{}_{}_{}_{}_{}_{}.h5'
#path = '/mnt/project2/rb42Data/BatchNorm/pretrain_{}_{}_{}_{}_{}.h5'
for run in range(5):
    workplace.init_file(path.format(MODEL, DATASET, EPSILON,
                                    DATA_AUGMENTATION, N, GAMMA, LR, run))
    workplace.execute_worker((train, valid, test), repeat=150)
    workplace.session.run(reset)
dnn = sknet.Network()



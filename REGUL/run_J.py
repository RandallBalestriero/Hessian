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
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--augmentation', type=int)
args = parser.parse_args()

DATASET = args.dataset
MODEL = args.model
LR = args.lr
AUGMENTATION = args.augmentation

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


options = {'train_set': "random_see_all",
           'valid_set': 'continuous',
           'test_set': 'continuous'}

dataset.create_placeholders(32, options, device="/cpu:0")
dnn = sknet.Network()
dnn.append(sknet.ops.RandomAxisReverse(dataset.images, axis=[-1]))
if DATASET == 'fashion':
    dnn.append(sknet.ops.RandomCrop(dataset.images, (28, 28), pad=(8, 8), seed=10))
elif DATASET in ['cifar10', 'cifar100', 'svhn']:
    dnn.append(sknet.ops.RandomCrop(dataset.images, (32, 32), pad=(8, 8), seed=10))

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

def compute_row(i):
    onehot = tf.ones((32 ,1))*tf.expand_dims(tf.one_hot(i, dataset.n_classes), 0)
    grad = tf.gradients(dnn[-1], dnn[1], onehot)[0]
    return tf.reduce_sum(tf.square(grad), [1, 2, 3])


prediction = dnn[-1]
hessian = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.map_fn(compute_row,
                       tf.range(dataset.n_classes), dtype=tf.float32), 0)))

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
trainh = sknet.Worker(hessian=hessian,context='train_set', to_print=loss,
                     feed_dict=dnn.deter_dict(True), name='trainh')
testh = sknet.Worker(hessian=hessian, context='test_set',
                    feed_dict=dnn.deter_dict(True), name='testh')



# Pipeline
workplace = sknet.utils.Workplace(dataset=dataset)
path = '/mnt/drive1/rbalSpace/Hessian/jacob_{}_{}_{}_{}.h5'
workplace.init_file(path.format(MODEL, DATASET, AUGMENTATION, LR))
workplace.execute_worker((trainh, testh), repeat=1)
workplace.execute_worker((train, test), repeat=150)
workplace.execute_worker((trainh, testh), repeat=1)




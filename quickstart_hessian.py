import sys
sys.path.insert(0, "../Sknet/")

import sknet
from sknet.optimizers import Adam
from sknet.losses import StreamingAccuracy,crossentropy_logits
from sknet.schedules import PiecewiseConstant

import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers
import argparse

if os.path.isfile('.sknetrc'):
    exec(open(".sknetrc").read())
else:
    SAVE_PATH = os.environ['SAVE_PATH']

parser = argparse.ArgumentParser()
parser.add_argument('--data_augmentation', type=int, default=0,choices=[0,1])
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='cnn',
                            choices=['cnn','resnetsmall','resnetlarge'])
parser.add_argument('--regularization', type=float)
args = parser.parse_args()

DATA_AUGMENTATION = args.data_augmentation
REGULARIZATION = args.regularization
DATASET = args.dataset
MODEL = args.model
if MODEL=='resnetsmall':
    D = 1
    W = 1
elif MODEL=='resnetlarge':
    D = 2
    W = 1


# Data Loading
#-------------

if DATASET=='cifar10':
    dataset = sknet.dataset.load_cifar10()
elif DATASET=='mnist':
    dataset = sknet.dataset.load_mnist()
elif DATASET=='svhn':
    dataset = sknet.dataset.load_svhn()
elif DATASET=='cifar100':
    dataset = sknet.dataset.load_cifar100()

if "valid_set" not in dataset.sets:
    dataset.split_set("train_set","valid_set",0.15)

preprocess = sknet.dataset.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = preprocess.transform(dataset['images/train_set'])
dataset['images/test_set'] = preprocess.transform(dataset['images/test_set'])
dataset['images/valid_set'] = preprocess.transform(dataset['images/valid_set'])

iterator = BatchIterator(16, {'train_set': "random_see_all",
                         'valid_set': 'continuous',
                         'test_set': 'continuous'})

dataset.create_placeholders(iterator, device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.Network(name='simple_model')

if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
    crop_shape = dataset.datum_shape('images')[1:]
    crop_shape = (crop_shape[0]-4,crop_shape[1]-4)
    dnn.append(ops.RandomCrop(dnn[-1], crop_shape, seed=10))
    start = 2
else:
    dnn.append(dataset.images)
    start = 1

noise = tf.random_normal(dnn[-1].get_shape().as_list())*0.0001

dnn.append(ops.Concat([dnn[-1],dnn[-1]+noise],axis=0))

if MODEL=='cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
else:
    sknet.networks.Resnet(dnn, dataset.n_classes, D=D, W=W)


# Quantities
#-----------

def compute_row(i):
    selected_row = tf.ones((32,1))*tf.one_hot(i,dataset.n_classes)
    grad = tf.gradients(dnn[-1], dnn[start], selected_row)[0]
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad[:16]-grad[16:]),
                          [1,2,3])))

prediction = dnn[-1]
loss = sknet.losses.crossentropy_logits(p=dataset.labels,q=prediction[:16])
hessian = tf.reduce_sum(tf.map_fn(compute_row, tf.range(dataset.n_classes),
                         dtype=tf.float32))
accu = StreamingAccuracy(dataset.labels,prediction[:16])


B = dataset.N('train_set')//16
lr = PiecewiseConstant(0.005, {100*B:0.0015,150*B:0.001})

if REGULARIZATION:
    loss_extra = REGULARIZATION*hessian
    optimizer = sknet.optimizers.Adam(loss+loss_extra,
                                      dnn.variables(trainable=True), lr)
else:
    optimizer = sknet.optimizers.Adam(loss, dnn.variables(trainable=True), lr)

minimizer = tf.group(optimizer.updates+dnn.updates)


# Workers
#---------

minimize = sknet.Worker(name='loss',context='train_set',
            op=[minimizer,loss, hessian],
            deterministic=False,period=[1,100,100], verbose=[0,2,2])

accuv = sknet.Worker(name='accu',context='valid_set', op=[accu],
            deterministic=True, verbose=1)

accut = sknet.Worker(name='accu',context='test_set',
            op=[accu, hessian], deterministic=True,
            verbose=[1,0])

queue = sknet.Queue((minimize, accuv, accut), filename=SAVE_PATH+'/HESSIAN/'\
                +'hessian_{}_{}_{}_{}.h5'.format(DATASET,MODEL,
                DATA_AUGMENTATION,REGULARIZATION))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=200)



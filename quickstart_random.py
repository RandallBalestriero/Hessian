import sys
sys.path.insert(0, "../Sknet/")

import sknet
from sknet.optimizers import Adam
from sknet.losses import accuracy,crossentropy_logits
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
parser.add_argument('--proportion', type=float, default=0)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='cnn',
                            choices=['cnn','resnetsmall','resnetlarge'])
args = parser.parse_args()

PROPORTION = args.proportion
DATASET = args.dataset
MODEL = args.model
if MODEL=='resnetsmall':
    D = 1
    W = 1
elif MODEL=='resnetlarge':
    D = 4
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

if "test_set" not in dataset.sets:
    dataset.split_set("train_set","test_set",0.25)
if "valid_set" not in dataset.sets:
    dataset.split_set("train_set","valid_set",0.15)

preprocess = sknet.dataset.Standardize().fit(dataset['images/train_set'])
dataset['images/train_set'] = preprocess.transform(dataset['images/train_set'])
dataset['images/test_set']  = preprocess.transform(dataset['images/test_set'])
dataset['images/valid_set'] = preprocess.transform(dataset['images/valid_set'])

# Random perturbation of the labels
dataset['true_labels/train_set'] = np.copy(dataset['labels/train_set'])
N = dataset.N('train_set')
n = int(N*PROPORTION)
random_labels = np.random.permutation(N)[:n]
random_labels = dataset['labels/train_set'][random_labels]
random_data = np.random.permutation(N)[:n]
dataset['labels/train_set'][random_data] = random_labels

dataset.create_placeholders(batch_size=32,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                       'valid_set':BatchIterator('continuous'),
                       'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.Network(name='simple_model')

dnn.append(dataset.images)
start = 1

noise = tf.random_normal(dnn[-1].get_shape().as_list())*0.00001
dnn.append(ops.Concat([dnn[-1],dnn[-1]+noise],axis=0))

if MODEL=='cnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
else:
    sknet.networks.Resnet(dnn, dataset.n_classes, D=D, W=W)


# Quantities
#-----------

def compute_row(i):
    selected_row = tf.ones((64,1))*tf.one_hot(i,dataset.n_classes)
    grad = tf.gradients(dnn[-1], dnn[start], selected_row)[0]
    return tf.sqrt(tf.reduce_sum(tf.square(grad[:32]-grad[32:]), [1,2,3]))

A_rows = tf.map_fn(compute_row, tf.range(dataset.n_classes), dtype=tf.float32)
prediction = dnn[-1]
loss = crossentropy_logits(p=dataset.labels,q=prediction[:32])
hessian = tf.transpose(A_rows) #(32,n_classes)
accu = accuracy(dataset.labels,prediction[:32])

B = dataset.N_BATCH('train_set')
lr = PiecewiseConstant(0.005, {100*B:0.0015,150*B:0.001})
optimizer = Adam(loss,lr,params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)


# Workers
#---------

minimize = sknet.Worker(name='loss',context='train_set', op=[minimizer,loss,
            hessian, dataset.labels,dataset.true_labels, accu],
            deterministic=False,period=[1,100,100,100,100,1],
            verbose=[0,2,0,0,0,1],
            transform_function=[None, None, None, None, None, np.mean])

accuv = sknet.Worker(name='accu',context='valid_set', op=[accu],
            deterministic=True, transform_function=[np.mean], verbose=1)

accut = sknet.Worker(name='accu',context='test_set',
            op=[accu, hessian, dataset.labels], deterministic=True,
            transform_function=[np.mean,None,None], verbose=[1,0,0])

queue = sknet.Queue((minimize, accuv, accut), filename=SAVE_PATH+'/HESSIAN/'\
                      +'random_{}_{}_{}.h5'.format(DATASET, MODEL, PROPORTION))

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_queue(queue,repeat=200)

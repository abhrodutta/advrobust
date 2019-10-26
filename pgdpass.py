""" alpha = 0.8, epsilon = 0.04, non-robust network examples where madry found adv examples """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt

import json
import math
import sys
import random

import tensorflow as tf
import numpy as np
import mosek
from math import *
from mosek.fusion import *

def firstDiff(arr):
    return np.max(arr)-arr[secondArgMax(arr)]

def secondArgMax(arr):
    x = sorted(enumerate(arr),key = lambda y: -y[1])
    return x[1][0]

def relu(blah):
    return np.array(list(map(lambda x: np.array(list(map(lambda y :max(y,0),x))),blah)))

if __name__ == '__main__':

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator
    W = sess.run(model.W) #, feed_dict={model.x_input: mnist.test.images, model.y_input: mnist.test.labels})
    V = sess.run(model.V) #  feed_dict={model.x_input: mnist.test.images, model.y_input: mnist.test.labels})

    path = 'attack.npy'

    #x_adv = np.concatenate(x_adv, axis=0)
    x_adv = np.load(path)
    #print(x_adv)
    x_adv = np.transpose(x_adv)
    W = np.transpose(W)
    V = np.transpose(V)

    advlayer1 = relu(np.dot(W,x_adv))
    #print((layer1 >= 0).all())
    #print(advlayer1.shape,"advlayer1")
    advlayer2 = np.dot(V,advlayer1)
    advlayer2 = np.transpose(advlayer2)

    advprediction = np.array([np.argmax(i) for i in advlayer2])


    #print(advprediction)
    print("Adversarial accuracy : ",100*np.sum(advprediction == mnist.test.labels)/10000.0,"%")

    #Natural Accuracy
    x_nat = np.transpose(mnist.test.images)
    natlayer1 = relu(np.dot(W,x_nat))
    natlayer2 = np.dot(V,natlayer1)
    natlayer2 = np.transpose(natlayer2)
    natprediction = np.array([np.argmax(i) for i in natlayer2])
    print("Natural accuracy : ",100*np.sum(natprediction == mnist.test.labels)/10000.0,"%")


    x_found = list(filter(lambda x : advprediction[x]!= natprediction[x] and natprediction[x]== mnist.test.labels[x]  , range(10000)))
    x_found.sort(key = lambda y : firstDiff(natlayer2[y]))
    #x_found = x_found[:100]

    x_found_labels = [secondArgMax(natlayer2[i]) for i in x_found]
    output = np.array([natlayer2[i] for i in x_found])

    savepath = 'pgdpass_indicesall.npy'#corrected for natural mistakes
    savepath2 = 'pgdpass_labelsall.npy'
    savepath3 = 'pgdpass_outputall.npy'

    np.save(savepath, x_found)
    np.save(savepath2, x_found_labels)
    np.save(savepath3, output)

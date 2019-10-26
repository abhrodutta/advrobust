
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

# def relu(blah):
#     x = np.zeros(blah.shape)
#     print(blah.shape[0])
#     for i in range(blah.shape[0]):
#         for j in range(blah.shape[1]):
#             x[i][j] = max(0,blah[i][j])

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
  # attack = NaiveSDPAttack(model,
  #                        config['epsilon'],
  #                        config['k'],
  #                        config['a'],
  #                        config['random_start'],
  #                        config['loss_func'])
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
    V = sess.run(model.V, feed_dict={model.x_input: mnist.test.images, model.y_input: mnist.test.labels})
    #print('Iterating over {} batches'.format(num_batches))

    # for ibatch in range(num_batches):
    #   bstart = ibatch * eval_batch_size
    #   bend = min(bstart + eval_batch_size, num_eval_examples)
    #   print('batch size: {}'.format(bend - bstart))
    #
    #   #x_batch = mnist.test.images[bstart:bend, :]
    #   #y_batch = mnist.test.labels[bstart:bend]
    #
    #   #x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    #
    #   x_adv.append(x_batch_adv)

    #print('Storing examples')
    #path = config['store_adv_path']

    #change here for different input files
    #path = 'naiveattack_backup.npy'
    path = 'attack.npy'

    #x_adv = np.concatenate(x_adv, axis=0)
    x_adv = np.load(path)
    #print(x_adv)
    x_adv = np.transpose(x_adv)
    W = np.transpose(W)
    V = np.transpose(V)

    #print(W.shape)
    #print(x_adv.shape)
    #print(V.shape)

    #Adversarial Accuracy

    #print(np.dot(W,x_adv).shape)
    #print(relu(np.dot(W,x_adv)).shape)
    advlayer1 = relu(np.dot(W,x_adv))
    #print((layer1 >= 0).all())
    #print(advlayer1.shape,"advlayer1")
    advlayer2 = np.dot(V,advlayer1)
    advlayer2 = np.transpose(advlayer2)

    advprediction = np.array([np.argmax(i) for i in advlayer2])


    #print(advprediction)
    print("Adversarial accuracy : ",100*np.sum(advprediction == mnist.test.labels)/10000.0,"%")
    #print(prediction[1])

    #Natural Accuracy
    x_nat = np.transpose(mnist.test.images)
    natlayer1 = relu(np.dot(W,x_nat))
    #print(natlayer1.shape,"natlayer1")
    natlayer2 = np.dot(V,natlayer1)
    natlayer2 = np.transpose(natlayer2)
    natprediction = np.array([np.argmax(i) for i in natlayer2])
    print("Natural accuracy : ",100*np.sum(natprediction == mnist.test.labels)/10000.0,"%")
    #print((x_nat-x_adv).shape)



    x_found = list(filter(lambda x : advprediction[x]== natprediction[x] and natprediction[x]== mnist.test.labels[x]  , range(10000)))
    x_found.sort(key = lambda y : firstDiff(natlayer2[y]))
    #x_found = x_found[:100]

    x_found_labels = [secondArgMax(natlayer2[i]) for i in x_found]
    output = np.array([natlayer2[i] for i in x_found])
    #print([firstDiff(natlayer2[x_found[i]]) for i in range(100)])

    # count = 0
    # x_found = []
    # x_found_labels = []
    # i = -1
    #
    # while(count < 50):
    #     i = random.randint(0,9999)
    #     if (advprediction[i]!=mnist.test.labels[i]) and (natprediction[i]==mnist.test.labels[i]) :
    #         x_found.append(i)
    #         x_found_labels.append(advprediction[i])
    #         count+=1

    savepath = 'pgdfail_indicesall.npy'#corrected for natural mistakes
    savepath2 = 'pgdfail_labelsall.npy'
    savepath3 = 'pgdfail_outputall.npy'

    np.save(savepath, x_found)
    np.save(savepath2, x_found_labels)
    np.save(savepath3, output)


    # plt.plot(list(map(np.max,np.abs(np.transpose(x_adv-x_nat)))))
    # plt.show()
    for i in range(len(x_nat)):
        plt.plot(abs(x_nat[i]-x_adv[i]))

    #print(len(np.abs(x_nat-x_adv)))
    #print("The max perturbation is", np.max(np.abs(x_nat-x_adv)))

    print("Fraction of switching labels : ", 100*np.sum(natprediction!=advprediction)/10000.0,"%")

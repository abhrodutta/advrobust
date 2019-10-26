"""This script truns the SDP attack on the MNIST examples where PGDattacks fails
to find an adversarial example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import mosek
from time import time
from mosek.fusion import *
from matplotlib import pyplot as plt

def relu(val):
    return np.array(list(map(lambda x: np.array(list(map(lambda y :max(y,0),x))),val)))

def relu1D(val):
    return np.array(list(map(lambda x: max(0,x),val)))

class FullSDPAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Compute a naive SDP attack ---- find a z that maximizes ||Wz|| -- only need to solve the SDP once"""
    self.model = model
    self.epsilon = epsilon

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, output, uinorm, uperp, objvalue, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""

    "shape of x is batch_size times dim"

    W = sess.run(self.model.W, feed_dict={self.model.x_input: x_nat, self.model.y_input: y})
    V = sess.run(self.model.V, feed_dict={self.model.x_input: x_nat, self.model.y_input: y})



    [n,d] = x_nat.shape
    z = np.zeros((n,d))

    for i in range(n):
      z[i,:] = self.solveSDP(x_nat[i,:], y[i], W, V, output[i], uinorm, uperp, objvalue, i)

    x = x_nat + z

    return x





  def solveSDP(self, x, y, W, V, classes, uinorm, uperp, objvalue, index):

    # modfiy here
    "solve the full SDP"

    [d,k] = W.shape


    "find true class of x"

    x = x.astype(float)
    x = np.asarray(x)
    advlayer1 = np.array(list(map(lambda x:max(0,x),(np.dot(x,W)))))
    advlayer2 = np.dot(advlayer1,V)
    advlayer2 = np.transpose(advlayer2)

    advprediction = np.array([np.argmax(i) for i in advlayer2])
    advprediction = np.argmax(advlayer2)

    z = [0]*d

    V = np.transpose(V)
    #num_classes is the multiclass parameter. Set numclasses to 1 for single class.
    for h in range(1,num_classes+1):
        v = V[classes[h],:] - V[advprediction,:]
        vplus = relu1D(v)
        vminus = relu1D(-v)
        W1 = np.matmul(np.diag(vplus), np.transpose(W))
        W2 = np.matmul(np.diag(vminus), np.transpose(W))

        W1 = W1.astype(float)
        W1 = np.asarray(W1)
        W2 = W2.astype(float)
        W2 = np.asarray(W2)

        W1minusW2 = (W1-W2).astype(float)
        W1minusW2 = np.asarray(W1minusW2)

        alpha = 0.2

        with mosek.fusion.Model("naiveSDP") as M:

            epsilonPrime = (self.epsilon)*alpha

            # define SDP over a (d+k+1) X  (d+k+1) PSD matrix and k additional variables
            X = M.variable("X", Domain.inPSDCone(k+d+1))
            R = M.variable("R",k, Domain.greaterThan(0,k))#****what domain?

            l1 = np.matmul(W1,x)
            obj1 = Expr.dot(l1,X.slice([0,1],[1,k+1]))
            obj2 = Expr.dot(W1,X.slice([1,k+1],[k+1,k+d+1]))
            obj3 = Expr.dot(np.matmul(np.transpose(np.ones(k)),W1minusW2),X.slice([0,k+1],[1,k+d+1]))
            obj4 = Expr.neg(Expr.sum(R))
            obj5 = Expr.mul(Expr.mul(Expr.transpose(Expr.ones(k)),W1minusW2),x)

            obj = Expr.add(obj1,Expr.add(obj2,Expr.add(obj3, Expr.add(obj4,obj5))))

            M.objective(ObjectiveSense.Maximize,obj)

            arr = np.ones(1+k+d)
            arr[k+1:] *= epsilonPrime**2
            M.constraint((X.diag()).slice(1,k+d+1),Domain.lessThan(arr[1:]))#v_i,u_i is delta=epsilonPrime?
            M.constraint(X.slice([0,0],[1,1]),Domain.equalsTo(1))#add a constraint u_0 = 1
            c1 = np.matmul(W2,x)
            c2 = Expr.mul(W2,Expr.transpose(X.slice([0,k+1],[1,k+d+1])))
            c3 = Expr.add(c1,c2)
            c4 = Expr.add(c3,R)
            c5 = Expr.add(c3,Expr.neg(R))

            M.constraint(c4,Domain.greaterThan(0,k,1))
            M.constraint(c5,Domain.lessThan(0,k,1))

            print("solving, alpha = ",alpha)
            starttime = time()
            M.solve()
            endtime = time()
            print("Time : ",endtime - starttime)
            print("SDP value : ", M.primalObjValue())
            L = np.linalg.cholesky((X.level()).reshape(d+k+1, d+k+1))

            uinorm[h-1][index] = [np.linalg.norm(L[k+i+1])**2 for i in range(d)]
            uperp[h-1][index] = [(np.linalg.norm(L[k+i+1]))**2 - (np.dot(L[0],L[k+i+1]))**2 for i in range(d)]
            objvalue[h-1][index]= M.primalObjValue()

            z = self.multiRound(L, x, y, W, V, X)

            if(np.max(np.abs(z))>0):
                return z
    return z



  def multiRound(self, L, x, y, W, V, X):

    global num,count
    count+=1
    print(count)
    [d,k]=W.shape

    z = [0]*d

    eps1range = [0.00001,0.00005,0.001, 0.005, 0.008, 0.01, 0.02, 0.04,0.06,0.08,
        0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.4]
    W = np.transpose(W)

    for e1 in eps1range:
      e2 = 1.0/e1

      for j in range(100):

        g = np.random.normal(0,1,d+k+1)

        rand = np.dot(g, L[0,:])

        z1 = (1-e2*rand)*(np.matmul(L[0,:],L[:,k+1:]))
        z2 = e2*np.matmul(L[k+1:,:],np.transpose(g))
        z = z1 + z2

        z = np.clip(z, -self.epsilon, self.epsilon) # ensure valid pixel range
        advx = np.clip(x+z, 0, 1)
        z = advx-x
        target = np.argmax(np.matmul(V,relu1D(np.matmul(W,(advx)))))
        if target != y :
            num+=1
            print ("Adversarial Example found! : ",count, num)
            return z

    return ([0]*d)



if __name__ == '__main__':
  import json
  import sys
  import math

  global num,count
  global batchnum,num_classes
  num_classes = 1
  global path_uinorm,path_uperp,path_objvalue,path_perturb
  path_uinorm = 'stats/pgdfail/uinorm/'
  path_uperp = 'stats/pgdfail/uperp/'
  path_objvalue = 'stats/pgdfail/objvalue/'
  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = FullSDPAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    adv_indices = np.load('pgdfail_indicesall.npy')
    adv_labels = np.load('pgdfail_labelsall.npy')
    output = np.load('pgdfail_outputall.npy')

    output = [sorted(enumerate(i) , key = lambda x : -x[1]) for i in output ]
    output = [list(map(lambda x : x[0] ,i)) for i in output]
    output = np.asarray(output)

    uinorm = np.zeros((1,num_classes,100,784))
    uperp = np.zeros((1,num_classes,100,784))

    num_found = []
    objvalue = np.zeros((1,num_classes,100))
    rand = np.array([np.random.randint(0,3179) for i in range(100)])
    np.save('100random3179.npy',rand)

    output = np.array([output[rand[i]] for i in range(100)])

    adv_indices = np.array([adv_indices[rand[i]] for i in range(100)])

    for batchnum in range(1):
        num = 0
        count = 0
        x = np.array([mnist.test.images[adv_indices[i]] for i in range(100)]) # adv accumulator
        y = np.array([mnist.test.labels[adv_indices[i]] for i in range(100)]) #ground truth for the adv eg

        x_perturb = attack.perturb(x, y, output,uinorm[batchnum],uperp[batchnum],objvalue[batchnum],sess)
        num_found.append(num)
        print(num)
        np.save('stats/pgdfailrandom100/perturb100.npy', x_perturb)

    np.save('stats/pgdfailrandom100/num_found.npy',num_found)

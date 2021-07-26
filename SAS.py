import os
os.environ['TF_CPP_Mninnrec_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import pandas as pd
import random
#from Auxiliary import GPU


def ReLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def inv(x):
    y = 1/(x + pow(10, -30) * np.ones_like(x)) * np.int64(x > 0)
    return y


def clip(x, bound):
    l2 = np.linalg.norm(x)
    #print(l2)
    if l2 > bound:
        print(1)
        return bound / l2 * x 
    else:
        return x


def sample(pi, m, Xi):
    w = np.random.binomial(1, 1-pi) * np.random.normal(m, np.sqrt(Xi))
    return w


class SAS():
    def __init__(self, nin, nrec, nout, T, optimizer):
        self.nin = nin
        self.nrec = nrec
        self.nout = nout
        self.T = T

        self.min = np.random.normal(size=(nrec, nin))/(nrec**0.5*nin**0.5)
        self.mout = np.random.normal(size=(nout, nrec))/(nout**0.5*nrec**0.5)
        self.m = np.random.normal(size=(nrec, nrec))/(nrec**0.5*nrec**0.5)
        self.piin = np.zeros((nrec, nin))
        self.piout = np.zeros((nout, nrec))
        self.pi = np.zeros((nrec, nrec))
        self.Xiin = np.random.random(size=(nrec, nin)) * 0.01
        self.Xiout = np.random.random(size=(nout, nrec)) * 0.01
        self.Xi = np.random.random(size=(nrec, nrec)) * 0.01

        self.optimizer = optimizer

        self.RMSmin = RMSprop()
        self.RMSmout = RMSprop()
        self.RMSm = RMSprop()
        self.RMSpiin = RMSprop()
        self.RMSpiout = RMSprop()
        self.RMSpi = RMSprop()
        self.RMSXiin = RMSprop()
        self.RMSXiout = RMSprop()
        self.RMSXi = RMSprop()
        
        self.Admin = Adam()
        self.Admout = Adam()
        self.Adm = Adam()
        self.Adpiin = Adam()
        self.Adpiout = Adam()
        self.Adpi = Adam()
        self.AdXiin = Adam()
        self.AdXiout = Adam()
        self.AdXi = Adam()

        self.greset()

    def reset(self, T):

        nin = self.nin
        nrec = self.nrec
        nout = self.nout

        self.x = np.zeros((T+1, nin))
        self.h = np.zeros((T+1, nrec))
        self.u = np.zeros((T+1, nrec))
        self.r = np.zeros((T+1, nrec))
        self.y = np.zeros((T+1, nout))
        self.z = np.zeros((T+1, nout))

        self.deltaout = np.zeros((T+1, nout))
        self.delta = np.zeros((T+1, nrec))

        self.epsilon = np.random.normal(size=(T+1, nrec))
        self.epsilonout = np.random.normal(size=(T+1, nout))

        self.gmout = np.zeros((T+1, nout, nrec))
        self.gpiout = np.zeros((T+1, nout, nrec))
        self.gXiout = np.zeros((T+1, nout, nrec))

        self.gmin = np.zeros((T+1, nrec, nin))
        self.gpiin = np.zeros((T+1, nrec, nin))
        self.gXiin = np.zeros((T+1, nrec, nin))

        self.gm = np.zeros((T+1, nrec, nrec))
        self.gpi = np.zeros((T+1, nrec, nrec))
        self.gXi = np.zeros((T+1, nrec, nrec))

        self.gLz = np.zeros((T+1, nout))
        self.dirac = np.zeros((T+1, nrec))

    def greset(self):
        self.ugmout, self.ugm, self.ugmin = np.zeros((nout, nrec)), np.zeros((nrec, nrec)), np.zeros((nrec, nin))
        self.ugpiout, self.ugpi, self.ugpiin = np.zeros((nout, nrec)), np.zeros((nrec, nrec)), np.zeros((nrec, nin))
        self.ugXiout, self.ugXi, self.ugXiin = np.zeros((nout, nrec)), np.zeros((nrec, nrec)), np.zeros((nrec, nin))

    def feedforward(self, input):

        self.h[0] = 0.1*np.ones(self.nrec).T
        self.r[0] = ReLU(self.h[0])
        input = input.reshape((28, 28))

        self.muin = (1 - self.piin) * self.min
        self.mu = (1 - self.pi) * self.m
        self.muout = (1 - self.piout) * self.mout

        self.rhoin = (1 - self.piin) * (self.min**2 + self.Xiin)
        self.rho = (1 - self.pi) * (self.m**2 + self.Xi)
        self.rhoout = (1 - self.piout) * (self.mout**2 + self.Xiout)

        self.Din = self.rhoin - self.muin**2
        self.Drec = self.rho - self.mu**2
        self.Dout = self.rhoout - self.muout**2

        for t, x in enumerate(input):
            s = t + 1
            self.x[s] = x.T

            self.delta[s] = np.sqrt(self.Din @ self.x[s]**2 + self.Drec @ self.r[s - 1]**2)
            self.deltaout[s] = np.sqrt(self.Dout @ self.r[s]**2)

            self.u[s] = self.mu @ self.r[s - 1] + self.muin @ self.x[s] + self.epsilon[s] * self.delta[s]
            self.h[s] = (1 - alpha) * self.h[s - 1] + alpha * self.u[s]
            self.r[s] = ReLU(self.h[s])
            self.z[s] = self.muout @ self.r[s] + self.epsilonout[s] * self.deltaout[s]
            self.y[s] = softmax(self.z[s])

        self.invdelta = inv(self.delta)
        self.invdeltaout = inv(self.deltaout)

    def sampleforward(self, input):

        self.wsample()

        x = np.zeros((T+1, nin))
        h = np.zeros((T+1, nrec))
        u = np.zeros((T+1, nrec))
        r = np.zeros((T+1, nrec))
        y = np.zeros((T+1, nout))
        z = np.zeros((T+1, nout))

        h[0] = 0.1*np.ones(self.nrec)
        r[0] = ReLU(h[0])

        input = input.reshape((28, 28))

        for t, xi in enumerate(input):
            
            self.wsample()

            s = t + 1

            x[s] = xi.T
            u[s] = self.w @ r[s - 1] + self.win @ x[s]
            h[s] = (1 - alpha) * h[s - 1] + alpha * u[s]
            r[s] = ReLU(h[s])
            z[s] = self.wout @ r[s]
            y[s] = softmax(z[s])

        return y
        
    
    def wsample(self):
        self.win = sample(self.piin, self.min, self.Xiin)
        self.w = sample(self.pi, self.m, self.Xi)
        self.wout = sample(self.piout, self.mout, self.Xiout)        

    def BPTT0(self):
        T = self.T
        dirac2 = np.zeros((T+1, nrec))
        dirac1 = np.zeros((T+1, nrec))

        self.dirac[T] = self.gLz[T].T @ (self.muout + (((self.rhoout - self.muout**2) * self.r[T]).T * self.epsilonout[T] * self.invdeltaout[T]).T) * np.int64(self.h[T] > 0)
        
        for t in range(T - 1):
            s = T - t - 1
            dirac2[s] = self.gLz[s].T @ (self.muout + (((self.rhoout - self.muout**2) * self.r[s]).T * self.epsilonout[s] * self.invdeltaout[s]).T) * np.int64(self.h[s] > 0)
            guh = (1 - self.pi) * self.m * np.int64(self.h[s] > 0) + \
                (((self.rho - self.mu**2) * self.r[s] * np.int64(self.h[s] > 0)).T * self.epsilon[s] * self.invdelta[s + 1]).T
            dirac1[s] = (1 - alpha) * self.dirac[s + 1] + alpha * self.dirac[s + 1] @ guh
            self.dirac[s] = dirac1[s] + dirac2[s]
            
    def BPTT1(self):

        T = self.T
        for t in range(T):

            s = T - t

            self.gmout[s] = (1 - self.piout) * self.r[s] + ((self.muout * self.piout * self.r[s]**2).T * self.epsilonout[s] * self.invdeltaout[s]).T
            self.gpiout[s] = -self.mout * self.r[s] + (((self.mout**2 * (1 - 2 * self.piout) - self.Xiout) * self.r[s]**2).T * self.epsilonout[s] / 2 * self.invdeltaout[s]).T
            self.gXiout[s] = (((1 - self.piout) * self.r[s]**2).T * self.epsilonout[s] / 2 * self.invdeltaout[s]).T

            self.gmin[s] = (1 - self.piin) * self.x[s] + ((self.muin * self.piin * self.x[s]**2).T * self.epsilon[s] * self.invdelta[s]).T
            self.gpiin[s] = -self.min * self.x[s] + (((self.min**2 * (1 - 2 * self.piin) - self.Xiin) * self.x[s]**2).T * self.epsilon[s] / 2 * self.invdelta[s]).T
            self.gXiin[s] = (((1 - self.piin) * self.x[s]**2).T * self.epsilon[s] / 2 * self.invdelta[s]).T

            self.gm[s] = (1 - self.pi) * self.r[s-1] + ((self.mu * self.pi * self.r[s-1]**2).T * self.epsilon[s] * self.invdelta[s]).T
            self.gpi[s] = -self.m * self.r[s-1] + (((self.m**2 * (1 - 2 * self.pi) - self.Xi) * self.r[s-1]**2).T * self.epsilon[s] / 2 * self.invdelta[s]).T
            self.gXi[s] = (((1 - self.pi) * self.r[s-1]**2).T * self.epsilon[s] / 2 * self.invdelta[s]).T


    def BPTT2(self):
        self.ggmout = np.sum(self.gmout * np.expand_dims(self.gLz, axis=2), axis=0)
        self.ggpiout = np.sum(self.gpiout * np.expand_dims(self.gLz, axis=2), axis=0)
        self.ggXiout = np.sum(self.gXiout * np.expand_dims(self.gLz, axis=2), axis=0)
        #print(self.gXiout.shape)

        self.ggm = alpha * np.sum(self.gm * np.expand_dims(self.dirac, axis=2), axis=0)
        self.ggpi = alpha * np.sum(self.gpi * np.expand_dims(self.dirac, axis=2), axis=0)
        self.ggXi = alpha * np.sum(self.gXi * np.expand_dims(self.dirac, axis=2), axis=0)

        #print(self.ggpi)
        self.ggmin = alpha * np.sum(self.gmin * np.expand_dims(self.dirac, axis=2), axis=0)
        self.ggpiin = alpha * np.sum(self.gpiin * np.expand_dims(self.dirac, axis=2), axis=0)
        self.ggXiin = alpha * np.sum(self.gXiin * np.expand_dims(self.dirac, axis=2), axis=0)

        self.ggmout, self.ggm, self.ggmin = clip(self.ggmout/bsize, 10), clip(self.ggm/bsize, 10), clip(self.ggmin/bsize, 10)
        self.ggpiout, self.ggpi, self.ggpiin = clip(self.ggpiout/bsize, 10), clip(self.ggpi/bsize, 10), clip(self.ggpiin/bsize, 10)
        self.ggXiout, self.ggXi, self.ggXiin = clip(self.ggXiout/bsize, 10), clip(self.ggXi/bsize, 10), clip(self.ggXiin/bsize, 10)

        #self.ggmout, self.ggm, self.ggmin = clip(self.ggmout, 10), clip(self.ggm, 10), clip(self.ggmin, 10)
        #self.ggpiout, self.ggpi, self.ggpiin = clip(self.ggpiout, 10), clip(self.ggpi, 10), clip(self.ggpiin, 10)
        #self.ggXiout, self.ggXi, self.ggXiin = clip(self.ggXiout, 10), clip(self.ggXi, 10), clip(self.ggXiin, 10)
        
        self.ugmout += self.ggmout
        self.ugm += self.ggm
        self.ugmin += self.ggmin
        self.ugpiout += self.ggpiout
        self.ugpi += self.ggpi
        self.ugpiin += self.ggpiin
        self.ugXiout += self.ggXiout
        self.ugXi += self.ggXi
        self.ugXiin += self.ggXiin
        #print(self.ugpi)

    def LOSS(self, label):
        ylabel = np.eye(10)[label]
        self.Loss = -ylabel @ np.log(self.y[-1])
        self.gLz = (self.y[-1] - ylabel).reshape((1, -1)) * np.eye(T+1)[:, [-1]]
        #print(self.gLz)
    
    def feedback(self, label):
        self.LOSS(label)
        self.BPTT0()
        self.BPTT1()
        self.BPTT2()
    
    def update(self):
        
        if self.optimizer == 'RMS':
            self.min = self.RMSmin.upgrad(self.ugmin, self.min)
            self.mout = self.RMSmout.upgrad(self.ugmout, self.mout)
            self.m = self.RMSm.upgrad(self.ugm, self.m)

            self.piin = self.RMSpiin.upgrad(self.ugpiin, self.piin)
            self.piout = self.RMSpiout.upgrad(self.ugpiout, self.piout)
            self.pi = self.RMSpi.upgrad(self.ugpi, self.pi)

            self.Xiin = self.RMSXiin.upgrad(self.ugXiin, self.Xiin)
            self.Xiout = self.RMSXiout.upgrad(self.ugXiout, self.Xiout)
            self.Xi = self.RMSXi.upgrad(self.ugXi, self.Xi)

        if self.optimizer == 'Adam':
            self.min = self.Admin.upgrad(self.ugmin, self.min)
            self.mout = self.Admout.upgrad(self.ugmout, self.mout)
            self.m = self.Adm.upgrad(self.ugm, self.m)

            self.piin = self.Adpiin.upgrad(self.ugpiin, self.piin)
            self.piout = self.Adpiout.upgrad(self.ugpiout, self.piout)
            self.pi = self.Adpi.upgrad(self.ugpi, self.pi)

            self.Xiin = self.AdXiin.upgrad(self.ugXiin, self.Xiin)
            self.Xiout = self.AdXiout.upgrad(self.ugXiout, self.Xiout)
            self.Xi = self.AdXi.upgrad(self.ugXi, self.Xi)

        self.piin = np.clip(self.piin, 0, 1)
        self.Xiin = np.maximum(self.Xiin, 0)
        self.pi = np.clip(self.pi, 0, 1)
        self.Xi = np.maximum(self.Xi, 0)
        self.piout = np.clip(self.piout, 0, 1)
        self.Xiout = np.maximum(self.Xiout, 0)    


class RMSprop():
    def __init__(self):
        self.lr = 0.01
        self.beta = 0.9
        self.esplion = 1e-8
        self.s = 0
        self.t = 0

    def initial(self):
        self.s = 0
        self.t = 0

    def upgrad(self, grad, theta):
        self.t += 1
        self.decay = 1e-4
        self.s = self.beta * self.s + (1 - self.beta) * grad**2
        theta -= self.lr * (grad / np.sqrt(self.s + self.esplion) + self.decay * theta)
        return theta


class Adam():
    def __init__(self):
        self.lr = 0.005
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.esplion = 1e-8
        self.v = 0
        self.s = 0
        self.t = 0
        
    def initial(self):
        self.v = 0
        self.s = 0
        self.t = 0

    def upgrad(self, grad, theta):
        self.t += 1
        self.decay = 1e-4
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2
        self.mhat = self.v / (1 - self.beta1**self.t)
        self.shat = self.s / (1 - self.beta2**self.t)
        #print(self.mhat / (np.sqrt(self.shat) + self.esplion))
        theta -= self.lr * (self.mhat / (np.sqrt(self.shat) + self.esplion) + self.decay * theta)
        return theta


nin = 28
nrec = 100
nout = 10
T = 28
alpha = 0.1
bsize = 2
optimizer = 'Adam'
epoch = 100


my_model = SAS(nin, nrec, nout, T, optimizer)


traindatas = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train_100.csv", header=None))
testdatas = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_10.csv", header=None))
#traindatas = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train.csv", header=None))
#testdatas = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_100.csv", header=None))

labels = []
accuarys = []

count = 0
for e in range(epoch):
    np.random.shuffle(traindatas)
    for i, traindata in enumerate(traindatas):
        my_model.reset(T)
        input = traindata[1:]/255
        label = traindata[0]
        my_model.feedforward(input)
        my_model.feedback(label)
        if i % bsize == 0 and i != 0:    
            my_model.update()
            my_model.greset()
        count += 1
        print("\rTrain Process: {:.2f}% ".format(count/(epoch * traindatas.shape[0])*100), end='')
        if count % (bsize * 10) == 0 and count != 0:
            print("\n", end='')
            count2 = 0
            scorecard = []
            for testdata in testdatas:
                my_model.reset(T)
                input = testdata[1:]/255
                label = testdata[0]
                y = my_model.sampleforward(input)
                #print(my_model.pi)
                #print(my_model.y)
                if label == np.argmax(y[-1]):
                    scorecard.append(1)
                else:
                    scorecard.append(0)
                count2 += 1
                print("\rTest Process: {:.2f}% ".format(count2/testdatas.shape[0]*100), end='')
            scorecard = np.array(scorecard)
            accuary = scorecard.sum() / scorecard.size
            print("Accuary: {:.2f}".format(accuary))
            accuarys.append(accuary)
        if count % 500 == 0 and count != 0:
            file = pd.DataFrame(data=accuarys)
            file.to_csv(r'F:\桌面文件\PMI\SAS\accuarys3.csv')

            
    












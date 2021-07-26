from operator import countOf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy.lib.function_base import append
from tensorflow.python.framework import dtypes
from Auxiliary import GPU
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import time


def ReLU(x):
    return tf.nn.relu(x)


def dReLU(x):
    return tf.constant(np.int64(x > 0), dtype=tf.float64)


def trans(x):
    dim = len(x.shape)
    return tf.transpose(x, tuple(range(dim-2)) + (dim-1, dim-2))


def exdim(x):
    return np.expand_dims(x, 0)


def exdim2(x):
    return np.expand_dims(np.expand_dims(x, 0), 0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    return tf.exp(x) / tf.expand_dims(tf.reduce_sum(tf.exp(x), 2), 2)


def inv(x):
    y = 1/(x + pow(10, -30) * tf.ones_like(x)) * dReLU(x)
    return y


def clip(x, bound):
    l2 = tf.norm(x)
    if l2 > bound:
        return bound / l2 * x
    else:
        return x


def sample(pi, m, Xi):
    w = tf.constant(np.random.binomial(1, 1-pi), dtype=tf.float64) * tf.constant(np.random.normal(m, tf.sqrt(Xi)), dtype=tf.float64)
    return w


def LOSS(y, label, bsize):
    # 损失函数
    ylabel = np.eye(10)[label]
    Loss = -ylabel @ tf.math.log(y[-1])
    dLoss = tf.expand_dims(tf.reshape((trans(y[-1]) - ylabel), (1, bsize, -1)) * np.eye(T+1)[:, [-1]].reshape(T+1, -1, 1), 3)
    return dLoss, Loss
    

def decay(process, floor, lr0, step):
    lr = 0.5**(process // step) * lr0
    lr[lr <= floor] = floor
    return lr


class SAS():
    def __init__(self, nin, nrec, nout, T, optimizer, alpha, bsize):
        self.nin = nin
        self.nrec = nrec
        self.nout = nout
        self.alpha = alpha
        self.bsize = bsize
        self.T = T

        # 模型参数
        self.min = tf.Variable(np.random.normal(size=(nrec, nin))/(nrec**0.5*nin**0.5))
        self.mout = tf.Variable(np.random.normal(size=(nout, nrec))/(nout**0.5*nrec**0.5))
        self.m = tf.Variable(np.random.normal(size=(nrec, nrec))/(nrec**0.5*nrec**0.5))
        self.piin = tf.Variable(np.zeros((nrec, nin)))
        self.piout = tf.Variable(np.zeros((nout, nrec)))
        self.pi = tf.Variable(np.zeros((nrec, nrec)))
        self.Xiin = tf.Variable(np.random.random(size=(nrec, nin)) * 0.01)
        self.Xiout = tf.Variable(np.random.random(size=(nout, nrec)) * 0.01)
        self.Xi = tf.Variable(np.random.random(size=(nrec, nrec)) * 0.01)

        # 创建优化器
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

    def feedforward(self, input):

        nrec = self.nrec
        nout = self.nout
        bsize = self.bsize

        # 时间步数和index的值相同
        self.deltaout = [tf.zeros((1, bsize, nout, 1), dtype=tf.float64)]
        self.delta = [tf.zeros((1, bsize, nrec, 1), dtype=tf.float64)]

        self.epsilon = tf.constant(np.random.normal(size=(T+1, bsize, nrec, 1)), dtype=tf.float64)
        self.epsilonout = tf.constant(np.random.normal(size=(T+1, bsize, nout, 1)), dtype=tf.float64)

        bsize = self.bsize

        self.h = []
        self.r = []
        self.x = [tf.zeros((1, bsize, nin, 1), dtype=tf.float64)]
        self.u = [tf.zeros((1, bsize, nrec, 1), dtype=tf.float64)]
        self.y = [tf.zeros((1, bsize, nout, 1), dtype=tf.float64)]
        self.z = [tf.zeros((1, bsize, nout, 1), dtype=tf.float64)]

        self.h.append(0.1*tf.ones((1, bsize, self.nrec, 1), dtype=tf.float64))
        self.r.append(ReLU(self.h[0]))

        input = tf.transpose(input.reshape((bsize, 1, -1, 28, 1)), (2, 1, 0, 3, 4))

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
            self.x.append(tf.constant(x))

            self.delta.append(tf.sqrt(self.Din @ self.x[s]**2 + self.Drec @ self.r[s - 1]**2))
            
            self.u.append(self.mu @ self.r[s - 1] + self.muin @ self.x[s] + self.epsilon[s] * self.delta[s])
            self.h.append((1 - self.alpha) * self.h[s - 1] + self.alpha * self.u[s])
            self.r.append((ReLU(self.h[s])))

            self.deltaout.append(tf.sqrt(self.Dout @ self.r[s]**2))

            self.z.append(self.muout @ self.r[s] + self.epsilonout[s] * self.deltaout[s])
            self.y.append(softmax(self.z[s]))
        
        self.y = tf.concat(self.y, 0)
        self.delta = tf.concat(self.delta, 0)
        self.deltaout = tf.concat(self.deltaout, 0)

    def sampleforward(self, input, tsize):

        h = []
        r = []
        x = [tf.zeros((1, tsize, nin, 1), dtype=tf.float64)]
        u = [tf.zeros((1, tsize, nrec, 1), dtype=tf.float64)]
        y = [tf.zeros((1, tsize, nout, 1), dtype=tf.float64)]
        z = [tf.zeros((1, tsize, nout, 1), dtype=tf.float64)]

        h.append(0.1*tf.ones((1, tsize, self.nrec, 1), dtype=tf.float64))
        r.append(ReLU(h[0]))

        input = tf.transpose(input.reshape((tsize, 1, -1, 28, 1)), (2, 1, 0, 3, 4))

        for t, xi in enumerate(input):
            
            self.wsample()

            s = t + 1

            x.append(tf.constant(xi))
            u.append(self.w @ r[s - 1] + self.win @ x[s])
            h.append((1 - self.alpha) * h[s - 1] + self.alpha * u[s])
            r.append(ReLU(h[s]))
            z.append(self.wout @ r[s])
            y.append(softmax(z[s]))

        y = tf.concat(y, 0)
        return y
        
    def wsample(self):
        # 采样权重w
        self.win = sample(self.piin, self.min, self.Xiin)
        self.w = sample(self.pi, self.m, self.Xi)
        self.wout = sample(self.piout, self.mout, self.Xiout)        

    def feedback(self, dLoss, learningrate):
        T = self.T

        bsize = self.bsize
        dirac = []
        dirac2 = [tf.zeros((1, bsize, 1, nrec), dtype=tf.float64)]
        dirac1 = [tf.zeros((1, bsize, 1, nrec), dtype=tf.float64)]

        piout, muout, Xiout, mout, rhoout = self.piout, self.muout, self.Xiout, self.mout, self.rhoout
        piin, muin, Xiin, min = self.piin, self.muin, self.Xiin, self.min
        pi, mu, Xi, m, rho = self.pi, self.mu, self.Xi, self.m, self.rho

        # 计算Δ的倒数，且防止该倒数的分母为0
        invdelta = inv(self.delta)
        invdeltaout = inv(self.deltaout)

        dirac.append(trans(dLoss[T]) @ (muout + (rhoout - self.muout**2) * self.epsilonout[T] * invdeltaout[T] * trans(self.r[T])) * trans(dReLU(self.h[T])))

        for t in range(T - 1):
            s = t + 1 
            dirac2.append(trans(dLoss[T-s]) @ (muout + (rhoout - muout**2) * self.epsilonout[T-s] * invdeltaout[T-s] * trans(self.r[T-s])) * trans(dReLU(self.h[T-s])))
            guh = (1 - pi) * m * trans(dReLU(self.h[T-s])) + (rho - mu**2) * trans(self.r[T-s] * dReLU(self.h[T-s])) * self.epsilon[T-s] * invdelta[T-s+1]
            dirac1.append((1 - self.alpha) * dirac[s-1] + self.alpha * dirac[s-1] @ guh)
            dirac.append(dirac1[s] + dirac2[s])

        dirac.append(tf.zeros((1, bsize, 1, nrec), dtype=tf.float64))
        dirac.reverse()

        self.dirac = trans(tf.concat(dirac, 0))
        self.r = tf.concat(self.r, 0)
        self.x = tf.concat(self.x, 0)

        gmout = (1 - piout) * trans(self.r) + muout * piout * trans(self.r**2) * self.epsilonout * invdeltaout
        gpiout = -mout * trans(self.r) + (mout**2 * (1 - 2 * piout) - Xiout) * trans(self.r**2) * self.epsilonout * invdeltaout / 2
        gXiout = (1 - piout) * trans(self.r**2) * self.epsilonout * invdeltaout / 2

        gmin = (1 - piin) * trans(self.x) + muin * piin * trans(self.x**2) * self.epsilon * invdelta
        gpiin = -min * trans(self.x) + (min**2 * (1 - 2 * piin) - Xiin) * trans(self.x**2) * self.epsilon * invdelta / 2
        gXiin = (1 - piin) * trans(self.x**2) * self.epsilon * invdelta / 2

        rr = tf.concat([tf.zeros((1, self.bsize, self.nrec, 1), dtype=tf.float64), self.r[:-1]], axis=0)

        gm = (1 - pi) * trans(rr) + mu * pi * trans(rr**2) * self.epsilon * invdelta
        gpi = -m * trans(rr) + (m**2 * (1 - 2 * pi) - Xi) * trans(rr**2) * self.epsilon * invdelta / 2
        gXi = (1 - pi) * trans(rr**2) * self.epsilon * invdelta / 2

        # 计算梯度
        ggmout = tf.reduce_sum(gmout * dLoss, 0)
        ggpiout = tf.reduce_sum(gpiout * dLoss, 0)
        ggXiout = tf.reduce_sum(gXiout * dLoss, 0)

        ggm = self.alpha * tf.reduce_sum(gm * self.dirac, 0)
        ggpi = self.alpha * tf.reduce_sum(gpi * self.dirac, 0)
        ggXi = self.alpha * tf.reduce_sum(gXi * self.dirac, 0)

        ggmin = self.alpha * tf.reduce_sum(gmin * self.dirac, 0)
        ggpiin = self.alpha * tf.reduce_sum(gpiin * self.dirac, 0)
        ggXiin = self.alpha * tf.reduce_sum(gXiin * self.dirac, 0)

        # 梯度裁剪
        ggmout, ggm, ggmin = clip(ggmout/bsize, 10), clip(ggm/bsize, 10), clip(ggmin/bsize, 10)
        ggpiout, ggpi, ggpiin = clip(ggpiout/bsize, 10), clip(ggpi/bsize, 10), clip(ggpiin/bsize, 10)
        ggXiout, ggXi, ggXiin = clip(ggXiout/bsize, 10), clip(ggXi/bsize, 10), clip(ggXiin/bsize, 10)
        
        # 更新权重
        if self.optimizer == 'RMS':
            self.min = self.RMSmin.upgrad(tf.reduce_sum(ggmin, 0), self.min, learningrate[0])
            self.mout = self.RMSmout.upgrad(tf.reduce_sum(ggmout, 0), self.mout, learningrate[1])
            self.m = self.RMSm.upgrad(tf.reduce_sum(ggm, 0), self.m, learningrate[2])

            self.piin = self.RMSpiin.upgrad(tf.reduce_sum(ggpiin, 0), self.piin, learningrate[3])
            self.piout = self.RMSpiout.upgrad(tf.reduce_sum(ggpiout, 0), self.piout, learningrate[4])
            self.pi = self.RMSpi.upgrad(tf.reduce_sum(ggpi, 0), self.pi, learningrate[5])

            self.Xiin = self.RMSXiin.upgrad(tf.reduce_sum(ggXiin, 0), self.Xiin, learningrate[6])
            self.Xiout = self.RMSXiout.upgrad(tf.reduce_sum(ggXiout, 0), self.Xiout, learningrate[7])
            self.Xi = self.RMSXi.upgrad(tf.reduce_sum(ggXi, 0), self.Xi, learningrate[8])

        if self.optimizer == 'Adam':
            self.min = self.Admin.upgrad(tf.reduce_sum(ggmin, 0), self.min, learningrate[0])
            self.mout = self.Admout.upgrad(tf.reduce_sum(ggmout, 0), self.mout, learningrate[1])
            self.m = self.Adm.upgrad(tf.reduce_sum(ggm, 0), self.m, learningrate[2])

            self.piin = self.Adpiin.upgrad(tf.reduce_sum(ggpiin, 0), self.piin, learningrate[3])
            self.piout = self.Adpiout.upgrad(tf.reduce_sum(ggpiout, 0), self.piout, learningrate[4])
            self.pi = self.Adpi.upgrad(tf.reduce_sum(ggpi, 0), self.pi, learningrate[5])

            self.Xiin = self.AdXiin.upgrad(tf.reduce_sum(ggXiin, 0), self.Xiin, learningrate[6])
            self.Xiout = self.AdXiout.upgrad(tf.reduce_sum(ggXiout, 0), self.Xiout, learningrate[7])
            self.Xi = self.AdXi.upgrad(tf.reduce_sum(ggXi, 0), self.Xi, learningrate[8])

        # 参数调整
        self.piin = tf.clip_by_value(self.piin, 0, 1)
        self.Xiin = tf.nn.relu(self.Xiin, 0)
        self.pi = tf.clip_by_value(self.pi, 0, 1)
        self.Xi = tf.nn.relu(self.Xi, 0)
        self.piout = tf.clip_by_value(self.piout, 0, 1)
        self.Xiout = tf.nn.relu(self.Xiout, 0)


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

    def upgrad(self, grad, theta, newlr):
        self.t += 1
        self.lr = newlr
        self.decay = 1e-4
        self.s = self.beta * self.s + (1 - self.beta) * grad**2
        theta -= self.lr * (grad / tf.sqrt(self.s + self.esplion) + self.decay * theta)
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

    def upgrad(self, grad, theta, newlr):
        self.t += 1
        self.lr = newlr
        self.decay = 1e-4
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2
        self.mhat = self.v / (1 - self.beta1**self.t)
        self.shat = self.s / (1 - self.beta2**self.t)
        theta = theta - self.lr * (self.mhat / (tf.sqrt(self.shat) + self.esplion) + self.decay * theta)
        return theta


def train(traindataset, testdataset, epoch, bsize, tsize, path, learnrate):
    accuarys = []
    Losses = []
    for e in range(epoch):
        np.random.shuffle(traindataset)
        traindatas = traindataset.reshape((-1, bsize, traindataset.shape[1]))
        for i, traindata in enumerate(traindatas):
            count = (i + 1) * bsize + e * traindataset.shape[0]
            process = count / (epoch * traindataset.shape[0]) * 100
            learningrate = decay(process, 1e-8, learnrate, 20)
            traininput = traindata[:, 1:]/255
            trainlabel = traindata[:, [0]]
            my_model.feedforward(traininput)
            dLoss, _ = LOSS(my_model.y, trainlabel, bsize)
            my_model.feedback(dLoss, learningrate)
            print("\rTrain Process: {:.2f}% ".format(process), end='')
            if count % (bsize * 1000) == 0 and count != 0:
                accuary, theloss = test(testdataset, tsize)
                print("\nAccuary: {:.2f} \t Loss: {:.2f}".format(accuary, theloss))
                accuarys.append(accuary)
                Losses.append(theloss)
                file = pd.DataFrame({'Accuary': accuarys, 'Loss': Losses})
                file.to_csv(path)


def test(testdataset, tsize):
    print("\n", end='')
    scorecard = []
    Loss = []
    testdatas = testdataset.reshape((-1, tsize, testdataset.shape[1]))
    for j, testdata in enumerate(testdatas):
        count = (j + 1) * tsize
        process = count/testdataset.shape[0] * 100
        testinput = testdata[:, 1:]/255
        testlabel = testdata[:, [0]]
        y = my_model.sampleforward(testinput, tsize)
        _, loss = LOSS(y, testlabel, tsize)
        scorecard.append(np.mean((np.argmax(y[-1], axis=1) - testlabel) == 0))
        Loss.append(np.mean(loss))
        print("\rTest Process: {:.4f}% ".format(process), end='')
    accuary = np.mean(np.array(scorecard))
    theloss = np.mean(np.array(Loss))
    return accuary, theloss


nin = 28
nrec = 100
nout = 10
T = 28
alpha = 0.1
bsize = 100
tsize = 50
endtsize = 500
optimizer = 'Adam'
epoch = 200
learnrate = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
 
time_start = time.time()
my_model = SAS(nin, nrec, nout, T, optimizer, alpha, bsize)


#traindataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train_100.csv", header=None))
#testdataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_10.csv", header=None))
traindataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train.csv", header=None))
testdataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_100.csv", header=None))
testdatasetend = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test.csv", header=None))


path1 = r'F:\桌面文件\PMI\SAS\the_accuarys5.csv'
path2 = r'F:\桌面文件\PMI\SAS\the_end_accuarys5.csv'

GPU()
train(traindataset, testdataset, epoch, bsize, tsize, path1, np.array(learnrate))
print("\nFinal Test", end="")
endaccuary, endloss = test(testdatasetend, endtsize)
print('\nThe final accuracy is {:.4f}, The final error is {:.4f}'.format(endaccuary, endloss))
time_end = time.time() 
print('totally cost', time_end-time_start)


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


def dReLU(x):
    return np.int64(x > 0)


def trans(x):
    dim = len(x.shape)
    return np.transpose(x, tuple(range(dim-2)) + (dim-1, dim-2))


def exdim(x):
    return np.expand_dims(x, 0)


def exdim2(x):
    return np.expand_dims(np.expand_dims(x, 0), 0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), 1)


def inv(x):
    y = 1/(x + pow(10, -30) * np.ones_like(x)) * np.int64(x > 0)
    return y


def clip(x, bound):
    l2 = np.linalg.norm(x)
    if l2 > bound:
        return bound / l2 * x 
    else:
        return x


def sample(pi, m, Xi):
    w = np.random.binomial(1, 1-pi) * np.random.normal(m, np.sqrt(Xi))
    return w


def LOSS(y, label, bsize):
    # 损失函数
    ylabel = np.eye(10)[label]
    Loss = -ylabel @ np.log(y[-1])
    dLoss = np.expand_dims((trans(y[-1]) - ylabel).reshape(1, bsize, -1) * np.eye(T+1)[:, [-1]].reshape(T+1, -1, 1), 3)
    return dLoss, Loss
    

class SAS():
    def __init__(self, nin, nrec, nout, T, optimizer, alpha, bsize):
        self.nin = nin
        self.nrec = nrec
        self.nout = nout
        self.alpha = alpha
        self.bsize = bsize
        self.T = T

        # 模型参数
        self.min = np.random.normal(size=(nrec, nin))/(nrec**0.5*nin**0.5)
        self.mout = np.random.normal(size=(nout, nrec))/(nout**0.5*nrec**0.5)
        self.m = np.random.normal(size=(nrec, nrec))/(nrec**0.5*nrec**0.5)
        self.piin = np.zeros((nrec, nin))
        self.piout = np.zeros((nout, nrec))
        self.pi = np.zeros((nrec, nrec))
        self.Xiin = np.random.random(size=(nrec, nin)) * 0.01
        self.Xiout = np.random.random(size=(nout, nrec)) * 0.01
        self.Xi = np.random.random(size=(nrec, nrec)) * 0.01

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
        self.deltaout = np.zeros((T+1, bsize, nout, 1))
        self.delta = np.zeros((T+1, bsize, nrec, 1))

        self.epsilon = np.random.normal(size=(T+1, bsize, nrec, 1))
        self.epsilonout = np.random.normal(size=(T+1, bsize, nout, 1))

        bsize = self.bsize

        self.x = np.zeros((T+1, bsize, nin, 1))
        self.h = np.zeros((T+1, bsize, nrec, 1))
        self.u = np.zeros((T+1, bsize, nrec, 1))
        self.r = np.zeros((T+1, bsize, nrec, 1))
        self.y = np.zeros((T+1, bsize, nout, 1))
        self.z = np.zeros((T+1, bsize, nout, 1))

        self.h[0] = 0.1*np.ones((bsize, self.nrec, 1))
        self.r[0] = ReLU(self.h[0])

        input = np.swapaxes(input.reshape((bsize, 28, -1, 1)), 0, 1)

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
            self.x[s] = x

            self.delta[s] = np.sqrt(self.Din @ self.x[s]**2 + self.Drec @ self.r[s - 1]**2)
            

            self.u[s] = self.mu @ self.r[s - 1] + self.muin @ self.x[s] + self.epsilon[s] * self.delta[s]
            self.h[s] = (1 - self.alpha) * self.h[s - 1] + self.alpha * self.u[s]
            self.r[s] = ReLU(self.h[s])

            self.deltaout[s] = np.sqrt(self.Dout @ self.r[s]**2)

            self.z[s] = self.muout @ self.r[s] + self.epsilonout[s] * self.deltaout[s]
            self.y[s] = softmax(self.z[s])

    def sampleforward(self, input, tsize):

        x = np.zeros((T+1, tsize, nin, 1))
        h = np.zeros((T+1, tsize, nrec, 1))
        u = np.zeros((T+1, tsize, nrec, 1))
        r = np.zeros((T+1, tsize, nrec, 1))
        y = np.zeros((T+1, tsize, nout, 1))
        z = np.zeros((T+1, tsize, nout, 1))

        h[0] = 0.1*np.ones((tsize, self.nrec, 1))
        r[0] = ReLU(h[0])

        input = np.swapaxes(input.reshape((tsize, 28, -1, 1)), 0, 1)

        for t, xi in enumerate(input):
            
            self.wsample()

            s = t + 1

            x[s] = xi
            u[s] = self.w @ r[s - 1] + self.win @ x[s]
            h[s] = (1 - self.alpha) * h[s - 1] + self.alpha * u[s]
            r[s] = ReLU(h[s])
            z[s] = self.wout @ r[s]
            y[s] = softmax(z[s])

        return y
        
    def wsample(self):
        # 采样权重w
        self.win = sample(self.piin, self.min, self.Xiin)
        self.w = sample(self.pi, self.m, self.Xi)
        self.wout = sample(self.piout, self.mout, self.Xiout)        

    def feedback(self, dLoss):
        T = self.T

        bsize = self.bsize
        dirac = np.zeros((T+1, bsize, 1, nrec))
        dirac2 = np.zeros((T+1, bsize, 1, nrec))
        dirac1 = np.zeros((T+1, bsize, 1, nrec))

        piout, muout, Xiout, mout, rhoout = self.piout, self.muout, self.Xiout, self.mout, self.rhoout
        piin, muin, Xiin, min = self.piin, self.muin, self.Xiin, self.min
        pi, mu, Xi, m, rho = self.pi, self.mu, self.Xi, self.m, self.rho

        # 计算Δ的倒数，且防止该倒数的分母为0
        self.invdelta = inv(self.delta)
        self.invdeltaout = inv(self.deltaout)

        dirac[T] = trans(dLoss[T]) @ (muout + (rhoout - self.muout**2) * self.epsilonout[T] * self.invdeltaout[T] * trans(self.r[T])) * trans(dReLU(self.h[T]))
        
        for t in range(T - 1):
            s = T - t - 1
            dirac2[s] = trans(dLoss[s]) @ (muout + (rhoout - muout**2) * self.epsilonout[s] * self.invdeltaout[s] * trans(self.r[s])) * trans(dReLU(self.h[s]))
            guh = (1 - pi) * m * trans(dReLU(self.h[s])) + (rho - mu**2) * trans(self.r[s] * dReLU(self.h[s])) * self.epsilon[s] * self.invdelta[s+1]
            dirac1[s] = (1 - self.alpha) * dirac[s + 1] + self.alpha * dirac[s+1] @ guh
            dirac[s] = dirac1[s] + dirac2[s]
        self.dirac = trans(dirac)

        self.gmout = (1 - piout) * trans(self.r) + muout * piout * trans(self.r**2) * self.epsilonout * self.invdeltaout
        self.gpiout = -mout * trans(self.r) + (mout**2 * (1 - 2 * piout) - Xiout) * trans(self.r**2) * self.epsilonout * self.invdeltaout / 2
        self.gXiout = (1 - piout) * trans(self.r**2) * self.epsilonout * self.invdeltaout / 2

        self.gmin = (1 - piin) * trans(self.x) + muin * piin * trans(self.x**2) * self.epsilon * self.invdelta
        self.gpiin = -min * trans(self.x) + (min**2 * (1 - 2 * piin) - Xiin) * trans(self.x**2) * self.epsilon * self.invdelta / 2
        self.gXiin = (1 - piin) * trans(self.x**2) * self.epsilon * self.invdelta / 2

        rr = np.concatenate((np.zeros((1, self.bsize, self.nrec, 1)), self.r[:-1]), axis=0)

        self.gm = (1 - pi) * trans(rr) + mu * pi * trans(rr**2) * self.epsilon * self.invdelta
        self.gpi = -m * trans(rr) + (m**2 * (1 - 2 * pi) - Xi) * trans(rr**2) * self.epsilon * self.invdelta / 2
        self.gXi = (1 - pi) * trans(rr**2) * self.epsilon * self.invdelta / 2

        # 计算梯度
        self.ggmout = np.sum(self.gmout * dLoss, axis=0)
        self.ggpiout = np.sum(self.gpiout * dLoss, axis=0)
        self.ggXiout = np.sum(self.gXiout * dLoss, axis=0)

        self.ggm = self.alpha * np.sum(self.gm * self.dirac, axis=0)
        self.ggpi = self.alpha * np.sum(self.gpi * self.dirac, axis=0)
        self.ggXi = self.alpha * np.sum(self.gXi * self.dirac, axis=0)

        self.ggmin = self.alpha * np.sum(self.gmin * self.dirac, axis=0)
        self.ggpiin = self.alpha * np.sum(self.gpiin * self.dirac, axis=0)
        self.ggXiin = self.alpha * np.sum(self.gXiin * self.dirac, axis=0)

        # 梯度裁剪
        self.ggmout, self.ggm, self.ggmin = clip(self.ggmout/bsize, 10), clip(self.ggm/bsize, 10), clip(self.ggmin/bsize, 10)
        self.ggpiout, self.ggpi, self.ggpiin = clip(self.ggpiout/bsize, 10), clip(self.ggpi/bsize, 10), clip(self.ggpiin/bsize, 10)
        self.ggXiout, self.ggXi, self.ggXiin = clip(self.ggXiout/bsize, 10), clip(self.ggXi/bsize, 10), clip(self.ggXiin/bsize, 10)
        
        # 更新权重
        if self.optimizer == 'RMS':
            self.min = self.RMSmin.upgrad(np.sum(self.ggmin, axis=0), self.min)
            self.mout = self.RMSmout.upgrad(np.sum(self.ggmout, axis=0), self.mout)
            self.m = self.RMSm.upgrad(np.sum(self.ggm, axis=0), self.m)

            self.piin = self.RMSpiin.upgrad(np.sum(self.ggpiin, axis=0), self.piin)
            self.piout = self.RMSpiout.upgrad(np.sum(self.ggpiout, axis=0), self.piout)
            self.pi = self.RMSpi.upgrad(np.sum(self.ggpi, axis=0), self.pi)

            self.Xiin = self.RMSXiin.upgrad(np.sum(self.ggXiin, axis=0), self.Xiin)
            self.Xiout = self.RMSXiout.upgrad(np.sum(self.ggXiout, axis=0), self.Xiout)
            self.Xi = self.RMSXi.upgrad(np.sum(self.ggXi, axis=0), self.Xi)

        if self.optimizer == 'Adam':
            self.min = self.Admin.upgrad(np.sum(self.ggmin, axis=0), self.min)
            self.mout = self.Admout.upgrad(np.sum(self.ggmout, axis=0), self.mout)
            self.m = self.Adm.upgrad(np.sum(self.ggm, axis=0), self.m)

            self.piin = self.Adpiin.upgrad(np.sum(self.ggpiin, axis=0), self.piin)
            self.piout = self.Adpiout.upgrad(np.sum(self.ggpiout, axis=0), self.piout)
            self.pi = self.Adpi.upgrad(np.sum(self.ggpi, axis=0), self.pi)

            self.Xiin = self.AdXiin.upgrad(np.sum(self.ggXiin, axis=0), self.Xiin)
            self.Xiout = self.AdXiout.upgrad(np.sum(self.ggXiout, axis=0), self.Xiout)
            self.Xi = self.AdXi.upgrad(np.sum(self.ggXi, axis=0), self.Xi)

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
        theta -= self.lr * (self.mhat / (np.sqrt(self.shat) + self.esplion) + self.decay * theta)
        return theta


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


my_model = SAS(nin, nrec, nout, T, optimizer, alpha, bsize)


#traindataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train_100.csv", header=None))
#testdataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_10.csv", header=None))
traindataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_train.csv", header=None))
testdataset = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test_100.csv", header=None))
testdatasetend = np.array(pd.read_csv(r"F:\桌面文件\PMI\python\python\神经网络训练数据\mnist_test.csv", header=None))


def train(traindataset, testdataset, epoch, bsize, tsize, path):
    accuarys = []
    Losses = []
    for e in range(epoch):
        np.random.shuffle(traindataset)
        traindatas = traindataset.reshape((-1, bsize, traindataset.shape[1]))
        for i, traindata in enumerate(traindatas):
            traininput = traindata[:, 1:]/255
            trainlabel = traindata[:, [0]]
            my_model.feedforward(traininput)
            dLoss, _ = LOSS(my_model.y, trainlabel, bsize)
            my_model.feedback(dLoss)
            count = (i + 1) * bsize + e * traindataset.shape[0]
            print("\rTrain Process: {:.2f}% ".format(count/(epoch * traindataset.shape[0])*100), end='')
            if count % (bsize * 10) == 0 and count != 0:
                accuary, theloss = test(testdataset, tsize)
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
        testinput = testdata[:, 1:]/255
        testlabel = testdata[:, [0]]
        y = my_model.sampleforward(testinput, tsize)
        _, loss = LOSS(y, testlabel, tsize)
        scorecard.append(np.mean((np.argmax(y[-1], axis=1) - testlabel) == 0))
        Loss.append(np.mean(loss))
        count = (j + 1) * tsize
        print("\rTest Process: {:.2f}% ".format(count/testdataset.shape[0]*100), end='')
    accuary = np.mean(np.array(scorecard))
    theloss = np.mean(np.array(Loss))
    print("\nAccuary: {:.2f} \t Loss: {:.2f}".format(accuary, theloss))
    return accuary, theloss


path1 = r'F:\桌面文件\PMI\SAS\the_accuarys.csv'
path2 = r'F:\桌面文件\PMI\SAS\the_end_accuarys.csv'

train(traindataset, testdataset, epoch, bsize, tsize, path1)
print("\nFinal Test", end="")
endaccuary, endloss = test(testdatasetend, endtsize)
print('The final accuracy is {:.6f}, The final error is {:.6f}'.format(endaccuary, endloss))







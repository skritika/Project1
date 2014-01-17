import numpy as np
import random
def load_data(file_loc,n):
    f = open(file_loc, 'r')
    m = sum(1 for line in f)
    matX = np.empty((m,n), dtype= float)
    matY = np.empty(m, dtype= float)
    r = 0;
    f = open(file_loc, 'r')
    for line in f:
        rw = line.split()
        matY[r] = int(rw[0])>0
        for i in range(1,n+1):
            sp = rw[i].split(":")
            matX[r,int(sp[0])-1] = int(sp[1])
        r = r + 1
    return (matX, matY)
def shuffle (X,y):
    (m,n) = X.shape
    mat = np.empty((m,n+1),dtype=float)
    mat[:,0]=y
    mat[:,1:]=X
    np.random.shuffle(mat)
    return (mat[:,1:],mat[:,0])
    
def scale_fit(X):
    return (np.mean(X,axis=0), np.std(X,axis=0))

def scale_transform(X, avg, sigma):
   return (X-avg)/sigma

def append_ones(X):
    return np.append(np.ones((X.shape[0],1)), X, axis=1)
     
def error(Y1, Y2):
    err = 0.0;
    for i in range(0,Y1.shape[0]):
        if (Y1[i]==Y2[i]):
            err = err+1
    return (1- err/Y1.shape[0]);

def predict(X,w):
    y = np.empty((X.shape[0],1))
    X = append_ones(X)
    y = sigmoid(np.dot(X,w))
    y[y>0.5]=1
    y[y<0.5] =0
    y[y==0.5] = random.randint(0,1) # :-)
    return y

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

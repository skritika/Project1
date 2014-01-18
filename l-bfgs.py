### loading modules
import numpy as np
from helper import *
from scipy.optimize import fmin_l_bfgs_b

### global variables
# lamda - learning rate, mu = regularisation param., beta = weight vector
global num_features, num_epochs, num_samples, avg, sigma, lamda, mu , beta, X, y
num_features = 800
lamda = 0.01
mu = 0.1

def f(w):
    ret = np.dot(y, np.log(sigmoid(np.dot(X,w)))) + np.dot(1-y, np.log(1- sigmoid(np.dot(X,w))))
    w[0]=0
    ret = ret - mu* np.multiply(w,w)
    return -ret

def grad(w): 
    ret =  np.dot((y - sigmoid(np.dot(X,w))), X)
    w[0]=0
    ret = ret - 2*mu*w
    return -ret

### loading data
(X_train, Y_train) = load_data("1571/train", num_features)
(X_test, Y_test) = load_data("1571/test", num_features)

num_samples = X_train.shape[0]
### feature scaling
(avg, sigma) = scale_fit(X_train) ## to add - don't do scaling for binay features
X_train = scale_transform(X_train, avg, sigma)
X_test = scale_transform(X_test, avg, sigma)


### append ones
X_train = append_ones(X_train)

beta = np.zeros(num_features+1, dtype=float)

X = X_train 
y = Y_train # X, y are global variables and will be used in f, grad functions for l_bfgs
beta = fmin_l_bfgs_b(f, beta, fprime = grad)[0]
Y_pred = predict(X_test, beta)
print error(Y_test, Y_pred) 
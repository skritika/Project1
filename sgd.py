'''
Trains a logistic regression model by tuning hyperparameters lambda, mu and parameters beta_i's for the given training set. 	
Function parameters:
	train_path - location of the training set		
	num_features - try to remove this
	num_epochs - usually 5
	val - fraction of the training data to be used for validation
	method - update method - sgd or bfgs
'''
import numpy as np
import functools
from helper import *
from scipy.optimize import fmin_l_bfgs_b

def f2(mu, X, y, w):
    ret = np.dot(y, np.log(sigmoid(np.dot(X,w)))) + np.dot(1-y, np.log(1- sigmoid(np.dot(X,w))))
    w[0]=0
    ret = ret - mu* np.multiply(w,w)
    return -ret

def grad2(mu, X, y, w):
    ret =  np.dot((y - sigmoid(np.dot(X,w))), X)
    w[0]=0
    ret = ret - 2*mu*w
    return -ret

def train(train_path, num_features, num_epochs, val, method):
	#loading data
	(X, Y) = load_data(train_path, num_features)
	(X, Y) = shuffle(X,Y)

	#feature scaling
	num_samples = X.shape[0]
	(avg, sigma) = scale_fit(X) ## to add - don't do scaling for binary features
	vset_size = int(val*num_samples)
	tset_size = num_samples - vset_size
	X_train = scale_transform(X[0:tset_size], avg, sigma)
	Y_train = Y[0:tset_size]
	X_test = scale_transform(X[tset_size:num_samples], avg, sigma)
	Y_test = Y[tset_size:num_samples]	
	(X_train, Y_train) = shuffle(X_train, Y_train)

	#append ones
	X_train = append_ones(X_train)

	#grid search on lambda and mu 
	if(method=='sgd'):	
		for lamda in map(np.exp,np.arange(-6,0,.5)):
			for mu in map(np.exp,np.arange(-6,0,.5)):
				iterations = 0
				beta = np.zeros(num_features+1, dtype=float)
				while (iterations < num_epochs):
					for i in range(0, tset_size):
						new_beta = beta
						e = Y_train[i]-sigmoid(np.dot(X_train[i,:],beta)) #calculating (y-pi)
						new_beta[0] = beta[0] + lamda*(e*X_train[i,0]) #parameter updation - beta_0 not regularised
						for j in range(1,num_features+1):
							if(X_train[i,j]!=0):
								new_beta[j] = beta[j] + lamda*((e*X_train[i,j]) - 2* mu * beta[j])   
						beta = new_beta;
					iterations = iterations+1
				Y_pred = predict(X_test, beta)
				print lamda, mu, error(Y_test, Y_pred) 
	elif(method=='bfgs'):
		for mu in map(np.exp,np.arange(-8,0,.1)):
			beta = np.zeros(num_features+1, dtype=float)
		        f = functools.partial(f2, mu, X_train, Y_train)
            		grad = functools.partial(grad2, mu, X_train, Y_train)
			beta = fmin_l_bfgs_b(f, beta, fprime = grad)[0]
			Y_pred = predict(X_test, beta)
			print mu, error(Y_test, Y_pred) 

train('1571/train', 800, 5, 0.25, 'bfgs')

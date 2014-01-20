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
import matplotlib.pyplot as plt
import functools
from helper import *
from scipy.optimize import fmin_l_bfgs_b

def train(train_path, num_features, num_epochs, val, method, plot):
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
        
        #lambda, mu to use, comment out the other two
        lamda_range = map(np.exp,np.arange(-6,0,.5))
        mu_range =  map(np.exp,np.arange(-6,0,.5))
        #lamda_range = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        #mu_range = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.5, 1]
        
        #for graph
        lcl_points = []
        
	#grid search on lambda and mu
        beta = np.zeros(num_features+1, dtype=float)
	if(method=='sgd'):	
	    for lamda in lamda_range:
		for mu in mu_range:
                    iterations = 0
                    beta = np.zeros(num_features+1, dtype=float)
                    while (iterations < num_epochs):
                        iterations = iterations+1
                        for i in range(0, tset_size):
                            e = Y_train[i]-sigmoid(np.dot(X_train[i,:],beta)) #calculating (y-pi)
                            beta = beta + lamda* (np.dot(X_train[i,:],e)-2*mu*np.append([0],beta[1:])) 
                            #if(((i*4)%tset_size)==0): lcl_points.append(lcl(X_train,Y_train,beta))
                    Y_pred = predict(X_test, beta)
                    print lamda, mu, error(Y_test, Y_pred)
                    if(plot == 'true'):
                        plt.figure()
                        plt.plot([i/4.0 for i in range(len(lcl_points))], lcl_points)
                        plt.title("lambda : "+ str(lamda) +",  mu : "+ str(mu))                
                    lcl_points=[]
                #if(plot == 'true'): plt.show()
                                
	elif(method=='bfgs'):
		for mu in map(np.exp,np.arange(-8,0,.1)):
			beta = np.zeros(num_features+1, dtype=float)
		        f = functools.partial(f2, mu, X_train, Y_train)
            		grad = functools.partial(grad2, mu, X_train, Y_train)
			beta = fmin_l_bfgs_b(f, beta, fprime = grad)[0]
			Y_pred = predict(X_test, beta)
			print mu, error(Y_test, Y_pred)


def test (test_path, beta):
        (X_test_final, Y_test_final) = load_data(test_path, num_features)
        X_test_final = scale_transform(X_test_final, avg, sigma)
        Y_pred_final = predict(X_test_final, beta)
        print "final error", error(Y_test_final, Y_pred_final) 

train('1571/train', 800, 5, 0.25, 'sgd', 'false')

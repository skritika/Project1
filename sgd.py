### loading modules
import numpy as np
from helper import *

### global variables
# lamda - learning rate, mu = regularisation param., beta = weight vector
global num_features, num_epochs, num_samples, avg, sigma, lamda, mu , beta
num_features = 800
num_epochs = 5
lamda = 0.01
mu = 0.01

### loading data
(X_train, Y_train) = load_data("1571/train", num_features)
(X_test, Y_test) = load_data("1571/test", num_features)

num_samples = X_train.shape[0]
### feature scaling
(avg, sigma) = scale_fit(X_train) ## to add - don't do scaling for binay features
X_train = scale_transform(X_train, avg, sigma)
X_test = scale_transform(X_test, avg, sigma)

### shuffle
(X_train, Y_train) = shuffle(X_train, Y_train)

### append ones
X_train = append_ones(X_train)

iterations = 0
beta = np.zeros(num_features+1, dtype=float)
while (iterations < num_epochs):
    for i in range(0, num_samples):
        new_beta = beta
        e = Y_train[i]-sigmoid(np.dot(X_train[i,:],beta)) #calculating (y-pi)
        new_beta[0] = beta[0] + lamda*(e*X_train[i,0]) #beta_0 not regularised
        for j in range(1,num_features+1):
            if(X_train[i,j]!=0):
                new_beta[j] = beta[j] + lamda*((e*X_train[i,j]) - 2* mu * beta[j])   
        beta = new_beta;
    iterations = iterations+1

Y_pred = predict(X_test, beta)
print error(Y_test, Y_pred) 


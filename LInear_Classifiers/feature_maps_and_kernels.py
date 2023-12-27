# -*- coding: utf-8 -*-
"""
Here are the objectives:
    1. Generate 2-dim data that is far from being linearly seprable. So not even SVM would make sense
    2. Transform into a higher dimension and fit percetron in higher dimension to see if that works 
    3. Impelment perceptron with kernel function and again see how it performs when using the RBF kernel
"""


from matplotlib import pyplot as plt
import numpy as np
from perceptron_and_svm import *


'''Lets make a non seperable data set :'''
mean_pos = [0,0]
cov_pos =  0.5 * np.identity((2))
mean_neg = [0,0]
cov_neg =  10 * np.identity((2))
num_pos = 18 # number of positive points
num_neg = 36 # number of negative points

data = random_data(num_neg, num_pos, mean_pos, mean_neg, cov_pos, cov_neg)
n = data.shape[0]

# I do not want negative points to mingle to much with negative, so I remove negative close to the origin:
mask = np.ones(n, dtype=bool)
for i in range(n):
    if data[i, -1] == -1 and np.linalg.norm(data[i, :-1]) <= 2: #negative point close to origin -> remove
        mask[i] = False
    
data = data[mask,:]
make_plots(data)


''' Now lets map features into higher dimensions and if this helps'''

# first a polynomial basis:
phi_data_1 = np.hstack([ data,np.ones((data.shape[0],1)), data[:,[0]]*data[:,[0]], data[:,[1]]*data[:,[1]],  data[:,[0]]*data[:,[1]]])

# Looking at the the data we can come up with a "smart" mapping x,y ->x, y, x**2 + y**2
phi_data_2 = np.hstack([ data, data[:,[0]]*data[:,[0]] + data[:,[1]]*data[:,[1]]])


# fit models and compare
def classifier_accuracy(X: np.ndarray, Y: np.ndarray, classifier: tuple)->float:
    """
    Returns the classifier accuracy 

    Parameters
    ----------
    X : np.ndarray (n, k) with n obsrvation and k features
    Y : np.ndarray (n, 1) labels
    classifier : tuple (theta, theta_0)

    Returns
    -------
    classifier accuracy: float
    """
    n = X.shape[0]
    theta, theta_0 = classifier
    
    return np.sum( np.sign(X @ theta + theta_0) == np.sign(Y))/n
    

# fit models and calculate accuracy on polynomial basis:
theta, theta_0, num_steps, converged   = percetpron( phi_data_1[:, :-1], phi_data_1[:, [-1]] , max_steps=100) 
accuracy_perceptron = classifier_accuracy(phi_data_1[:, :-1], phi_data_1[:, [-1]], (theta, theta_0))

theta_svm, theta_0_svm = svm(phi_data_1[:, :-1], phi_data_1[:, [-1]], _lambda=0.1 )
accuracy_svm = classifier_accuracy(phi_data_1[:, :-1], phi_data_1[:, [-1]], (theta_svm, theta_0_svm))


print('Polynomial basis: ')
print('Perceptron:  Converged?: {} in {} steps. Accuracy: {}\n'.format(converged, num_steps, accuracy_perceptron))
print('SVM: Accuracy: {}\n'.format(accuracy_svm))
print('Apparently we could just as well toss a coin. This simple feature mapping does not help')


# fit models and calculate accuracy on the smart feature space:
theta, theta_0, num_steps, converged = percetpron( phi_data_2[:, :-1], phi_data_2[:, [-1]] , max_steps=100) 
accuracy_perceptron = classifier_accuracy(phi_data_2[:, :-1], phi_data_2[:, [-1]], (theta, theta_0))

theta_svm, theta_0_svm = svm(phi_data_2[:, :-1], phi_data_2[:, [-1]], _lambda=0.1 )
accuracy_svm = classifier_accuracy(phi_data_2[:, :-1], phi_data_2[:, [-1]], (theta_svm, theta_0_svm))

print('Smart transformation: ')
print('Perceptron:  Converged?: {} in {} steps. Accuracy: {}\n'.format(converged, num_steps, accuracy_perceptron))
print('SVM: Accuracy: {}\n'.format(accuracy_svm))
print('Now it works')


""" 
    Now lets try to do perceptron kernel with the so called radial basis kernel RBF
    that should act as tranformation of feature space into an infinitely 
    dimensional space
"""

def RBF(x: np.ndarray, y:np.ndarray, sigma2: float=1.0)->float:
    """
    Retutrns the the value of radial basis kernel function
    
    Parameters
    ----------
    X : np.ndarray (n,k) vectors in R^k. It may be one vector
    Y : np.ndarray (1, k) exactly one vector in R^k
    sigma2 : float small sigma^2 tend to increase non linearity of boundry

    Returns
    -------
    float : the RBF value
    
    """
    # the  np.newaxis is a trick, to make sure that we get RBF for each vector in x with y
    return np.exp(-np.linalg.norm(x[:,np.newaxis,:]-y, axis=2)**2/(2*sigma2))


def kernel_perceptron(X: np.ndarray, Y: np.ndarray, kernel_func, max_steps: int=1000)->np.ndarray:
    """
    Runs the kerenel perceptron algorithm
    Parameters
    ----------
    X : np.ndarray (n,k) observations
    Y : np.ndarray (n,1) labels
    kernel_func : function 
    max_steps : int number of iterations above which procedure halts

    Returns
    -------
    alpha : np.ndarray(m,1) # of mistakes per peach observation
    num_steps : int how many iterations over data set were made
    converged : bool did the algorithm converge
    """
    
    # initialize used variables. In this case, I will embedd the intercept into the features
    n,k = X.shape
    alpha = np.zeros((n,1)) # number of mistakes that we make
    num_steps = 0
    converged = False
    Z = np.hstack((X,Y))
    
    while not converged and num_steps < max_steps:
        converged =True # if no mistake is made over a full run through data we halt
        
        # iterate through data and 
        for j in range(n):
            # sum over all data points:
            pred = np.sum(alpha[:,[0]] * Z[:,[-1]] * kernel_func(Z[:,:-1], Z[[j],:-1] ))
            
            pred = np.sign(pred)
            
            if pred != Z[[j],-1]:
                alpha[[j], 0] += 1
                converged = False
        
        num_steps += 1
        
    return alpha, num_steps, converged
    
'''
Let's chceck now how the kernel percetpron performs on the data set. For I need 
new code to calculate classifier accuracy. Kernel percetron return the number
of mistakes per each observation and not a hyperplane
'''


def predict_with_alphas(new_x, X, Y, alpha, kernel_func):
    """
    Returns prediction based on kerenel perceptron trained on X,Y with kernel_func

    Parameters
    ----------
    new_x : np.ndarray(1,k) trnaspose of an observation
    X : np.ndarray(n,k) observations
    Y : np.ndarray(n,1) labels
    alpha : np.ndarray(n,1) number of mistakes per observation
    kernel_func : func kernel function

    Returns
    -------
    int: +1 or -1 

    """
    n,k = X.shape
    pred = 0
    for i in range(n):
        pred += alpha[[i],0] * Y[[i], 0] * kernel_func(X[[i],:] , new_x)
    return np.sign(pred)


def kernel_perceptron_accuracy(X, Y, alpha, kernel_func):
    """
    Parameters
    ----------
    X : np.ndarray(n,k) observations
    Y : np.ndarray(n,1) labels
    alpha : np.ndarray(n,1) number of mistakes per observation
    kernel_func : func kernel function

    Returns
    -------
    float: fraction of observations correctly classified
    """
    cnt = 0
    for i in range(X.shape[0]):
        if predict_with_alphas(X[[i], :], X, Y, alpha, kernel_func) == Y[[i],[0]]:
            cnt+=1
        else:
            print(X[[i],:], predict_with_alphas(X[[i], :], X, Y, alpha, RBF))
    return cnt/X.shape[0]


''' How will kernel perceptron perform on the original data:'''
alpha, num_steps, converged = kernel_perceptron(data[:, :-1], data[:, [-1]], RBF)
print('Did kernel perceptron converge: {} Ater {} steps. Accuracy: {}'.format( 
    converged,
    num_steps,
    kernel_perceptron_accuracy(data[:,:-1], data[:,[-1]], alpha, RBF),
 ))


''' Let's draw some data from one dsitribution, so that there is no clear separation 
    between positive and negative points. Will kernel percetpron do the job in ths case?
'''

wild_data = random_data(num_neg=20, num_pos=20, mean_pos=[0,0], mean_neg=[0,0], cov_pos = 5*np.identity(2), cov_neg=5*np.identity(2))
make_plots(wild_data)

alpha, num_steps, converged = kernel_perceptron(wild_data[:, :-1], wild_data[:, [-1]], RBF)
print('Did kernel perceptron converge: {} Ater {} steps. Accuracy: {}'.format( 
    converged,
    num_steps,
    kernel_perceptron_accuracy(wild_data[:,:-1], wild_data[:,[-1]], alpha, RBF),
 ))


# -*- coding: utf-8 -*-
"""
Here are the objectives:
    1. Implement perceptron algorithm and run it on linearly ssperable mock-up data
    2. On the same data learn a large margin classifier
    3. Change the data, so it's not linearly seprable any more and still fir the large margin classifier
"""

from matplotlib import pyplot as plt
import numpy as np


'-------------- Creating mock-up daa ------------------'
def random_data(num_neg: int, num_pos: int, mean_pos: list, mean_neg: list, cov_pos: list, cov_neg:list, random_seed: int=1)->np.ndarray:
    np.random.seed(random_seed)
    '''
    Returns data set with two sorts of labels +1 and -1
    The data is being generated from a two multivariate normal distributions
    '''
    
    points_positive  = np.random.multivariate_normal(mean_pos, cov_pos, num_pos)
    points_negative  = np.random.multivariate_normal(mean_neg, cov_neg, num_neg)
    
    points_positive = np.hstack((points_positive, np.ones((points_positive.shape[0],1))))
    points_negative = np.hstack((points_negative, -np.ones((points_negative.shape[0],1))))
    
    data = np.vstack((points_negative, points_positive))
    
    return data


'-------------- Perceptorton ------------------'
def percetpron(X: np.ndarray, Y: np.ndarray, max_steps: int=500)->np.ndarray:
    '''
    What we know is that the algorithm converges if the data is linearly seperable
    Morover if gamma is the max margin (separting distance) and all points are contained in 
    a ball of radius R, then max number of iterations is bounded by (R**2)/(gamma**2)
    
    
    Parameters
    ----------
    X : np.ndarray(n,k) data
    Y : np.ndarray(n,1) labels
    max_stpes : int maximum number of steps after which we halrt

    Returns
    -------
    theta : np.ndarrat(k , 1) the linear seprator normal vector to hyperplane
    theta_0 : float the intercept
    num_stpes : int
    converged : bool if the algorithm fails to converge in num_steps then False

    '''    
    # initialize used variables
    n,k = X.shape
    theta, theta_0 = np.zeros((k,1)), 0
    num_steps = 0
    converged = False

    # we do a random shuffle
    Z = np.hstack((X,Y))
    np.random.shuffle(Z)
    
    while not converged and num_steps < max_steps:
        converged =True # if no mistake is made over a full run through data we halt
        
        # iterate through data and 
        for i in range(n):
            
            if Z[[i], -1] * (Z[[i], :-1] @ theta + theta_0) <= 0:
                theta = theta +  Z[[i], [-1]] * Z[[i], :-1].T
                theta_0 = theta_0  + Z[[i], [-1]] 
                converged = False 
    
        num_steps += 1
    
    return theta, theta_0, num_steps, converged


'---------------- Large margin classifier  - SVM ------------------'
def svm(X, Y, _lambda, lr=0.001, epochs=200):
    '''
    The idea is to choose a linear classifier that is separates the 
    in a best way in some sense. If the set is linearly seperable then
    thise will yield a largest margin. Also we may choose ot missclasify 
    some points at but stay with a large (wide) margin. This leads
    to the use of hingle loss, which by no means is obvious nor trivial.
    Here I run gradient descent (not stochastic) with a regularization
    _lanbda to fit t
    
    
    Parameters
    ----------
    X : np.ndarray(n,k) data
    Y : np.ndarray(n,1) labels
    _lambda : float the regulatization hyper parameter
    lr : float learning rate for gradient decent
    epochs : int hyper parameter specifying the number of iterations

    Returns
    -------
    theta : np.ndarrat(k , 1) the linear seprator, normal vector to hyperplane
    theta_0 : float the intercept
    
    '''
    # initialize used variables
    n,k = X.shape
    theta, theta_0 = np.zeros((k,1)), 0
    grad_theta, grad_theta_0 = np.zeros((k,1)), 0
    
    # combine and shuffle data
    Z = np.hstack((X,Y))
    np.random.shuffle(Z)
    
    for epoch in range(epochs):
        for i in range(n): # update caclulate gradient for each observation
            if Z[[i], -1] * (Z[[i], :-1] @ theta + theta_0) < 1: 
                grad_theta = grad_theta + _lambda * theta  - Z[[i], -1] * Z[[i], :-1].T 
                grad_theta_0 = grad_theta_0 - Z[[i], -1] 
            else:
                grad_theta = grad_theta + _lambda * theta
                
        # update theta after one cycle over the data    
        theta = theta - lr * (1/n)*grad_theta
        theta_0 = theta_0 - lr * (1/n)*grad_theta_0
    
    return theta, theta_0


def make_plots(data: np.ndarray, linear_classifiers: list=[])->None:
    '''
    Generates plots. Included to avoid repetition of the same code
    
    Parameters
    ----------
    data : np.ndarray(n,k+1) last column contains labels +1, -1
    classifiers : list every elemnt of list is a (theta, theta_0) hyperplane
    
    Returns
    -------
    None.

    '''
    #  Generate data and min and max vlaues for setting plot window
    #data  = simple_linear_data(num_neg, num_pos, mean_pos, mean_neg, cov_pos, cov_neg)
    
    x_min, x_max = np.min(data[:,0]), np.max(data[:,0])
    y_min, y_max = np.min(data[:,1]), np.max(data[:,1])
    
    x_plot = np.linspace(x_min, x_max,num=50)
    
    # Prepare negative and positive points to plot 
    positive  = data[data[:, -1] == 1]
    negative  = data[data[:, -1] == -1]
    
    # Generate the plots    
    plt.scatter(positive[:, 0], positive[:, 1], marker='o', color='blue', label='positive') # +
    plt.scatter(negative[:, 0], negative[:, 1], marker='o', color='red', label='negative') # -
    
    # set plotting window size:
    plt.ylim(y_min - 1, y_max + 1)
    plt.xlim(x_min - 1, x_max + 1)

    for classifier in linear_classifiers:
        theta, theta_0, classifier_name = classifier
        y_plot = (-theta[0] * x_plot  - theta_0)/theta[1] # linear
        plt.plot(x_plot, y_plot, label=classifier_name)

    plt.title('Linear classifiers visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()


'----Finally I would like to generate data, fit classifiers and plot them ----'
if __name__ == '__main__':
    # Some means and covariance matrices to generate random data
    mean_pos = [-1.5,-1.5]
    cov_pos =  np.identity((2))
    mean_neg = [1.5,1.5]
    cov_neg =  np.identity((2))
    num_pos = 18 # number of positive points
    num_neg = 13 # number of negative points
    
    # Genrate data:
    data  = random_data(num_neg, num_pos, mean_pos, mean_neg, cov_pos, cov_neg)
    
    # run perceptron:
    theta, theta_0, converged, num_stpes = percetpron(data[:,:-1], data[:,[-1]] )
    
    # run svm:
    theta_svm, theta_0_svm = svm(data[:,:-1], data[:,[-1]], _lambda=0.1)
    
    # generate plots:
    classifiers = [(theta, theta_0, 'perceptron'), (theta_svm, theta_0_svm, 'svm')]
    make_plots(data,classifiers)


    '-------------------- Not seprable data --------------------'
    '''
    Now lets increase the variance so to make the data is not linearly 
    separable and will see how the svm performs. AS we can see it
    will missclasify a number of points but visually seem to be
    doing a pretty good job
    '''
    cov_pos =  10*np.identity((2))
    cov_neg =  10*np.identity((2))
    
    # regenerate data with higher variance:
    data  = random_data(num_neg, num_pos, mean_pos, mean_neg, cov_pos, cov_neg)    
    
    # run svm:
    theta_svm, theta_0_svm = svm(data[:,:-1], data[:,[-1]], _lambda=0.1)
    
    # generate plots:
    classifiers = [(theta, theta_0, 'perceptron'), (theta_svm, theta_0_svm, 'svm')]
    make_plots(data,[(theta_svm, theta_0_svm, 'svm')])


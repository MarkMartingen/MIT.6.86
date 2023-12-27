# -*- coding: utf-8 -*-
"""
    Objectives:
        1. Implment k-means clustering
        2. Implement k-medoids clusterng to see the differences
        3. Run algorithms on some mock-up data set, generetate scree plots 
        plot clusters
"""


from matplotlib import pyplot as plt
import numpy as np
from perceptron_and_svm import *
from typing import Callable


def l2_distance(X: np.ndarray ,C: np.ndarray)->np.ndarray:
    """
    Returns a distance array. The (i,j) entry  is the l2 distance between the 
    i-th row in X and j-th row in C

    Parameters
    ----------
    X : (n,d) array with n observation
    C : (K,d) array with d centers of clusters

    Returns
    -------
    aarray of shape (n,K)

    """
    return np.linalg.norm(X[:,np.newaxis,:] - C[np.newaxis,:,:], axis=-1)


def l1_distance(X: np.ndarray ,C: np.ndarray)->np.ndarray:
    """
    Returns a distance array. The (i,j) entry  is the l2 distance between the 
    i-th row in X and j-th row in C. Parameters just like in l2_distance
    """
    return np.linalg.norm(X[:,np.newaxis,:] - C[np.newaxis,:,:], ord=1, axis=-1)


def cluster_cost(X: np.ndarray, centers: np.ndarray, distance: Callable[...,float])->float:
    """
    Calcultates the cost as the sum of distances of all points from the assigned
    cluster center
    
    Parameters
    ----------
    X : (n,d) array with n observations
    centers : (K, d) array with centers of K clusters
    distance : function that clculates distance between X and centers

    Returns
    -------
    float: the cost (sum of squared distance) of clustering given by centers and distance

    """
    return np.sum(np.min(distance(X,centers), axis=1)**2)


def kmeans(X: np.ndarray, K: int, distance: Callable[...,float], max_steps: int=200, epsilon: float =0.00001, get_cost: bool=False)->tuple:
    """
    Performs kmeans clustering

    Parameters
    ----------
    X : (n,d) array of observations
    K : int number of clusters
    distance : function that clculates distance between X and centers
    max_steps : number of iterations after which procedures halts
    epsilon : float small number used a as a threshold for determining convergence
    get_cost : boo, if True, then the procedures returns a list of costs 

    Returns
    -------
    tuple with cluster, assignemnt, cluster centers, num iterations, convergence and a list of costs
    """
    
    # check spread of data and dimensions:
    min_value, max_value  = np.min(X), np.max(X)
    n,d = X.shape
    
    # initialize cluster centers with some random values
    # After initialization we assign points to clusters and make sure there are
    # no empty clusters
    initalization_in_progres = True
    
    while initalization_in_progres:
        C = np.random.uniform(low=min_value, high=max_value, size =(K, d))
    
        cluster_assignment = np.argmin(distance(X,C), axis=1).reshape((n,1))
        if np.all(np.isin(np.arange(K), cluster_assignment)):
            initalization_in_progres = False
    
    # max_steps to halt in case the algorithm was not convering
    num_steps = 0
    converged = False
    cost=[]
    
    while not converged and num_steps < max_steps:
        # assign points to clusters:
        cluster_assignment = np.argmin(distance(X,C), axis=1).reshape((n,1))
            
        # compute new cluster means:
        Z = np.hstack((X,cluster_assignment))
        new_C = np.zeros((K, d))
        
        for m in range(K):
            new_C[[m],:] = np.mean(Z[Z[:,-1]==m], axis=0).reshape((1,d+1))[:,:-1]
        
        
        # check if new cluster means new_C differ form old means C:
        if np.sum(np.abs(new_C - C)) < epsilon:
            converged = True
        
        # increment steps and prepare new centeres for next iteration
        C = new_C[:]
        num_steps += 1
        
        if get_cost:
            cost.append(cluster_cost(X, C, distance))
    
    if get_cost:    
        return Z, C, num_steps, converged, cost
    else:
        return Z, C, num_steps, converged


    
def kmedoids(X: np.ndarray, K: int, distance: Callable[...,float], max_steps: int=200, epsilon: float =0.00001, get_cost: bool=False)->tuple:
    """
    Performs k-medoids clustering

    Parameters
    ----------
    Same as in case of k-means
    Returns
    -------
    same as in case of k-means
    """
    
    # initialize cluster centers with some ranodm points from X
    n,d = X.shape
    C = X[np.random.choice(n, K, replace=False)]
    

    # max_steps to halt in case the algorithm was not convering
    num_steps = 0
    converged = False
    cost=[]

    while not converged and num_steps < max_steps:
        # assign points to clusters:
        cluster_assignment = np.argmin(distance(X,C), axis=1).reshape((n,1))
        
        #find the best representative per cluster:
        Z = np.hstack((X,cluster_assignment))
        new_C = np.zeros((K, d))
        
        for m in range(K):
            # for each cluster we choose the point that minimizes the sum of distances
            ind = np.argmin(np.sum(distance(Z[Z[:,-1]==m][:,:-1] , Z[:, :-1]), axis=0))
            new_C[[m],:] = X[[ind], :]
            
        # check if new cluster means new_C differ form old means C:
        if np.sum(np.abs(new_C - C)) < epsilon:
            converged = True
        
        # increment steps and prepare new centeres for next iteration
        C = new_C[:]
        num_steps += 1
        
        if get_cost:
            cost.append(cluster_cost(X, C, distance))
        
    if get_cost:    
        return Z, C, num_steps, converged, cost
    else:
        return Z, C, num_steps, converged


def gaussian_pdf(X, mu, cov):
    def gaussian_pdf_single(row, mu, cov):
        part1 = 1 / ( ((2* np.pi)**(mu.shape[1]/2)) * (np.linalg.det(cov)**(1/2)) )
        part2 = (-1/2) * ((row-mu).dot(np.linalg.inv(cov))).dot((row-mu).T)
        return float(part1 * np.exp(part2))
    return np.apply_along_axis(lambda row:\
        gaussian_pdf_single(row, mu, cov), axis=1, arr=X).reshape((X.shape[0],1))
    


def em(X: np.ndarray, K: int, distance, max_steps: int=200, epsilon: float=0.00001, get_cost=False):
    # check spread of data and dimensions:
    min_value, max_value  = np.min(X), np.max(X)
    n,d = X.shape
    
    # initiliaze gaussian mizture:
    mu = np.random.uniform(low=min_value, high=max_value, size =(K, d))
    var = np.random.uniform(low=0.001, high=max_value, size=K) 
    probs = np.random.dirichlet(np.ones(K)).reshape((K,1))
    num_steps = 0
    converged = False
    cost = []
    
    
    while not converged and num_steps < max_steps:
        # the e-step. Here we estimate the posterior probs of the observations in X
        posteriors = np.zeros((n, K)) #posteriors[i, j] is probability of j cluster give i-th observation
        for j in range(K):
            posteriors[:, [j]] = gaussian_pdf(X, mu[[j], :], cov=var[j]*np.identity(d))*probs[[j],[0]]
        
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        # print(posteriors[:,[0]].shape)
        # print(X.shape)
        # print((X * posteriors[:,[0]]).shape)
        # # now the m step 
        
        # first we update the probs (priors):
        probs = np.sum(posteriors, axis=0).reshape((K,1))
        probs = probs / np.linalg.norm(probs)
        
        # now we update the mu and variance:
        for j in range(K):
            mu[[j], :] = np.sum(X * posteriors[:, [j]], axis=0, keepdims=True) /np.sum(posteriors[:, [j]])
            var[j] = np.sum((np.linalg.norm(X - mu[[j], :], axis=1)**2).reshape((n,1))* posteriors[:, [j]],axis=0, keepdims=True) 
            var[j] = var[j] / (d * np.sum(posteriors[:, [j]]))
            var[j] = max(var[j], 0.001)
       
        num_steps += 1
        cost.append(cluster_cost(X, mu, distance))
        
        if len(cost)>1 and abs(cost[-1] - cost[-2])<epsilon:
            converged = True
        
    
    cluster_assignment = np.argmax(posteriors, axis=1, keepdims=True)
    
    return  np.hstack((X, cluster_assignment)), mu, num_steps, converged, cost
    
    
    

def run_clustering(X: np.ndarray, K_max: int, K_to_plot: int,  distance: Callable[...,float], method: Callable[...,tuple])->tuple:
    """
    Runs clustering on that X and produces a scree plot and plots
    the data given that it is 2 dimensional    
    
    Parameters
    ----------
    X : (n,d) n observations to be clustered
    K_max : int, the maximum number of clusters. Procedure will run clustering from 1 to K_max
    K_to_plot : int, choose the number of clusters to plot, works only if d==2
    distance : a function that clculates distance between X and cluster centers
    method : clustering method, for now 'kmeans' or 'kmedoids'

    Returns
    -------
    Genreates plots and returns a tuple with specific data for K_to_plot clusters
    """
    scree_plot = []
    
    if X.shape[1]==2:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        axes = np.array([axes]) 
    
    for k in range(1, K_max+1, 1):    
        Z, C, num_steps, converged, cost  = method(X, k, distance, get_cost=True)
        scree_plot.append(cost[-1])
        if k == K_to_plot and X.shape[1]==2:
            for i in range(k):
                axes[1].scatter(Z[Z[:,-1]==i][:, 0], Z[Z[:,-1]==i][:, 1])
            axes[1].set_title('{} Clusters. Convergence: {}. # clusters: {}'.format(k, converged, k))
            Z_return = Z[:]
            C_return = C[:]
            num_steps_return =num_steps 
            converged_return = converged 
            cost_returned = cost  
            
    axes[0].plot(range(1,K_max+1,1), scree_plot)
    axes[0].set_xticks(np.arange(1,K_max,1))
    axes[0].set_title('Scree plot')
    axes[0].set_ylabel('Cost')
    axes[0].set_xlabel('Number of clusters')
    
    plt.tight_layout()
    plt.show()
    
    # Just in case we would like to see the details:
    return Z_return, C_return, num_steps_return, converged_return, cost_returned
    
        
if __name__=='__main__':

    '''
        Generate data from 4 2-dmiensional gaussians to see how clustering algo's perform
    '''
    np.random.seed(1)
    
    X1 = np.random.multivariate_normal([2,3], 1*np.identity(2), 20)
    X2 = np.random.multivariate_normal([-2,-2], 2*np.identity(2), 50)
    X3 = np.random.multivariate_normal([3,-2], 2*np.identity(2), 20)
    X4 = np.random.multivariate_normal([-2,4], 2*np.identity(2), 10)    
    X = np.vstack([X1,X2,X3,X4])

    # Checking if I can add more dimensions
    # X = np.hstack( (X, np.random.uniform(size = (X.shape[0],1)) ))
    # run_clustering(X, 12, 5, l1_distance, kmeans)
    run_clustering(X, 12, 4, l2_distance, em)
    # print(em(X, 4, l2_distance))
    
    


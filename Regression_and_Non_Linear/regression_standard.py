
"""
OBJECTTIVES:
    
    - Generate some random multivaiate data x1,..., xn
    - for any sklarars a1,...,an definde y = a1*x1+..._an*xn +b + E, where E
    is random nnormal noise
    - use the closed formula to derive the a1,.., an from data x1,...,x, y
    - use stochastic gradient decent to derive a1,...., an
"""

import numpy as np
from typing import Callable as function


def draw_from_cauchy(size: tuple, location=0, scale=1)->np.ndarray:
    '''
        Returns an numpy array of size drawn from a cauchy 
        distribution with location=0 and scale=1
        Note that these are not mean and variance as the latter
        do not exist fot a cauchy distribution
    '''
    if scale <= 0:
        raise ValueError('Scale must be positive!') 
    return location * np.ones(size)  + scale * np.random.standard_cauchy(size)    
    

def draw_from_normal(size: tuple, location=0, scale=1)->np.ndarray:
    '''
        Returns an numpy array of size drawn from a normal 
        distribution with location=0 and scale=1
    '''
    if scale <= 0:
        raise ValueError('Scale must be positive!') 
    return np.random.normal(location, scale, size)
    

def generate_data(size: tuple, distribution: function, scalars: np.ndarray, intercept: float, error: function)->np.ndarray:
    '''
        Generates mock up data for the purpose of this exercise
        First a numpy array - X of size is drawn from the provided distribution
        Then X is multiplied by scalars, intercept and a random error is added
    '''
    X = distribution(size)
    errors  = error((size[0],1))
    Y = np.matmul(X, scalars) + intercept + errors
    return np.hstack((X,Y)) 
        

def regression_forumla(X: np.ndarray, y: np.ndarray)->np.ndarray:
    '''
        uses closed form regression formula to return the 
        regression paramenters
    '''
    return np.matmul(np.linalg.inv(np.matmul(X.T,X)), np.matmul(X.T,y))


def r_squared(X: np.ndarray, y: np.ndarray, fitted_params: np.ndarray)->float:
    '''
        Returns the R^2 goodness of fit measure
    '''
    prediction = np.matmul(X, fitted_params)
    return 1 - np.sum((y - prediction)**2) / np.sum((y - np.mean(y))**2)

def rmse(X: np.ndarray, y: np.ndarray, fitted_params: np.ndarray)-> float:
    '''
        Return the root mean square error 
    '''
    prediction = np.matmul(X, fitted_params)
    return np.std(prediction - y)
    

def quadratic_cost(X: np.ndarray, fitted_params: np.ndarray, y: np.ndarray)->float:
    '''
        Returns the quadratic cost for the fitted parameters
    '''
    prediction = np.matmul(X, fitted_params)
    return np.sum((y - prediction)**2)/(2*X.shape[0])

def quadratic_cost_grad(X: np.ndarray, fitted_params: np.ndarray, y: np.ndarray)->np.ndarray:
    '''
        Returns the gradient of the quadratic cost with respect to thr
        fitted parameters
    '''
    return np.mean((np.matmul(X, fitted_params) - y) * X, axis=0, keepdims=True).T
    
    
def GD(X, y, lr, epsilon, max_iter: int)->tuple:
    '''
        Performs gradien decent on the full data set
        X - araay with observations as rows
        y - the corresponding labels
        lr - leartning rate
        epsilon - if cost falls below epsilon procedure halts
        max_iter - maximum number of iteratrion
        Returns fited parameters, the final cost and number of interations 
        performed
    '''
    fitted_params = np.zeros((X.shape[1],1))
    
    cost = quadratic_cost(X, fitted_params, y)
    print('Initial cost: ', cost)
    counter = 0
    
    while counter < max_iter and abs(cost) >= epsilon:
        counter +=1 
        gradient = quadratic_cost_grad(X, fitted_params, y)
        fitted_params = fitted_params - lr * gradient
        cost = quadratic_cost(X, fitted_params, y)
        
    return fitted_params, cost, counter
    
    
def normalize(X: np.ndarray)->np.ndarray:
    '''
        Takes an numpy array with observations as rows 
        and returns a normalized version of the array, i.e.
        with 0 mean and std equal to 1
    
    '''
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std==0] = 1
    return (X - mean)/std, mean, std
    
    


if __name__ == '__main__':
        
    '''
        Create some artificial data
    '''
    num_obs = 500
    num_features = 4
    size = (num_obs, num_features)
    
    scalars = draw_from_cauchy((num_features,1))
    intercept = float(draw_from_cauchy((1,1)))

    
    X = generate_data(size, 
                      lambda x: draw_from_cauchy(x, 0, 20), 
                      scalars, 
                      intercept,
                      lambda x: draw_from_normal((num_obs,1),0, 5)
                      )
    
    
    
    '''
        First I try to use thec losed formula to fir scalars from data
    '''
    
    # since we want to consider intercept we will add a column of ones
    y = X[:, [-1]]
    X = np.hstack(( X[:,:-1], np.ones((num_obs,1))))
    
    fitted_params = regression_forumla(X,y)
    scalars_hat = fitted_params[:-1,:]
    interecept_hat = fitted_params[-1,:]
    
    print ('Parameters fiited with regression formula: \n', [el for el in fitted_params[:,0]],'\n')  
    print ('True parameters used to generate data: \n', list(scalars.reshape(num_features,)) + [intercept], '\n')
    print('The R squaared goodness of fit measuere is: ', r_squared(X, y, fitted_params), '\n')
    print('The RMSE is : ', rmse(X, y, fitted_params), '\n')
    
    
    
    
    '''
        Now we fit the parameters using gradient decent. GD in the case is
        very sensitive to the learning rate. Very often the algorithm may diverge,
        so we first normalize to mitigat this problem
    '''
    print('----------------------------------------------------------------')
    

    X_normalized, mean_X, std_X = normalize(X)
    X_normalized[:,-1]=1 # leave the intercept
    y_normalized, mean_y, std_y = normalize(y)
    
    
    GD_params, cost, counter = GD(X_normalized, y_normalized, lr=0.01, epsilon=0.0001, max_iter=10000)
    
    
    '''
        After fitting parameters on the normalized data set, I need to apply 
        transformations in order to retrieve parameters in the original scale
        This is straightforward algebra, although it does requires some thought
    '''
    
    GD_params_base = (GD_params[:-1,:] * std_y) * (1/std_X[:,:-1].T)
    GD_intercept = GD_params[-1,:]*std_y + mean_y - np.dot( mean_X[:,:-1],GD_params_base)
    
    GD_params  = np.vstack((GD_params_base, GD_intercept))
    

    
    print('Final cost after learning is ', cost, '\n')
    print('Number of iterations', counter, '\n')
    print ('Parameters fiited with gradient decent: \n', [el for el in GD_params[:,0]],'\n')
    print ('True parameters used to generate data: \n', list(scalars.reshape(num_features,)) + [intercept], '\n')
    print('The R squaared goodness of fit measuere is: ', r_squared(X, y, GD_params),'\n')
    print('The RMSE is : ', rmse(X, y, fitted_params), '\n')
    
    
    
    
    
    
    
    
    
    
    
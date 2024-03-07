"""
Objectives:
     - Load the MNIST data set. I have downloaded it in csv files from KAGGLE
     - implement from scratch a feed forward nueral network with ReLU and 
     softmax activation on the output layer
     - implement back propagation algorithm 
     - train the network so that it achieves more at least 90% accurracy
    
"""

import numpy as np
from matplotlib import pyplot as plt


def get_mnist_data()->tuple:
    '''
        Loading MNIST data from file
        Returns 4 data sets: X_train, y_train, X_test and y_test
        Each column represents an observation. Rows are features (pixels)
    '''
    print('Loading mnist data....')
    mnist_test = np.genfromtxt('mnist_test.csv', delimiter=',', skip_header=1)
    mnist_train = np.genfromtxt('mnist_train.csv', delimiter=',', skip_header=1)
    
    X_train, y_train = mnist_train[:,1:].T , mnist_train[:, [0]].T
    X_test, y_test = mnist_test[:,1:].T, mnist_test[:, [0]].T
    
    print('Done loading!')
    return X_train, y_train, X_test, y_test
    

def normalize(X: np.ndarray)->np.ndarray:
    '''
        Takes a numpuy array with obervations in columns 
        and returns a normalized transformation of X
    '''
    mean_values = np.mean(X, axis=1, keepdims=True)
    std_values = np.std(X, axis=1, keepdims=True)

    std_values[std_values == 0] = 1.0 #we need this to avod division by zero

    normalized_X = (X - mean_values) / std_values
    return normalized_X
    


def plot_image(x: np.ndarray)->None:
    '''
        Plots the image of a the digit represented by a (784,1) numpy vector
    '''
    img = x.reshape(28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
    
    
def relu(x: np.ndarray)->np.ndarray:
    '''
        Returns the relu activation of x
    '''
    return np.vectorize(lambda y: max(0,y))(x)


def grad_relu(x: np.array)->np.array:
    '''
        Returns the gradient of the relu function at x
    '''
    return np.vectorize(lambda y: 1 if y>0 else 0)(x)


def sigmoid(x: np.array)->np.array:
    '''
        Returns the signoid activation of a numpy array or float
    '''
    return np.vectorize(lambda y: 1/(1 + np.exp(-y)))(x)


def grad_sigmoid(x: np.array)->np.array:
    '''
        Returns the gradient of the sigmoid function at x
    '''
    return np.vectorize(lambda y: sigmoid(y)*(1-sigmoid(y)))(x)


def softmax(x: np.array)->np.array:
    '''
        Returns the softmax activation of a numpy array x 
    '''
    stable_x =  x - np.max(x,axis=0) # for numerical stability
    return np.exp(stable_x)/sum(np.exp(stable_x))



def cross_entropy(X: np.array ,Y:np.array, epsilon=0.00000000001)->float:
    '''
        Returns the cross entropy cost of X and Y
        X :  numpy array of size (10, n). Each column of X gives probabilities ofdgiits from 0 to 9
        Y : numpy array of size (1,n) - each column is the true digit
        
        If a column k of X assigns probabilty one to the corresponfing label in Y
        then the cost will be zero. Otherwise there will be cost
        The epsilon is just for numerical stabikity. To not to take logs of 0.
    '''
    
    Y_one_hot = np.zeros((10,Y.shape[1]))
    for j in range(Y.shape[1]):
        Y_one_hot[int(Y[0,j]),j] = 1

    assert X.shape == Y_one_hot.shape
    return - np.sum(Y_one_hot * np.log(X + epsilon))


def grad_cross_entropy(x: np.array, y: np.array)->float:
    '''
        Returns the dL/dZ - the output layer error:
        d'Cross_entropy_cost(softmax_activation(z values of last layer))/d'z values of last layer
        So this is not really a gradient of cross entropy loss
        The formula is trivial x-y, but the derivation is not trivial
        
    '''
    y_one_hot = np.zeros((10,1))
    y_one_hot[int(y),0] = 1
    assert x.shape == y_one_hot.shape
    return x - y_one_hot
    
    
    
class NN(object):
    '''
        Class NN captures the dense feed forwards nueral network object. This includes the size
        of the input, the nummber of neurons in follwoing layes, the size of the outpu
        and also the activattions and their gradients.
    '''
    
    def __init__(self, layers: list, activations: list)-> object:
        '''
             Creates an NN object   
        Parameters
        ----------
        layers : list. Pass a list of ints such as [784, 32, 10], which means
        that the input size is 784, one hidden layer has 32 neurons and a vectors
        of 10,1 is output by the network
        
        activations : list. This is a list of tuples of size two, with names of 
        activation functions and their gradients. These functions must be implemented
        outside the network. Eg: [(relu, grad_relu), (softmax, cross_entropy)]
    
        '''
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.z_vals = []
        self.a_vals = []
  
        # initiate weights with random normal smaples:
        for i in range(len(layers)-1):    
            self.weights.append(np.random.randn(layers[i+1], layers[i])*(1/layers[i+1]))
            self.biases.append(np.random.randn(layers[i+1],1)*(1/layers[i+1]))
            
    
    def forward(self, x: np.ndarray, learning = True)->np.ndarray:
        '''
            Method performs the forward pass of the network, stores the
            intermediary z and a values and returns the final output
    
            x: a vector that will  be processed through the network
            learning: If TRUE, the all intermediary values self.z_vals and
            self.a_vals will me stored to be used by backpropagation
        
        '''
        if learning:
            self.z_vals = []
            self.a_vals = []

        output = x
        
        for W, activation, b in zip(self.weights, self.activations, self.biases):
            z = np.matmul(W, output) + b
            if learning:
                self.z_vals.append(z)
            output = activation[0](np.matmul(W, output) + b)
            if learning:
                self.a_vals.append(output)
        return output


    def backprop(self, x: np.ndarray, y: np.ndarray)->list:
        '''
            Based on one training example x,y this method computes the gradient
            of the loss function with respect to all the weights in the network
        
            x: input vecotr
            y: label corresponding to x
            
            Method returns a list of tuples of size two:
            [(gradient with respect to weigts in frst layer, gradient with respect to first layer biased),... ]
            The elements of each tuple are np.ndarray's
            
        '''
        
        self.forward(x)
        
        deltas = []
        grads = []
     
        # output layer error and gradients:
        dLdZ = self.activations[-1][1](self.a_vals[-1], y)
        deltas.append(dLdZ)
        
        dW = np.matmul(dLdZ, self.a_vals[-2].T)
        db = dLdZ
        
        grads.append((dW, db))
        
    
        # hidden layers - output and gradients:    
        layers = len(self.weights)    
        for layer in range(layers-1,0,-1 ):
            
            dLdZ  = np.matmul(self.weights[layer].T, deltas[-1]) * self.activations[layer-1][1](self.z_vals[layer-1])
            deltas.append(dLdZ)
            
            db = dLdZ
            if layer == 1:
                dW = np.matmul(dLdZ,x.T)
            else:
                dW = np.matmul(dLdZ, self.a_vals[layer-2].T)
            grads.append((dW, db))

        grads.reverse()
        return grads
        
        
                     
def evaluate(Network: NN, X_test: np.ndarray, y_test: np.ndarray)->float:
    '''
        Returns the accuracy of the Netowrk on test set
 
    Network :  object of class NN - a neural network
    X_test :  numpy array -  wehre colmns are images as vectors of size (784,1)
    y_test :  numpy array  -  of size (1, X.shape[1]) with test labels
    '''
    prediction = np.argmax(Network.forward(X_test, learning=False), axis=0).reshape(y_test.shape)
    return np.sum(prediction == y_test)/y_test.shape[1]

                

def train_network(Net: NN, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray, 
                  num_iters: int, 
                  batch_size: int, 
                  lr: float, 
                  gamma=0.9, 
                  num_display=100)->None:
    '''
    Trains the neural network in mini batches using stochastic gradient desent. 
    
    ----------
    Net : object of class NN - a neural network, 
    X_train : numpy array -  wehre colmns are images as vectors of size (784,1)
    y_train : numpy array  -  of size (1, X.shape[1]) with training labels
    num_iters : int - the number of weight updates that will be perforemed
    batch_size : int - the size of the mini batch. I calculate the gradients
    for each point in mini batch and take the average to update weights
    lr : float - learning rate for gradient decent
    gamma : float - the parameter for momentum method. Default set to 0.9
    num_display : int - every num_display  iteration I present the current cost
    

    '''
    init_cost  =  cross_entropy(Net.forward(X_train, learning=False), y_train)
    Z = np.vstack((X_train, y_train))
    
    # initatie momentums to zero
    Momentum = [[0,0] for _  in range(len(Net.layers) - 1)]
    
    
    for i in range(num_iters):
        
        grads  = None    
        # here I choose at random the mini batch
        indices = np.random.choice(Z.shape[1], size = batch_size, replace = False)
        
        for ind in indices:
            x = Z[:-1,[ind]]
            y = Z[[-1], [ind]]
            nabla = Net.backprop(x, y) # backprop each point in mini batch
            
            if grads is None:
                grads = nabla
            else: # adding grads up. 
                grads = [[g[0] + n[0], g[1] + n[1]]  for g, n in zip(grads, nabla)]

        #taking the average of grads over the mini batch. Grads are as a double list    
        grads = [[(1/batch_size) * el[0], (1/batch_size) * el[1]] for el in grads]
            
            
        # updatting the weights of the Network
        for j in range(len(Net.layers)-1): # updating weights of the network:
            Momentum[j][0] = gamma * (Momentum[j][0] - grads[j][0]) + grads[j][0]
            Momentum[j][1] = gamma * (Momentum[j][1] - grads[j][1]) + grads[j][1]
            
            Net.weights[j] = Net.weights[j] - lr * Momentum[j][0]
            Net.biases[j] = Net.biases[j] - lr * Momentum[j][1]
            
    
        if i % num_display == 0: # calculate and display the cost evenry num_display:
            cost = cross_entropy(Net.forward(X_train, learning=False), y_train)
            print('Training iteration: ', i, '/', num_iters, '  Current cost: ', round(cost,0) )
        
            # learning rate decay. Although momentum should take care of this, I do include a manual decay:
            if cost > init_cost:
                lr = 0.95 * lr
            init_cost = cost
        
        

if __name__ =='__main__':
    
    '''
    Get the MNIST data, I transpose, as I want to have observations stacked 
    horizontaly, so that each column represents an input vector.Rows represent
    features. 
    
    '''
    
    X_train, y_train, X_test, y_test = get_mnist_data()
    
    
    ''' plot 4 random images, just for fun'''
    for i in np.random.choice(range(X_train.shape[1]), size=4, replace=False):
        x = X_train[:,[i]]
        plot_image(x)
    
    
    ''' Normalize before training '''
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    

    '''
        Initiate the neural network with two hidden layers with 64 and 16 neurons
        and check how it performs prior to training .Intuitively, prior to 
        trainng the nework shold have an accuracy of around 10% 
        (chance of randomly guessing an image). After trainng 
        we expect around 95% accuracy.
    '''

    Net = NN([X_train.shape[0], 64, 16, 10] , [(relu, grad_relu),(relu, grad_relu), (softmax, grad_cross_entropy)])

    
    print('Accuracy before training: ', evaluate(Net, X_test, y_test))
   
    
    '''
        Now lets train the network and see what is the final accuracy. Here I 
        provide a few paramers that seem to work decently
    '''
    
    gamma = 0.9 # momentum, moving average parameter
    lr = 0.01 # learnng rate
    batch_size = 50 # this is the mini batch size
    num_iters = 30000  # for 30 k times I will update weights

    train_network(Net, X_train, y_train, num_iters, batch_size, lr, gamma=0.9, num_display=1000)
    print('Accuracy after training: ', evaluate(Net, X_test, y_test)) 
    
    
    '''
        We can also compare how would the training go for the same
        architecture but with sigmoid activations.This howeever takes much more 
        time to train.
    '''
    
    Net_sigmoid = NN([X_train.shape[0], 64, 16, 10] , [(sigmoid, grad_sigmoid), (relu, grad_relu), (softmax, grad_cross_entropy)])
    
    print('Accuracy before training of net with sigmoid activatios: ', evaluate(Net_sigmoid, X_test, y_test))
    train_network(Net_sigmoid, X_train, y_train, num_iters, batch_size, lr, gamma=0.9, num_display=1000)
    print('Accuracy after training of net with sigmoid activations: ', evaluate(Net, X_test, y_test)) 
    
    '''
        Finally let's see if we have only one hidden layer. Would that make a difference?
        
    '''
    
    Net_simple = NN([X_train.shape[0], 64, 10] , [(relu, grad_relu),(softmax, grad_cross_entropy)])
    train_network(Net_simple, X_train, y_train, num_iters, batch_size, lr, gamma=0.9, num_display=1000)
    print('Accuracy of simple network after training: ', evaluate(Net_simple, X_test, y_test)) 
    
    


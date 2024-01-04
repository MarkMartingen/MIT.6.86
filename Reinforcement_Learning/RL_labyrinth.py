# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:06:05 2023
@author: m.basiuk

Objectives:
    
     - implement a Q-learning algorithm that will find the best way out of a labyrinth
     - The labirinth will be given by an nxm grid.
     - Some states will be terminal or inaccessible. 
     - The temrinal states will come with a reward
     - there will be a negatove reward asociated with every move
     - there will be a maximum number of moves
     - moves will be allowed around the grid aprat from inaccessible states and
     there wilb no randomness with the moves. 
     
     - Given a labyrinthm the agent will learn over a number of epchs. 
     Each epoch will ocnsist of a number of train runs and test runs.
     - The rewards over epchs will be plotted
"""
import numpy as np
import random
from matplotlib import pyplot as plt
import copy


class Labyrinth(object):
    '''
        Labirynth is a two dimensional grid represented by a numpy array.
        Each entry can be either np.nan - not accessible, 0 - accessible ,
        1 accessible and the agent being in that particular state
        
        There is also rewards numpy array, and every time the agent enter 
        a particular state he gets a corresponding reward.
        
        Finally there is also a list of terimnal states.
    '''
    def __init__(self, states, rewards, terminal_states):
        self.states = states # 2-dim array representing the labyrinth.
        self.rewards = rewards
        self.terminal_states = terminal_states

        assert np.nansum(self.states) == 1 #exactly one state must be occupied
        assert self.rewards.shape == self.states.shape
    
    def get_current_state(self):
        ''' gets the i,j coordinates of the state in which the agent is '''
        return np.unravel_index(np.nanargmax(self.states), self.states.shape)
    
    
    def get_actions(self):
        ''' Reutrns a list of possible actions corresponding to the current state '''
        # current state:
        i,j = self.get_current_state()
    
        actions = []
        
        for ind in [(i,j+1), (i,j-1), (i-1,j) , (i+1,j)]:
            if  0 <= ind[0] < self.states.shape[0] and 0 <= ind[1] < self.states.shape[1]:
                if not np.isnan(self.states[ind]): # cannot move into inaccessible
                    actions.append(ind)                
        return actions


    def take_action(self, action):
        ''' Takes a valid action , changes the the state of the agent by changing 
        self.states and returns the associated reward
        
        '''
        # safety checks:
        assert  0 <= action[0] <= self.states.shape[0] and 0 <= action[1] <= self.states.shape[1]
        assert  not np.isnan(self.states[action])
        
        i,j = self.get_current_state()
        
        # change the state of the agent
        self.states[i,j] = 0
        self.states[action]=1 
        
        return self.rewards[self.get_current_state()] 
        
        
    def set_state(self, new_states):
        ''' Sets the states. Used to 
        '''
        self.states = new_states
        
    
    
def run(Board: Labyrinth, epsilon: float, update: bool, gamma: float, learning_rate: float=0.2, max_steps: int=30)->tuple:
    """
    Function makes one interaction of agent with baord using the epsilon greedy
    approach. If update is True it will update a gloab Q table. 
    
    Paramters:
        Bard: instance of Labyrinth class
        epsilon: float in [0,1] probability that the agent will draw action at random
        update: bool if True uodates the Q value 
        gamma: discount factor
        larning rate: hyperparameter for updateing the Q table
        max_steps: maximum numner of iterations
        
    Returns:
        trajectory: as string representing the moves - up, down, left, right in sequence
        total_rewards: the total discounted reward after the interaction
        
    """
    is_terminal = (Board.get_current_state() in Board.terminal_states)
    num = 0
    total_reward = 0
    trajectory = str(Board.get_current_state())
    
    while not is_terminal and num < max_steps:
        
        actions = Board.get_actions() 
        state = Board.get_current_state()
    
        a_ind = np.argmax([Q[state + action] for action in actions])
        
        if epsilon > random.random():
            a = random.choice(actions)
        else:
            a = actions[a_ind]
        
        # take action here:
        reward = Board.take_action(a)
        total_reward += reward * gamma**(num)
        
        # get new state and check if it is terminal:
        new_state = Board.get_current_state()
        is_terminal = (new_state in Board.terminal_states)
       
        # if update then update Q using the Bellman equations:
        if update:
            if is_terminal:
                R = reward
            else:
                new_actions = Board.get_actions() 
                a_ind = np.argmax([Q[new_state + action] for action in new_actions])
                R = reward + gamma * Q[new_state + new_actions[a_ind]]
             
            Q[state + a] = (1 - learning_rate) * Q[state + a] + learning_rate * R
            
        
        num += 1
        # Code to print out the trajectory of the agent:
        if new_state[0] < state[0]:
            direction = 'up'    
        elif new_state[0] > state[0]:
            direction = 'down'    
        elif new_state[1] < state[1]:
            direction = 'left'
        elif new_state[1] > state[1]:
                direction = 'right'
        else:
            direction = 'wrong'
        # trajectory = trajectory + '-->' + str(new_state) #str(Board.get_current_state()) #+ '/' + str(Board.rewards[Board.get_current_state()])
        trajectory = trajectory + '-->' + direction
    return trajectory, total_reward


def run_epochs(Board: Labyrinth, num_epochs: int, num_train: int, num_test: int, \
     epsilon_train: float, epsilon_test: float, gamma: float, learning_rate: float)->list:
    """
        Runs num_epochs experiments. Each experiment consists of a train batch
        and test batch of interactions of agent with the Board (Labyrinth).
        During the train batch the agent trains running  num_train interactions.
        
        In the test batch, the agent interacts for num_test times and his 
        average reward, obtained in the test batch is returned.
        
        Finall the average rewards from each test batch per eopch is plotted
        against the number of epochs.

        The function returns a list of average rewards - one per each epoch
    """
    
    init_states = Board.states    
    test_rewards = []
    
    for epoch in range(num_epochs):
        
        for _ in range(num_train):
            Board.set_state(copy.deepcopy(init_states))
            trajectory, total_reward = run(Board, epsilon_train, True, gamma, learning_rate)
            
        rewards = 0 # prepare for calculcating average reward in the test batch
        for _ in range(num_test):
            Board.set_state(copy.deepcopy(init_states))
            trajectory, total_reward = run(Board, epsilon_test, False, gamma, learning_rate)
            # print(total_reward)
            # print(trajectory)
            rewards += total_reward
        
        test_rewards.append(rewards / num_test)
    
    plt.plot(range(num_epochs), test_rewards)
    plt.xlabel('numnber of epoch')
    plt.ylabel('Avergae reward in epoch on test batch')
    plt.title('Finding the way out of a labyrinth')
    plt.show()
    
    return(test_rewards)
    
    
if __name__ =='__main__':
    '''
        First take a simple labirynth. s denotes start state, X - unaccessible,
        and numbers representing rewards are in terminal states. Also every move
        costs 0.05 (rewards is -0.05)
            
        --------------------------
        |    |    |    |    | +1 |
        --------------------------
        |    |  X |  X |    | -1 |
        --------------------------
        |    |    |    |    |  S |
        --------------------------
        
        So the the agent should take the route left->up->up->right
        
        I will train the agent in a number of epochs. Each epoch will consists
        of a few train rubs and aftwerwards a few test runs. For each epoch I 
        take the average rewards of the test batch and plot the results to see
        how the algorithm converges
        
        
   '''
    # initiate the Board/Labirynth:
    init_states = np.array([[0,0,0,0,0], [0, np.nan, np.nan, 0,0], [0,0,0,0,1]])
    
    rewards = -0.05 *np.ones(init_states.shape)
    rewards[0,4] = 1
    rewards[1,4] = -1
    terminal_states = [(0,4), (1,4)]
    Board = Labyrinth(init_states, rewards, terminal_states)

    # intiializie the Q table. THere is some redundacy, as my Q table 
    # has placeholders for every state,action pair even if action is illegal in 
    # a certain state, but this allows for concise code
    global Q
    Q = np.zeros((Board.states.shape+Board.states.shape))

    # execute the experiment:
    num_epochs = 50
    train_batch = 5
    test_batch = 10
    train_epsilon = 0.5
    test_epsilon = 0.025
    gamma = 0.9
    learning_rate = 0.2
    
    rewards = run_epochs(Board, num_epochs, train_batch, test_batch, \
        train_epsilon, test_epsilon, gamma , learning_rate)
        

    '''
    I also want to see, if setting epsilon to zero after training will result
    in the expecter trajectory:
    '''
    
    Board.set_state(copy.deepcopy(init_states))    
    trajectory, rewward = run(Board, 0, False, gamma)
    print('Simple case trajectory: ', trajectory)


    '''
        Now lets take a slightly more sophisicated labirynth
        
        Again we run the experiment as a number of epochs, each with a train and
        test batch, plot the avergae rewards in test batch for each epoch.
        
        The same algorithm should find it's way independet of the geometry of the 
        labirynth
        
        -------------------------------
        |  S | -1 | +5 |    |    | -1 |
        -------------------------------
        |    |  X | -1 |  X |    |    |
        -------------------------------
        |    |    |    |    |  X |    |
        -------------------------------
        |    |  X |  X | -1 |    |    |
        -------------------------------
        |    |    |    |    |  X |    |
        -------------------------------
        |    |  X |    |    |    |    |
        -------------------------------
        
    There is more than one optimal route. Also in this case I increase the 
    batch sizes to have quicker convergence with the number of epochs
    '''
    # initialize Board:
    init_states = np.array([[1,0,0,0,0,0], [0, np.nan, 0,np.nan, 0,0], \
                            [0,0,0,0,np.nan, 0],[0,np.nan,np.nan, 0,0,0],\
                                [0,0,0,0,np.nan,0],[0,np.nan, 0,0,0,0]])
    rewards =  -0.05* np.ones(init_states.shape)
    rewards[0,1]=-1
    rewards[0,5]=-1
    rewards[1,2]=-1
    rewards[3,4]=-1
    rewards[0,2]=5 # big reward in one terminal state
    terminal_states = [(0,1), (0,5),(1,2),(3,4),(0,2)]
    Board = Labyrinth(init_states, rewards, terminal_states)

    # set Q table to zeros:
    Q = np.zeros((Board.states.shape+Board.states.shape))
    
    
    # run the experiment:
        
    num_epochs = 50
    train_batch = 10
    test_batch = 10
    train_epsilon = 0.5
    test_epsilon = 0.02
    gamma = 0.9
    learning_rate = 0.2
    
    rewards = run_epochs(Board, num_epochs, train_batch, test_batch, \
        train_epsilon, test_epsilon, gamma , learning_rate)
        

    '''
    AS before I also print out the learned route, which one can check that 
    it is indeed the optimal one:
    '''
    Board.set_state(copy.deepcopy(init_states))    
    trajectory, rewward = run(Board, 0, False, 0.9)
    print('Harder  case trajectory: ', trajectory)

    
    





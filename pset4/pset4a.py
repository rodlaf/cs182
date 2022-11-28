# -*- coding: utf-8 -*-
"""
CS 182 Pset4: Python Coding Questions - Fall 2022
Due November 28, 2022 at 11:59pm
"""


#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset4.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


######################################
# Question 2: Reinforcement Learning #
######################################

### Package Imports ###
from os import system
from time import sleep
import gym 
import numpy as np
### Package Imports ###

# Instructions: In this problem, you will be implementing various planning and reinforcement learning algorithms on OpenAI's 
# Frozen Lake Environment. You will need the packages gym==0.21.0, IPython==7.29.0, and matplotlib==3.4.3. In this environment,
# an agent controls the movement of a character in a 4x4 grid world. Some tiles of the grid are walkable S, for start, F, for 
# frozen, G, for goal), and others lead to the agent falling into the water (H, for hole). The agent is rewarded +1 for reaching 
# the goal state and 0 otherwise. We will work with a few variations of the Frozen Lake environment. In the first version, the 
# parameter is_slippery is set to False, so every action leads to a deterministic tile. When is_slippery is set to True, the 
# movement direction of the agent is uncertain. In particular, if an agent chooses to go in one direction, there is a 1/3 
# probability the agent goes in the intended direction and a 1/3 probability that the agent goes in each of the directions that 
# are perpendicular to the intended direction. If an agent is on the edge of the map and attempts to move off the map, it simply 
# stays in place.

# Part 2: Implement value iteration in by filling out the method value_iteration within the class DynamicProgramming. You may find
# updated_action_values to be a useful helper function when writing this.

# Part 3: Report the mean and variance of the rewards over 1000 episodes of the final policy using the parameters gamma=0.9, 
# epsilon=0.001. For an agent using the policy found by value iteration, plot a histogram of the number of steps it takes an agent 
# to reach the goal over those 1000 episodes using the final policy. If the agent falls into a hole in the ice and never reaches the 
# goal state, let that be recorded as a zero. Does this agent always reach the goal? Why or why not? Use the map to inform your 
# explanation.

# Part 4: Implement active, model free reinforcement learning in the form of Q-learning by filling out the functions choose_action and 
# q_learning within the class QLearning. Use alpha(k_{sa}) = min(0.1, 10 k_{sa}^{-0.8}).

# Part 5: Plot the mean returns over 100 episodes of the Q-learning agent that acts solely based on max-Q values after every 1000 
# episodes (this should be done by using the compute_episode_rewards function). Use the parameters gamma|=0.9, epsilon=0.01. How 
# does your Q-learning agent compare to the value-iteration agent following the policy derived from part 3?

# Do not use any external libraries other than the ones that have already been imported above.

# ------------------------- #
#----- SAMPLE SOLUTION -----#
# ------------------------- #

State = int
Action = int

class DynamicProgramming:
    """
    Write your algorithms for Policy Iteration and Value Iteration in the appropriate functions below.
    """

    def __init__(self, env, gamma=0.95, epsilon=0.001):
        '''
        Initialize policy, environment, value table (V), policy (policy), and transition matrix (P)
        '''
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = self.create_initial_policy()
        self.V = np.zeros(self.num_states) # Vector of estimated utilities for each state, states are numbered 0-15, initialized at zero
        self.P = self.env.P # Dict of a Dict of list of tuples. Outer dict keys = states, 0-15, index into a state, you get a dict of 4 actions that can be performed, for each action we have
        # a list of possiable states that you can end up in, then it's a tuple that tells us what prob of ending up in each of those states, the state indexer, the reward of doing that, finally
        # a boolean indicator of if its a terminal state or not
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Convergence parameter
        self.rewards = {state:0 for state in range(self.num_states)} # The reward function R(s)
        self.terminal_states = {state:False for state in range(self.num_states)} # Returns a True or False indicating if the game state provided 
        # is in a terminal state (i.e. no further actions can be taken)
        
        for state,actions in env.P.items():
            for action,action_data in actions.items():
                prob, state, r, is_end = action_data[0]
                if is_end==True:
                    self.terminal_states[state] = True # Record if terminal state (ice hole or goal)
                    if r == 1:
                        self.rewards[state] = 1 # If a goal state, then R(s) = 1, else R(s) left at 0
                
    def create_initial_policy(self):
        '''
        A policy is a numpy array of length self.num_states where
        self.policy[state] = action

        You are welcome to modify this function to test out the performance of different policies.
        '''
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    def updated_action_values(self, state: State) -> np.ndarray:
        """
        This is a useful helper function for implementing value_iteration.
        Given a state (given by index), returns a numpy array of entries
        
        Z_i = SUM[p(s'|s,a_i)*U(s')] over all s' for action a_i
        
        i.e. return a np.array: [Z_1, ..., Z_n]
        
        based on current value function self.V for U(s')
        """
        ##########################
        ##### YOUR CODE HERE #####
        ##########################
        Z = np.zeros(self.num_actions)

        for action in range(self.num_actions):
            for i in range(len(self.P[state][action])):
                prob, next_state, reward, done = self.P[state][action][i]
                Z[action] += prob * self.V[next_state]

        return Z
    

    def value_iteration(self):
        """
        Perform value iteration to compute the value of every state under the optimal policy.
        This method does not return anything. After calling this method, self.V should contain the
        correct values for each state. Additionally, self.policy should be an array that contains
        the optimal policy, where policies are encoded as indicated in the `create_initial_policy` docstring.
        """
        ##########################
        ##### YOUR CODE HERE #####
        ##########################

        # Compute values
        while True:
            new_V = np.zeros(self.num_states)

            for state in range(self.num_states):
                new_V[state] = self.rewards[state] + self.gamma * np.max(self.updated_action_values(state))

            if max([abs(new_V[i] - self.V[i]) for i in range(self.num_states)]) < self.epsilon * (1 - self.gamma) / self.gamma:
                break

            self.V = new_V
        print(self.V)

        # Extract optimal policy
        for state in range(self.num_states):
            self.policy[state] = np.argmax(self.updated_action_values(state))

        print(self.policy)

            
    def play_game(self, display=False):
        '''
        Play through one episode of the game under the current policy
        display=True results in displaying the current policy performed on a randomly generated environment in the terminal.
        '''
        self.env.reset()
        episodes = []
        finished = False

        curr_state = self.env.s
        total_reward = 0

        while not finished:
            # display current state
            if display:
                system('cls')
                self.env.render()
                sleep(0.1)

            # find next state
            action = self.policy[curr_state]
            try:
                new_state, reward, finished, info = self.env.step(action)
            except:
                new_state, reward, finished, info, _ = self.env.step(action)
            reward = self.rewards[new_state] # Rewards are realized by entering a new state
            total_reward += reward
            episodes.append([new_state, action, reward])
            curr_state = new_state # Set the current state equal to the new state for the next while loop iteration

        # display end result
        if display:
            system('cls')
            self.env.render()

        print(f"Total Reward from this run: {total_reward}")
        return episodes

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes);steps_taken = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                try:
                    new_state, reward, finished, info = self.env.step(action)
                except:
                    new_state, reward, finished, info, _ = self.env.step(action)
                reward = self.rewards[new_state]
                total_rewards[episode] += reward
                num_steps += 1
                curr_state = new_state # Set the current state equal to the new state for the next while loop iteration
                

            if reward != 0:
                # If the agent falls into a hole and gets a reward of zero, then record that as zero (already the value of the array)
                # Otherwise, if they do eveutally reach the goal, then record then number of steps taken
                steps_taken[episode] = num_steps

        return np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards), steps_taken


    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        '''
        Prints information from compute_episode_rewards
        '''
        mean, var, best, steps_array = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean:.2f}, Variance of Episode Rewards: {var:.2f}, Best Episode Reward: {best}")

# Model free reinforcement learning
class QLearning:
    """
    Write you algorithm for active model-free Q-learning in the appropriate functions below.
    """

    def __init__(self, env, gamma=0.95, epsilon=0.01):
        """
        Initialize policy, environment, and Q table (Q)
        """
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n 
        self.Q = np.zeros((self.num_states, self.num_actions)) # A 2d-array of states and the values of each action
        self.state_action_counter = np.zeros((self.num_states, self.num_actions))   # keeps track of k_sa
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploration rate
        self.rewards = {state:0 for state in range(self.num_states)} # The reward function R(s)
        self.terminal_states = {state:False for state in range(self.num_states)} # Returns a True or False indicating if the game state provided 
        # is in a terminal state (i.e. no further actions can be taken)
        
        for state,actions in env.P.items():
            for action,action_data in actions.items():
                prob, state, r, is_end = action_data[0]
                if is_end==True:
                    self.terminal_states[state] = True # Record if terminal state (ice hole or goal)
                    if r == 1:
                        self.rewards[state] = 1 # If a goal state, then R(s) = 1, else R(s) left at 0


    def choose_action(self, state: State) -> Action:
        """
        Returns action based on Q-values using the epsilon-greedy exploration strategy
        """
        ##########################
        ##### YOUR CODE HERE #####
        ##########################
        raise NotImplementedError

    def q_learning(self, num_episodes=10000, interval=1000, display=False, step_limit=10000):
        """
        Implement the tabular update for the table of Q-values, stored in self.Q

        Boilerplate code of running several episodes and retrieving the (s, a, r, s') transitions has already been done
        for you.
        """
        mean_returns = []
        for e in range(1, num_episodes+1):
            self.env.reset()
            finished = False

            curr_state = self.env.s
            num_steps = 0

            while not finished and num_steps < step_limit:
                # display current state
                if display:
                    system('cls')
                    self.env.render()
                    sleep(1)

                action = self.choose_action(curr_state)
                try:
                    next_state, reward, finished, info = self.env.step(action)
                except:
                    next_state, reward, finished, info, _ = self.env.step(action)
                
                reward = self.rewards[next_state]
                
                # Update the state_action_counter
                ##########################
                ##### YOUR CODE HERE #####
                ##########################
                raise NotImplementedError
                
                # update Q values. Use the alpha schedule given here. k_SA = how many time we took action A at state S
                alpha = min(0.1, 10 / self.state_action_counter[curr_state][action] ** 0.8)
                
                # Q-learning update rule
                ##########################
                ##### YOUR CODE HERE #####
                ##########################
                raise NotImplementedError
                
                num_steps += 1
                curr_state = next_state

            # run tests every interval episodes
            if e % interval == 0:
                print(str(e)+"/"+str(num_episodes),end=" ")
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)

        return mean_returns

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                best_actions = np.argwhere(self.Q[curr_state] == np.amax(self.Q[curr_state])).flatten()
                action = np.random.choice(best_actions)
                try:
                    next_state, reward, finished, info = self.env.step(action)
                except:
                    next_state, reward, finished, info, _ = self.env.step(action)
                reward = self.rewards[next_state]
                total_rewards[episode] += reward
                num_steps += 1
                curr_state = next_state

        mean, var, best = np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)
        print(f"Mean of Episode Rewards: {mean:.2f}, Variance of Episode Rewards: {var:.2f}, Best Episode Reward: {best}")
        return mean, var, best


if __name__ == "__main__":
    
    ### Part 3 - Value Iteration ###
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True) # Set up the Frozen lake environmnet 
    env.reset()

    print("Testing Value Iteration...")
    sleep(1)
    my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.001) # Instantiate class object
    my_policy.value_iteration() # Iterate to derive the final policy
    my_policy.play_game() # Play through one episode of the game under the current policy
    my_policy.print_rewards_info() # Prints information from compute_episode_rewards
    sleep(1)

    # Compute the episode rewards over 1000 episodes of game playing
    mean_val, var_val, max_val, num_steps_array =  my_policy.compute_episode_rewards(num_episodes=1000, step_limit=1000)
    print(f"Mean of Episode Rewards: {mean_val:.2f}, Variance of Episode Rewards: {var_val:.2f}, Best Episode Reward: {max_val}")
    
    import matplotlib.pyplot as plt
    
    ##########################
    ##### YOUR CODE HERE #####
    ##########################
    raise NotImplementedError
    
    
    ### Part 4 - Model Free Q-Learning ###
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    print("Testing Q-Learning...")
    sleep(1)
    my_policy = QLearning(env, gamma=0.9, epsilon=0.01) # Instanciate a new class object with the Q learning methods
    
    ### Part 5 ###
    
    # Plot the mean returns over 100 episodes of the Q-learning agent that acts solely based on max-Q values after 
    # every 1000 episodes (this should be done by using the compute_episode_rewards function). Use the parameters 
    # gamma=0.9, epsilon=0.01. How does your Q-learning agent compare to the value-iteration agent following the 
    # policy derived from part 3?
    
    ##########################
    ##### YOUR CODE HERE #####
    ##########################
    raise NotImplementedError



#####################################
# Question 4.1: Logistic Regression #
#####################################

### Package Imports ###
import pandas as pd
from typing import Tuple
### Package Imports ###

if __name__ == "__main__":
    
    ### Package Imports ###
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    ### Package Imports ###
    
    ## Starter-code DO NOT EDIT
    iris = datasets.load_iris() # Load the dataset from sklearn
    X = pd.DataFrame(iris.data) # Convery to a pandas dataframe
    X.columns = pd.Series(iris.feature_names).str.replace(" (cm)","",regex=False).str.replace(" ","_") # Add col headers
    X = X.loc[:,["petal_length","petal_width"]] # Subset for only the petal features
    y = iris.target # Extract the response variable
    X = X.loc[y!=0,:];y = y[y!=0] # Remove one of the classes, the one labeled 0 for this data analysis
    y = y-1 # Adjust labels to be 0 and 1 instead of 1 and 2 in the response variable
    
    
    # Instructions: Use the LogisticRegression module of the sklearn Python library to implement a logistic regression model on 
    # the Iris dataset. Starter code is provided in pset4.py. The Iris data set contains data about iris flowers. Each data point 
    # in the data set represent a separate flower examined. The X data describes the flower in terms of petal length and width. 
    # Our response vector (y) is a boolean vector of 0s and 1s indicating the true species classifications of these flowers into 
    # 2 distinct categories. Our goal is to build a model to classify iris flowers into these 2 species categories based on their 
    # petal dimensions alone. Our model will be both descriptive and predictive. 
    
    # Do not use any external libraries other than the ones that have already been imported above.
    
    # Part A: Plot this dataset with blue dots for y = 0 and orange dots for y = 1. Then run a logistic regression using the features 
    # x_1 = petal_length and x_2 = petal_width and plot the resulting decision boundary. Be sure to label your plot axes, include a legend 
    # and a title. Make sure to include your plot in your PDF write-up. 
    
    ##########################
    #### YOUR CODE HERE ####
    ##########################
    
    # Part B: What are the estimated coefficients and intercept? (include this in your PDF)
    
    ##########################
    #### YOUR CODE HERE ####
    ##########################
    
    # Part C: What is the in-sample accuracy of your model at distinguishing between these 2 plant types? What is the baseline accuracy 
    # that you'd achieve by simply choosing the majority class? Does this model provide a substantial improvement to that baseline?
    # (include this in your PDF)
    
    ##########################
    #### YOUR CODE HERE ####
    ##########################



######################################
# Question 4.2: Perceptron Algorithm #
######################################

# Toy data set for practice with the perceptron algorithm
x1 = [1,1,2,2,2,3,3] # First data set feature
x2 = [0,2,1,2,3,3,1] # Second data set feature
x0 = [1]*len(x1) # Create a vector of 1s to add a bias term
X = pd.DataFrame({"x0":x0, "x1":x1,"x2":x2}) # Create a pandas DF for the x with the bias term included
y = np.array([ 1,  1, -1, -1,  1, -1, -1]) # Response vector
weight_vector = np.array([1,1,0]) # In the form of (x0, x1, x2)


# Instructions: Implement the Perceptron algorithm with initial weight w = (1, 1, 0) using the toy data set contained in the starter 
# code above. Please include your code the pdf file and write down how many passes through the data it took for your algorithm to 
# converge. Does your algorithm converge when you add the point ((x_1, x_2), y) = ((2.5, 0), 1) to you dataset? Why or why not?
# Complete the functions below to create a perceptron algorithm model. First complete the perceptron_iter() helper function that 
# performs 1 iteration of the perceptron algorithm on 1 data point in the data set. Then complete the run_perceptron_algo() wrapper
# function that takes in the full X and y data set along with a weight vector and specified number of max iterations and runs the
# perceptron algorithm until either convergence is reached or the max iterations is exceeded.

# Note: While you will get the same convergance (if one exists) with enough iterations regardless of what starting weight vector is
# supplied, but in order to produce the same number of iterations in the fitting proceadure as the instructor solution, a starting 
# weight vector is provided to the fitting function.

# Do not use any external libraries other than the ones that have already been imported above.

def perceptron_iter(X:pd.Series, y:float, weight_vector:np.array)->Tuple[np.array,bool]:
    """Takes in an X pd.Series representing the feature vector for 1 obs, a y value of that obs, and a weight vector. Performs 
    1 iteration of the perceptron algorithm to update the weight vector using this data point from the broader dataset"""
    change=False # Also return a boolean value indicating if the weight vector has been changed from input to output

    ##########################
    ##### YOUR CODE HERE #####
    ##########################
    raise NotImplementedError
    
    return weight_vector, change

def run_perceptron_algo(X:pd.DataFrame, y:np.array, weight_vector:np.array, max_iter:int=1000)->Tuple[np.array,int]:
    """Takes in a dataset denoted by X and y, with a starting weight vector and runs the perceptron algorithm until convergence
    or a max iteration threshold has been exceeded, returns the fitted weight vector and the number of iterations required"""
    iterations=0
    
    # Hint: Call the perceptron_iter() helper function from above
    
    ##########################
    ##### YOUR CODE HERE #####
    ##########################
    raise NotImplementedError
    
    return weight_vector, iterations

if __name__ == "__main__":
    # Run the perceptron algorithm and print the results
    weight_vector, iterations = run_perceptron_algo(X, y, weight_vector)
    print("Iterations Until Convergence: "+str(iterations))
    print("Classifier Weights:\n"+str(weight_vector))
    
    # Add another observation and re-run
    X.loc[7,:]=[1,2.5,0] # Append a new row to the dataframe
    y=np.append(y,1) # Append a new response variable observation to the y-vector
    
    # Run the perceptron algorithm and print the results
    weight_vector, iterations = run_perceptron_algo(X, y, weight_vector)
    print("Iterations Until Convergence: "+str(iterations))
    print("Classifier Weights:\n"+str(weight_vector))
    
    # Plot the set of data points including the newly added point at (2.5, 1)
    plt.figure(figsize=(8,6)) # Configure plot size and add in background classification region shading
    colors = ['red' if term==-1 else 'blue' for term in y ] # Determine colors for data points based on y
    plt.scatter(X.iloc[:,1],X.iloc[:,2],color=colors) # Plot the originial data points
    plt.xlabel("X1");plt.ylabel("X2");plt.title("X1, X2 Feature Space with Added Point")
    plt.show()           



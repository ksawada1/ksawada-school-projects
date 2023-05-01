#!/usr/bin/env python3
#

import numpy as np 
import gymnasium as gym
from gym.wrappers import TimeLimit
import sys
import argparse
import logging
import os.path
import joblib
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import random
from mario_gym_env import GymEnv 

class QLeft:
    """
    Policy to always move left
    """
    def __init__(self):
        return

    def get_best_action(self, state):
        return 0

class QRight:
    """
    Policy to always move right
    """
    def __init__(self):
        return

    def get_best_action(self, state):
        return 1

class QRandom:
    """
    Policy to randomly choose left/right with equal probability
    """
    def __init__(self):
        return

    def get_best_action(self, state):
        return random.choice([0,1])

class QFunction:
    """
    Learnable policy function, where the state-space is continuous, but the action space is discrete.
    """

    def __init__(self, state_shape, n_actions):
        """
        state_shape:   tuple of integers; number of state variables in each dimension
        n_actions:     integer; number of actions
        """
        self.Q = self.create_Q_function(state_shape, n_actions)
        self.state_shape = state_shape
        self.num_actions = n_actions
        return

    def n_actions(self):
        """
        """
        return self.num_actions

    def actions(self):
        """
        """
        return [action for action in range(self.n_actions())]

    def create_Q_function(self, state_shape, n_actions):
        """
        state_shape:   tuple of integers; number of state variables in each dimension
        n_actions:     integer; number of actions
        
        
        For CartPole, a state has 4 floating point variables.

        Input shape is (4,), meaning a list (or 1-d tensor) with 4 items.

        There are 2 actions: left, right
        Output shape is [*, 2], meaning a 2-d tensor, with the first
        dimension used for the number of predictions, and the second
        dimension being the action number (0-1).
        """
        model = keras.models.Sequential()

        model.add(keras.layers.Input(shape=state_shape))
        model.add(keras.layers.Dense(32, activation="elu"))
        model.add(keras.layers.Dense(16, activation="elu"))
        model.add(keras.layers.Dense(8, activation="elu"))
        model.add(keras.layers.Dense(n_actions, activation="linear"))

        model.compile(loss="mse", optimizer=keras.optimizers.Adam())
        # print("model summary", model.summary())
        
        return model

    def update(self, state, action, next_state, reward, gamma, prediction, next_prediction, done):
        """
        state:           np.array of shape self.state_shape; the floating point values describing the current state
        action:          np.int64; action index
        next_state:      np.array of shape self.state_shape; the floating point values describing the next state
        reward:          float; immediate reward
        gamma:           float; future discount factor
        prediction:      np.array of shape (1, n_actions); expected value of actions in state
        next_prediction: np.array of shape (1, n_actions); expected value of actions in next_state
        done:            bool; whether the episode is done or not

        For CartPole:
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        action is an integer (0,1)
        next_state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        reward is a python float
        gamma is a python float
        prediction is a np.array of shape=[1,2]
        next_prediction is a np.array of shape=[1,2]
        done is a bool
        """
        # This is what we want the quality value for action in state to be.
        target_quality_value = reward + (1.0-done) * gamma * np.max(next_prediction)
        # target_quality_value is a numpy float

        # Change for action's value to be target
        target_vec = prediction
        target_vec[0][action] = target_quality_value

        # converts the single state into an array with 1 state
        state = state.reshape(-1, state.shape[0])
        # # cause the state to be a list of 1 state, because the fit() method needs lists of inputs and target outputs
        # state = np.array([state])
        
        # Cause the network to update its weights to attempt to give this target value.
        self.Q.fit(state, target_vec, epochs=1, verbose=0)
        return

    def predict(self, state):
        """
        state:           np.array of shape self.state_shape; the floating point values describing the current state

        returns
        prediction:      np.array of shape (1, self.n_actions); the floating point predicted value of each action in the current state.

        For CartPole:
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        prediction is a np.array shape=[1,2] ((left_score,right_score)) of floats
        """
        # converts the single state into an array with 1 state
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        return prediction
    
    def get_best_action(self, state):
        """
        state:           np.array of shape self.state_shape; the floating point values describing the current state

        returns
        action:          the action that is predicted to give the best score in the current state

        For CartPole:
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        action is an integer (0,1) for (left,right)
        """
        # converts the single state into an array with 1 state
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        action = np.argmax(prediction[0])
        return action

    def get_Q_value(self, state, action):
        """
        state:           np.array of shape self.state_shape; the floating point values describing the current state
        action:          an integer; the action to give a value for

        returns
        q-value:         a float; the predicted value for taking action in state

        For CartPole:
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        action is an integer (0,1) for (left,right)
        q-value is a float
        """
        # converts the single state into an array with 1 state
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        return prediction[0][action]

    def get_best_action_value(self, state):
        """
        state:           np.array of shape self.state_shape; the floating point values describing the current state

        returns
        action:          the action that is predicted to give the best score in the current state
        q-value:         a float; the predicted value for taking action in state

        For CartPole:
        state is a np.array shape=[4] (x, x_dot, theta, theta_dot) of floats
        action is an integer (0,1) for (left,right)
        q-value is a float
        """
        # converts the single state into an array with 1 state
        state = state.reshape(-1, state.shape[0])
        prediction = self.Q.predict(state, verbose=0)
        best_action = np.argmax(prediction[0])
        best_Q_value = prediction[0][action]
        return best_action, best_Q_value

    def show(self):
        # for state in self.states():
        #     state = np.array([state])
        #     prediction = self.Q.predict(state)
        #     print(prediction)
        state = np.array([[0.0,0.0,0.0,0.0]])
        prediction = self.Q.predict(state, verbose=0)
        print(prediction)
        return

    def save(self, model_file):
        self.Q.save(model_file)
        return

    def load(self, model_file):
        self.Q = keras.models.load_model(model_file)
        return

def get_model_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-model.h5".format(environment_name)
    return model_file

def get_rewards_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-rewards.csv".format(environment_name)
    return model_file

# The openai gym environment is loaded
def load_environment(my_args):
    if my_args.track_steps:
        render_mode = "human"
    else:
        render_mode = None
    if my_args.environment == 'mario':
        env = GymEnv()
        env = TimeLimit(env, max_episode_steps=100)
    elif my_args.environment == 'cart':
        env = gym.make('CartPole-v1', render_mode=render_mode)
    else:
        raise Exception("Unexpected environment: {}".format(my_args.environment))
    # env.observation.n, env.action_space.n gives number of states and action in env loaded
    return env

def learn_epoch(Q, env, chance_epsilon, gamma, my_args):
    action_list = Q.actions()

    # Reset environment, getting initial state
    state, info = env.reset()
    prediction = Q.predict(state)
    
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table temporal difference learning algorithm
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q function
        # To facilitate learning, have chance of random action
        # instead of always choosing the best action
        chance = np.random.sample(1)[0]
        if chance < chance_epsilon:
            action = np.random.choice(action_list)
        else:
            action = Q.get_best_action(state)

        # Take action, get the new state and reward
        print(action)

        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        # if env.env.model.mario.life == 0:
        #     reward -= 1000

        #valocity(speed -0.07~+0.07)
        # reward += abs(next_state[1])

        next_prediction = Q.predict(next_state)

        # Update Q-Table with new data
        Q.update(state, action, next_state, reward, gamma, prediction, next_prediction, epoch_done)
        epoch_total_reward += reward
        state = next_state
        prediction = next_prediction

    return state, epoch_total_reward

def evaluate_epoch(Q, env, my_args):
    # Reset environment, getting initial state
    state, info = env.reset()
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table policy evaluation
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q table
        action = Q.get_best_action(state)

        # Take action, get the new state and reward
        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)

        # Update reward and state
        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def Q_learn(Q, env, my_args):
    almost_one = my_args.epsilon_chance_factor
    gamma = my_args.gamma
    epoch_rewards = [] # rewards per epochs
    chance_epsilon = almost_one

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = learn_epoch(Q, env, chance_epsilon, gamma, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
            sys.stdout.flush()

        # make less likely to experiment
        # assumes positive scores for successful completion
        if epoch_total_reward > -195:
            chance_epsilon *= almost_one
            chance_epsilon = max(chance_epsilon, 0.01)
        
    return epoch_rewards

def Q_evaluate(Q, env, my_args):
    epoch_rewards = [] # rewards per epochs

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = evaluate_epoch(Q, env, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
        
    return epoch_rewards

def do_learn(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Build new Q-function structure
    # assumes that the environment has Box observation space and discrete action space
    Q = QFunction(env.observation_space.shape, env.action_space.n)

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    if os.path.exists(model_file):
        print("Model loading from {}.".format(model_file))
        Q.load(model_file)
    
    # Learn
    epoch_rewards = Q_learn(Q, env, my_args)

    print("Learn: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    Q.save(model_file)
    print("Model saved to {}.".format(model_file))

    rewards_file = get_rewards_filename(my_args.rewards_file, my_args.environment)
    df = pd.DataFrame(columns = ["epoch","reward"])
    for i in range(0, len(epoch_rewards)):
        df.loc[i] = [i, epoch_rewards[i]]
    df.to_csv(rewards_file, index=False)
    
    return

def do_score(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Load existing Q-Table
    # assumes that the environment has discrete observation and action spaces
    Q = QFunction([0], 0)
    model_file = get_model_filename(my_args.model_file, my_args.environment)
    print("Model loading from {}.".format(model_file))
    Q.load(model_file)


    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return

def do_left(my_args):
    # Load Environment
    env = load_environment(my_args)

    Q = QLeft()

    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return

def do_right(my_args):
    # Load Environment
    env = load_environment(my_args)

    Q = QRight()

    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return

def do_random(my_args):
    # Load Environment
    env = load_environment(my_args)

    Q = QRandom()

    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Q-Table Learning')
    parser.add_argument('action', default='learn',
                        choices=[ "learn", "score", "left", "right", "random" ], 
                        nargs='?', help="desired action")
    
    parser.add_argument('--environment',   '-e', default="mario", type=str,  choices=('cart','m_car','mario' ), help="name of the OpenAI gym environment")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from environment)")
    parser.add_argument('--rewards-file',  '-r', default="",    type=str,   help="name of file for the rewards (default is constructed from environment)")

    #
    # hyper parameters
    #
    parser.add_argument('--gamma', '-g', default=0.98,  type=float, help="Q-learning hyper parameter (default=0.5)")
    parser.add_argument('--epsilon-chance-factor', '-c', default=0.06,  type=float, help="Scaling factor for learning policy chance of choosing random action (default=0.1)")

    parser.add_argument('--n-epochs', '-n',   default=10, type=int,   help="number of episodes to run (default=10).")

    # debugging/observations
    parser.add_argument('--track-epochs',    '-t', default=1,         type=int,   help="0 = don't display per-epoch information, 1 = do display per-epoch information (default=0)")
    parser.add_argument('--track-steps',     '-s', default=0,         type=int,   help="0 = don't display per-step information, 1 = do display per-step information (default=0)")


    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'learn':
        do_learn(my_args)
    elif my_args.action == 'score':
        do_score(my_args)
    elif my_args.action == 'left':
        do_left(my_args)
    elif my_args.action == 'right':
        do_right(my_args)
    elif my_args.action == 'random':
        do_random(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import FrameStack
from gymnasium import ObservationWrapper
import matplotlib.pyplot as plt
import pandas
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras.layers import Flatten, MaxPool2D, Reshape, Conv2D, Dense
from networks.convolutional_pong import ConvolutionalPong24, ConvolutionalPong32
from utils.csv_writer import CSVWriter

arg_parser = argparse.ArgumentParser(description="give a filename of a network to evaluate")

arg_parser.add_argument("path",
                        help="The path to the network, usually in" 
                            + "results/<date_network>/<network>.h5")

args = arg_parser.parse_args()

def karpathy_prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()

def run_stochastic_policy(policy_network, observation):
    # Reshape observation to (1,num_features)
    observation = observation[np.newaxis,:]
    # Run forward propagation to get softmax probabilities
    action_probabilities = policy_network(observation).numpy().reshape(-1)
    #print("ACTION PROBABILITIES: ", action_probabilities)
    # Select action using a biased sample
    # this will return the index of the action we've sampled
    action = np.random.choice(range(3), p=action_probabilities)
    #print("action: ", action)
    assert(action<3)
    return action

environment_name='ALE/Pong-v5'

policy_network=keras.models.load_model(filepath=args.path)

if True:
    env = gym.make(environment_name, render_mode="human") # For this demo, we do want to render the lander
    env.action_space = [1, 2, 3]

    for episode in range(10):
        observation, info = env.reset(seed=episode) # Policy gradient has high variance, seed for reproducability
        #use karpathy's pre-processing

        observation = karpathy_prepro(observation)

        episode_reward = 0
        done = False
        steps=0
        episode_observations=[]
        episode_actions=[]
        episode_rewards=[]

        prev_observation = None
        while not(done):
            # 1. Get difference between current and previous frame to retain
            #       movement of balls and paddles
            between_obs = None
            if prev_observation is not None:
                    between_obs = observation - prev_observation
            
            # 2. choose action based on observation
            if between_obs is not None:
                action = run_stochastic_policy(policy_network, between_obs)
                # show inbetween obs 
                #plt.imshow(np.reshape(between_obs, ((80, 80))))
                #input("Enter to continue")
            else:
                action = run_stochastic_policy(policy_network, observation)

            # 3. Take action in the environment
            observation_, reward, terminated, truncated, info= env.step(env.action_space[action])
            observation_ = karpathy_prepro(observation_)
            done=(terminated or terminated)
            steps+=1
            reward+=0.5/700 # reward longer episodes(agent hitting ball)

            # 4. Store transition for training
            if between_obs is not None:
                episode_observations.append(between_obs)
            else:
                episode_observations.append(observation)
            episode_rewards.append(reward)
            episode_actions.append(action)
            # 5.Save new observation, store previous observation
            prev_observation = observation
            observation = observation_
# This code initially came from https://github.com/gabrielgarza/openai-gym-policy-gradient
# Then modified to work with TensorlowV2.x by M. Fairbank, with many further enhancements.
# Majority of comments by M. Fairbank.
# cleaned up non-graphical version intended to work on the HPC
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import ResizeObservation
from gymnasium.wrappers import FrameStack
from gymnasium import ObservationWrapper
import matplotlib.pyplot as plt
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
from networks.convolutional_pong import ConvolutionalPong24, ConvolutionalPong32, ConcatConvolutionalPong32
from networks.KPHidden import KPHidden, KPHidden2
from utils.csv_writer import CSVWriter

# flag for cutting off runs early
mutable_numeps = False

karp_prepro = True

arg_parser = argparse.ArgumentParser(
    description="Select num_episodes, learning rate and network architecture")

arg_parser.add_argument("eps",
                        nargs='?',
                        default="1000",
                        help="The number of episodes to run for. Select 1 if you"
                            + "want to run for as long as there is improvement"
                            + "(limited to 20000 episodes)")
arg_parser.add_argument("lr",
                        nargs='?',
                        default="0.0005",
                        help="The learning rate.")
arg_parser.add_argument("df",
                        nargs='?',
                        default="0.9",
                        help="The discount factor.")
arg_parser.add_argument("network",
                        nargs='?',
                        choices=["hidden200", "hidden400", "conv_24", "conv_32",
                                  "concat_conv", "hidden100x2", "hidden200x2"],
                        default="hidden200",
                        help="The structure of the network")
arg_parser.add_argument("opt",
                        nargs='?',
                        choices=["sgd", "rmsprop", "adam"],
                        default="rmsprop",
                        help="The structure of the network")

args = arg_parser.parse_args()

num_episodes = int(args.eps)
if num_episodes == 1:
    # this will be an adaptive number of episodes, will stop only when network 
    # stops improving with an upper bound of 20,000 episodes
    print("Using adaptive number of episodes")
    num_episodes = 20000
    mutable_numeps = True

learning_rate = float(args.lr)
discount_factor = float(args.df)
NET_ARCHITECTURE = args.network
OPTIMISER = args.opt
print(f"Argument 1: {num_episodes}")
print(f"Argument 2: {learning_rate}")
print(f"Argument 3: {NET_ARCHITECTURE}")
print(f"Argument 3: {OPTIMISER}")


# creating observation cropping wrapper
# use karpathy's preprocessing instead of this
class CropObservation(ObservationWrapper):
    def __init__(self, env, crop_top, crop_bottom):
        super(CropObservation, self).__init__(env)
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        temp_shape  = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255, 
            shape=(temp_shape[0]-crop_top-crop_bottom, temp_shape[1]))

    def observation(self, obs):
        return obs[self.crop_top:-self.crop_bottom]
    
# This function is meant to calculate (dL/d Theta), 
# where L=(\sum_t (log(P_t))(R-b).
def calculate_reinforce_gradient(
        episode_observations, rewards_minus_baseline, 
        episode_actions, policy_network):
    
    
    # Train on episode
    batch_trajectory=np.stack(episode_observations)
    batch_action_choices=np.stack(episode_actions).astype(np.int32)
    
    # Check all input arrays are the correct shape...
    assert batch_trajectory.shape[0]==rewards_minus_baseline.shape[0]
    assert batch_trajectory.shape[0]==batch_action_choices.shape[0]
    assert len(batch_trajectory.shape)==2
    assert len(rewards_minus_baseline.shape)==1
    assert len(batch_action_choices.shape)==1
    
    # This is the REINFORCE gradient calculation
    with tf.GradientTape() as tape:
        # Note, don't need a tape.watch here because tensorflow by default 
        # always "watches" all Variable tensors, 
        # i.e. all of our neural network weights.
        trajectory_action_probabilities=policy_network(batch_trajectory)

        # Note that the next 2 lines could be replaced by a single call 
        # to tf.keras.losses.SparseCategoricalCrossentropy
        # this returns a tensor of shape [trajectory_length]
        chosen_probabilities=tf.gather(
            trajectory_action_probabilities, indices=batch_action_choices,
            axis=1, batch_dims=1) 
        log_probabilities=tf.math.log(chosen_probabilities)

        # Instead of using R-baseline, we are using R_t-baseline here, 
        # i.e. where R_t is the reward to go from step t
        logprobrewards=log_probabilities*rewards_minus_baseline 
        L=tf.reduce_mean(logprobrewards)

    # checking the original large array has gone through a reduce_sum
    assert len(L.shape)==0 

    # This calculates the gradient required by REINFORCE
    # This function doesn't actually do the update.  
    # It just calculates the gradient ascent direction, and returns it!
    grads = tape.gradient(L, policy_network.trainable_weights)
    return grads


def calculate_accumulated_discounted_rewards(episode_rewards, discount_factor):
    #print("episode_rewards: ", episode_rewards)
    discounted_episode_rewards= np.ones_like(episode_rewards)
    cumulative = 0
    for t in reversed(range(len(episode_rewards))):
        cumulative = cumulative * discount_factor + episode_rewards[t]
        discounted_episode_rewards[t] = cumulative
    #print(discounted_episode_rewards)
    # We need to return a numpy array of the same length and shape 
    # as the input array episode_rewards.
    return discounted_episode_rewards 
    

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

# from Kaprthy's blog: http://karpathy.github.io/2016/05/31/rl/
def karpathy_prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

environment_name='ALE/Pong-v5'
env_silent = gym.make(environment_name, render_mode=None)

## pre processing
if not karp_prepro:    
    env_silent = GrayScaleObservation(env_silent)
    env_silent = ResizeObservation(env_silent, shape=(105, 80))
    env_silent = CropObservation(env_silent, crop_top=17, crop_bottom = 9)
    env_silent = FlattenObservation(env_silent)

#2 = up
#3 = down
env_silent.action_space = [1, 2, 3]
env=env_silent

rewards = []

if OPTIMISER == "rmsprop":
    optimizer=keras.optimizers.RMSprop(learning_rate, rho=0.99)    
elif OPTIMISER == "sgd":
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate)
elif OPTIMISER == "adam":
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)

if karp_prepro:
    num_inputs = 6400
else:
    num_inputs=env.observation_space.shape[0]

# gymnasium has 6 actions for pong, but only two move the paddle  
# change it back to :
# num_outputs=env.action_space.n
# to restore compatability with other games
num_outputs=3

# Build a keras neural-network for the stochastic policy network:
if NET_ARCHITECTURE=='conv_32':
    nn = ConvolutionalPong32(num_inputs, num_outputs, OPTIMISER)
elif NET_ARCHITECTURE=='conv_24':
    nn = ConvolutionalPong24(num_inputs, num_outputs, OPTIMISER)
elif NET_ARCHITECTURE=='concat_conv':
    nn = ConcatConvolutionalPong32(num_inputs, num_outputs, OPTIMISER)
elif NET_ARCHITECTURE=='hidden200':
    nn = KPHidden(num_inputs, num_outputs, 200, OPTIMISER)
elif NET_ARCHITECTURE=='hidden400':
    nn = KPHidden(num_inputs, num_outputs, 400, OPTIMISER)
elif NET_ARCHITECTURE=='hidden100x2':
    nn = KPHidden2(num_inputs, num_outputs, 100, OPTIMISER)
elif NET_ARCHITECTURE=='hidden200x2':
    nn = KPHidden2(num_inputs, num_outputs, 200, OPTIMISER)

policy_network = nn.get_model()

reward_history=[]
mean_discounted_reward_history=[]
best_avg_fitness = -21

# store results in new directory
new_dir_name = ("results/" + datetime.now().strftime("%Y%m%d_%H%M") + "_" 
                + NET_ARCHITECTURE + "_DF_" + str(discount_factor) + "_LR_" 
                + str(learning_rate) + "_OPT_" + OPTIMISER)
os.makedirs(new_dir_name)

for episode in range(num_episodes):
    env=env_silent

    # Policy gradient has high variance, seed for reproducability
    observation, info = env.reset(seed=episode)

    #use karpathy's pre-processing
    if karp_prepro:
            observation = karpathy_prepro(observation)

    episode_reward = 0
    done = False
    steps=0
    episode_observations=[]
    episode_actions=[]
    episode_rewards=[]

    prev_observation = None
    # limit to 1000 steps
    while not(done) and steps < 1000:
        # 1. Get difference between current and previous frame to retain
        #       movement of balls and paddles
        between_obs = None
        if prev_observation is not None:
                between_obs = observation - prev_observation
        
        # 2. choose action based on observation
        if between_obs is not None:
            action = run_stochastic_policy(policy_network, between_obs)
        else:
            action = run_stochastic_policy(policy_network, observation)

        # 3. Take action in the environment
        observation_, reward, terminated, truncated, info = env.step(
            env.action_space[action])

        if karp_prepro:
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
        
    episode_rewards_sum = sum(episode_rewards)
    rewards.append(episode_rewards_sum)
    max_reward_so_far = np.amax(rewards)

    # save currently best policy network
    if episode_rewards_sum == max_reward_so_far:
        # delete old best network
        for filename in os.listdir(new_dir_name):
            if "BEST" in filename:
                file_path = os.path.join(new_dir_name, filename)
                os.remove(file_path)

        keras.models.save_model(policy_network, new_dir_name 
                                + f"/Pong_BEST_{episode_rewards_sum:.2f}" 
                                + "_DF" + str(discount_factor) 
                                + "_LR" + str(learning_rate) 
                                + "_EPS" + str(episode)
                                + "_KP_Conv.h5",save_format='h5')

    print("==========================================")
    print("Episode: ", episode)
    print("Steps:",steps)
    print("Reward: ", episode_rewards_sum)
    print("Max reward so far: ", max_reward_so_far)
    reward_history.append(episode_rewards_sum)

    # store the network at regular intervals
    if episode % 500 == 0 and episode != 0 :
        # Save the current model into a local folder
        curr_average_fitness=np.array(reward_history[-50:]).mean()
        keras.models.save_model(policy_network, new_dir_name 
                                + f"/Pong_FIT{curr_average_fitness:.2f}_DF" 
                                + str(discount_factor) 
                                + "_LR" + str(learning_rate) 
                                + "_EPS" + str(episode)
                                + "_KP_Conv.h5",save_format='h5')
        
    if episode % 250 == 0 and episode != 0 :
        print("checking whether to continue...")
        # write results to a csv file
        csv_writer = CSVWriter(new_dir_name + "/results.csv", reward_history)
        csv_writer.write()

        #get average fitness across the previous 100 episodes
        curr_average_fitness=np.array(reward_history[-100:]).mean()

        # check if there has been notable improvement
        if curr_average_fitness > best_avg_fitness + 0.1:
            print(
                f"{curr_average_fitness} greater than {best_avg_fitness + 0.1}")
            print("Continuing")
            best_avg_fitness = curr_average_fitness
        # no notable improvement, break out of loop    
        elif mutable_numeps:
            # for book keeping purposes
            num_episodes = episode
            print(f"{curr_average_fitness} less than {best_avg_fitness + 0.1}")
            print("NO IMPROVEMENT, ENDING.")
            break

    # 5. Train neural network
    # Discount and normalize episode reward
    if len(mean_discounted_reward_history)>2:
        hist_len=min(len(mean_discounted_reward_history),1)
        arr=np.array(mean_discounted_reward_history[-hist_len:])

        # This our estimate of a good BASELINE to be used in the REINFORCE 
        # algorithm.
        # The baseline we've used here is a moving average, but really 
        # should be a fixed quantity for REINFORCE algorithm derivation.
        baseline=np.mean(arr)

        # This attempts to rescale the rewards so that they are closer to being 
        # in the range [-1,1], which should make the learning rate 
        # more appropriate.
        reward_scaler=1/(np.std(arr)+1) 
        print("reward scalar:", reward_scaler)
        
    else:
        reward_scaler=0
        baseline=0
    discounted_episode_rewards = calculate_accumulated_discounted_rewards(
        episode_rewards, discount_factor)
    
    mean_discounted_reward_history.append(np.mean(discounted_episode_rewards))
    discounted_episode_rewards -= baseline
    discounted_episode_rewards *= reward_scaler
    #print("Discounted rewards: ", discounted_episode_rewards)
    grads=calculate_reinforce_gradient(
        episode_observations, discounted_episode_rewards, 
        episode_actions, policy_network)
    
    # Put a minus sign before all of the gradients - because in RL we are trying
    # to MAXIMISE a rewards, but optimizer.apply_graidents only works 
    # with MINIMISATION.
    grads=[-g for g in grads] 
    # This updates the parameter vector
    optimizer.apply_gradients(zip(grads, policy_network.trainable_weights)) 
    #print(grads)
    print("Gradient Mags: ", [np.abs(g).sum() for g in grads])


if karp_prepro:
    final_average_fitness=np.array(reward_history[-50:]).mean()
    print("Finished Training.  Final average fitness",final_average_fitness)
    keras.models.save_model(policy_network, new_dir_name 
                                + f"/Pong_FIT{final_average_fitness:.2f}_DF" 
                                + str(discount_factor) 
                                + "_LR" + str(learning_rate) 
                                + "_EPS" + str(num_episodes)
                                + "_KP_Conv.h5",save_format='h5')
else:
    final_average_fitness=np.array(reward_history[-50:]).mean()
    print("Finished Training.  Final average fitness",final_average_fitness)
    # Save the current model into a local folder
    keras.models.save_model(policy_network, "Model_"+environment_name+"_DF" 
                            + str(discount_factor) + "_LR"+str(learning_rate) 
                            + "_EPS"+str(num_episodes)+"_Conv.h5",
                            save_format='h5')

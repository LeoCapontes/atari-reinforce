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
from tensorflow import keras
from keras import layers
from keras import Sequential
from keras.layers import Flatten, MaxPool2D, Reshape, Conv2D, Dense

class KPHidden:
    def __init__(self, num_inputs, num_outputs, hidden_nodes, optimiser):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_nodes = hidden_nodes
        self.optimiser = optimiser
        self._build_model()

    def _build_model(self):
        hidden_nodes=[self.hidden_nodes]
        inputs = keras.Input(shape=(self.num_inputs,))
        x = inputs
        x_ = layers.Dense(hidden_nodes[0], activation="tanh", kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None))(x)
        x = tf.concat([x,x_],axis=1) # This passes shortcut connections from all earlier layers to this next one

        outputs = layers.Dense(self.num_outputs, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None), activation='softmax')(x)
        self.policy_network = keras.Model(inputs=inputs, outputs=outputs, name="policy_network")
        self.policy_network.compile(optimizer=self.optimiser, loss='sparse_categorical_crossentropy')

    def get_model(self):
        return self.policy_network
    
class KPHidden2:
    def __init__(self, num_inputs, num_outputs, hidden_nodes, optimiser):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_nodes = hidden_nodes
        self.optimiser = optimiser
        self._build_model()

    def _build_model(self):
        hidden_nodes=[self.hidden_nodes]
        inputs = keras.Input(shape=(self.num_inputs,))
        x = inputs

        x1 = layers.Dense(hidden_nodes[0], activation="tanh", kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None))(x)
        x2 = layers.Dense(hidden_nodes[0], activation="tanh", kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None))(x1)
        x2_concat = tf.concat([x, x2], axis=1) # This passes shortcut connections from all earlier layers to this next one

        outputs = layers.Dense(self.num_outputs, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.000005, seed=None), activation='softmax')(x2_concat)
        self.policy_network = keras.Model(inputs=inputs, outputs=outputs, name="policy_network")
        self.policy_network.compile(optimizer=self.optimiser, loss='sparse_categorical_crossentropy')

    def get_model(self):
        return self.policy_network
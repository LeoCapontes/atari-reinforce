
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
from keras.layers import Flatten, MaxPool2D, Reshape, Conv2D, Dense, Concatenate, Input
from keras.initializers import he_normal

# inspiration for kernel sizes etc. come from the following notebook:
# https://colab.research.google.com/drive/1KuzxUPUL3Y50xQFk8RvsfWDx88vDitZS
# the kernels etc. are not the same, they were used as a starting point, the
# kernels and features in this file were selected after trial and error.
class ConvolutionalAtari24:
    def __init__(self, num_inputs, num_outputs, optimizer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.optimizer = optimizer
        self._build_model()

    def _build_model(self):
        inputs = keras.Input(shape=(self.num_inputs,))
        self.policy_network = Sequential()
        self.policy_network.add(Reshape((80, 80, 2), input_shape = (self.num_inputs,)))
        self.policy_network.add(Conv2D(24, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Flatten())
        self.policy_network.add(Dense(self.num_outputs, activation='softmax'))
        self.policy_network.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy')

    def get_model(self):
        return self.policy_network

class ConvolutionalAtari32:
    def __init__(self, num_inputs, num_outputs, optimizer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.optimizer = optimizer
        self._build_model()

    def _build_model(self):
        inputs = keras.Input(shape=(self.num_inputs,))
        self.policy_network = Sequential()
        self.policy_network.add(Reshape((80, 80, 2), input_shape = (self.num_inputs,)))
        self.policy_network.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Conv2D(24, kernel_size=(3,3), padding='same', activation='relu'))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu'))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Conv2D(8, kernel_size=(3,3), padding='same', activation='relu'))
        self.policy_network.add(MaxPool2D(pool_size=(2, 2)))
        self.policy_network.add(Flatten())
        self.policy_network.add(Dense(self.num_outputs, activation='softmax'))
        self.policy_network.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy')

    def get_model(self):
        return self.policy_network
    
class ConcatConvolutionalAtari32:
    def __init__(self, num_inputs, num_outputs, optimizer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.optimizer = optimizer
        self._build_model()

    # need to use functional keras for concatenating
    def _build_model(self):
        input_layer = Input(shape=(self.num_inputs,))
        reshaped_input = Reshape((80, 80, 2))(input_layer)
        
        # vonvolutional layers
        conv1 = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(reshaped_input)
        pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(24, kernel_size=(3,3), padding='same', activation='relu')(pool1)
        pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(12, kernel_size=(3,3), padding='same', activation='relu')(pool2)
        pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(8, kernel_size=(3,3), padding='same', activation='relu')(pool3)
        pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
        flatten = Flatten()(pool4)
        # concatenate final flattened layer with input
        shortcut_concat = Concatenate()([flatten, input_layer]) 

        output_layer = Dense(self.num_outputs, activation='softmax')(shortcut_concat)       
        self.policy_network = keras.Model(inputs=input_layer, outputs=output_layer)
        self.policy_network.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy')

    def get_model(self):
        return self.policy_network
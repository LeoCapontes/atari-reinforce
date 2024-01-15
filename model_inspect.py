# mostly taken from:
# https://www.binarystudy.com/2022/10/visualizing-filters-and-feature-maps-in-cnn-tensorflow-keras.html
# loads a .h5 file and visualises the learnt features
# also prints summary

import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
from tensorflow import keras
from keras import Model
import sys
from matplotlib import pyplot as plt

with tf.device("/cpu:0"):
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py <path_to_h5_file>")
        sys.exit(1)

    model_path = sys.argv[1]

    model = tf.keras.models.load_model(model_path)

    model.summary()

    plt.ion()
    plt.figure()


    def plot_feature_maps(model, layer_name, input_image):

        layer = model.get_layer(layer_name)
        # og is grayscale
        input_image = Image.open(input_image).convert('L')  
        input_image = np.array(input_image) / 255.0  
        #original netowrk input is [-1, 1]
        input_image = (input_image *2) -1 
        # network input is flat
        input_image = input_image.reshape(-1) 
        input_image = np.expand_dims(input_image, axis=0) 

        # create a subset model (from input to selected layer)
        feature_map_model = Model(inputs=model.input, outputs=layer.output)

        # Predict to get the feature maps
        feature_maps = feature_map_model.predict(input_image)

        # [lot feature maps
        num_features = feature_maps.shape[-1]
        featureplot_num = int(np.sqrt(num_features)) + 1

        for i in range(num_features):
            plt.subplot(featureplot_num, featureplot_num, i+1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')

        plt.show()
        input("Press anything to continue")

    # Use the function
    input_image = 'prepro_ex.png'
    plot_feature_maps(model, 'conv2d_2', input_image)
# Neural Network Agent for Atari Pong with Policy Gradient
## Description

This project explores the development of a neural network agent trained to play the Atari game Pong using a policy gradient learning algorithm. It delves into various neural network architectures, emphasizing the effectiveness of dense and convolutional layers in reinforcement learning environments.
## Technologies Used

Programming Language: Python
Libraries: TensorFlow, Keras, Gymnasium. Seaborn for plots
Tools: GitLab for version control

## Methodology

Adapted reinforcement learning techniques for the Atari Pong environment.
Experimented with different neural network architectures including dense and convolutional layers. Also experimented with layer skipping.
Extensive hyper-parameter tuning to optimize performance.

## Results

Developed neural network configurations that improved over time and were capable of defeating the Atari AI.

After 500 episodes:

![CNN after 500 episodes of training](media/Pong_ep_500.gif)

After 7500 episodes:

![CNN after 7500 episodes of training](media/Pong_ep_7500.gif)

Detailed analysis and comparison of network features and their impact on performance.

### Network depth comparison using feature maps

One take away from this project was the impact of network depth on the 'readability' of extracted features.

CNN with 4 layers:

![4 layer feature map](media/conv32_map.png)

CNN with 3 layers:

![3 layer feature map](media/conv24_map.png)

Note the clarity and resolution of the extracted features by the final layer. At a certain point adding layers proved detrimental.
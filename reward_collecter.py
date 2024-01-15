# from chatgpt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


net_list = [
    'hidden200',
    'hidden400',
    'hidden100x2',
    'hidden200x2',
    '24',
    '32',
    'concat'
]

lr_list = [
    #'7.5e-05',
    '0.0001',
    '0.00025',
    '0.0005', 
    '0.00075',
    #'0.0009',
    '0.001',
    '0.00125',]

df_list = [
    '0.88',
    '0.9',
    '0.92',
    '0.99'
]

# Path to the parent directory containing all neural network directories
parent_dir = 'results/'

# Empty dictionary to store final rewards for each network
rewards_data = {}
episodes_data = {}

# Traverse through each directory
for directory in os.listdir(parent_dir):
    dir_path = os.path.join(parent_dir, directory)
    
    # Check if it's a directory and contains results.csv
    if (os.path.isdir(dir_path) and 'results.csv' in os.listdir(dir_path) 
        and directory.__contains__('finalPong') #and directory.__contains__('conv') 
        and directory.__contains__('rmsprop')):
        csv_path = os.path.join(dir_path, 'results.csv')
        params = directory.split('_')

        for hyper_param in net_list:
            if hyper_param in params:
                directory = hyper_param
                break
        else:
            print(directory)
        
        # Use pandas to read the CSV without headers and get the last reward
        df = pd.read_csv(csv_path, header=None)
        final_reward = df.iloc[-1, 2]
        final_episode = df.iloc[-1, 0]
        max_avg = df[2].iloc[51:].max()
        print(f"{directory} max is {max_avg}")

        # Add or append the reward to the rewards_data
        if directory not in rewards_data:
            rewards_data[directory] = [final_reward]
        else:
            rewards_data[directory].append(final_reward)

        if directory not in episodes_data:
            episodes_data[directory] = [final_episode]
        else:
            episodes_data[directory].append(final_episode)
        


# Convert dictionary to DataFrame and save as CSV
final_reward_df = pd.DataFrame.from_dict(rewards_data, orient='index').transpose()
final_episode_df = pd.DataFrame.from_dict(episodes_data, orient='index').transpose()

final_reward_df.to_csv('final_rewards.csv', index=False)

# Melt the data for seaborn
melted_rewards_df = final_reward_df.melt(var_name='Network', value_name='Reward')
melted_rewards_df['Metric'] = 'Reward'

melted_episodes_df = final_episode_df.melt(var_name='Network', value_name='Episodes')
melted_episodes_df['Metric'] = 'Episode Count'

print(rewards_data)

# Create catplot with box kind
sns.catplot(x='Network', y='Episodes', kind='box', data=melted_episodes_df, height=6, aspect=1.5, order=net_list)
plt.title("Number of episodes trained by Network Architecture", fontsize=14)
plt.xlabel('Network', fontsize=13)
plt.ylabel('Episodes', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.tight_layout()

sns.catplot(x='Network', y='Reward', kind='box', data=melted_rewards_df, height=6, aspect=1.5, order=net_list)
plt.title("Average reward (last 50 episodes) at episode 2000 by Network", fontsize=14)
# Rotate x-axis labels for better visibility if you have many networks
plt.xlabel('Network', fontsize=13)
plt.ylabel('Reward', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

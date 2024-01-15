import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

sns.set_theme()
lr_list = [
    '7.5e-05',
    '0.0001',
    '0.00025',
    '0.0005', 
    '0.00075',
    '0.0009',
    '0.001',
    '0.00125',]
net_list = [
    'hidden100x2',
    'hidden200x2',
    'hidden200',
    'hidden400',
    '24',
    '32',
    'concat'
]
df_list = [
    '0.88',
    '0.9',
    '0.92',
    '0.99'
]
opt_list = [
    'sgd',
    'rmsprop',
    'adam'
]

lr_colour_list = sns.color_palette('Set1', n_colors=len(lr_list))
lr_colour_mapping = dict(zip(lr_list, lr_colour_list))

net_colour_list = sns.color_palette('Set1', n_colors=len(net_list))
net_colour_mapping = dict(zip(net_list, net_colour_list))

df_colour_list = sns.color_palette('Set1', n_colors=len(df_list))
df_colour_mapping = dict(zip(df_list, df_colour_list))

opt_colour_list = sns.color_palette('Set1', n_colors=len(opt_list))
opt_colour_mapping = dict(zip(opt_list, opt_colour_list))

# use command line to inspect results of a single run with its directory
arg_parser = argparse.ArgumentParser(description="Give the directory where the results.csv is")

arg_parser.add_argument("dir",
                        nargs="?",
                        default="",
                        help="The dir containing results, usually " 
                            + "results/<date_network>/")
args = arg_parser.parse_args()
dir = "results"

if args.dir != "":
    data = pd.read_csv(os.path.join(args.dir, "results.csv"), header=None)

    plt.ylim(-21, 21)

    plt.plot(data[0], data[1])
    plt.plot(data[0], data[2])
    plt.show() 
# plot all results if sepcific directory not given
else:
    # categorise results into lists depending on network configuration
    hidden200_dirs = []
    hidden400_dirs = []
    c24_dirs = []
    c32_dirs = []
    test_dirs = []
    # use this to group something specific
    string_to_test = "finalPong"

    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        # play with these conditionals to find specific configs
        if re.search ('pong', subdir, re.I):
            if re.search('conv_24', subdir, re.I):
                c24_dirs.append(subdir_path)
            if re.search('conv_32', subdir, re.I) and re.search('0.00075', subdir, re.I) and re.search('0.9', subdir, re.I):
                c32_dirs.append(subdir_path)
            if re.search('hidden200', subdir, re.I) and re.search('0.0005', subdir, re.I) and re.search('0.92', subdir, re.I) and not re.search("hidden200_2", subdir, re.I):
                hidden200_dirs.append(subdir_path)
            if re.search('hidden400', subdir, re.I) and re.search('0.9', subdir, re.I) and re.search('rmsprop', subdir, re.I):
                hidden400_dirs.append(subdir_path)

        if re.search(string_to_test, subdir, re.I):
            test_dirs.append(subdir_path)
    
    print("done")

    plt.figure(1, figsize=(12, 8))
    plt.title("Running average reward over past 50 episodes vs. Iteration. "
              + "Grouped by Optimiser. Hidden 200", fontsize=16)
    # only use each label once to group by hyper-parameter
    labels_plotted = []
    for run in hidden200_dirs :
        data = pd.read_csv(os.path.join(run, "results.csv"), header=None, skiprows=range(0, 51))
        # remove directory from label
        label = run.replace("results\\", "")
        params = label.split('_')

        # set label to be just the parameter to isolate
        for hyper_param in opt_list:
            if hyper_param in params:
                label = hyper_param
                break
        else:
            print(label)

        data.columns = ['Episode', 'Reward', 'Running Average Reward']
        colour = opt_colour_mapping.get(label)
        print(label, colour)
        if label not in labels_plotted :
            #sns.lineplot(data = data, x='X' ,y='Y', label='Y-values')
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label=label, color = colour, linestyle='-')
            labels_plotted.append(label)
        else: 
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label="_no_legend_", color = colour, linestyle='-')
    ax = plt.gca()
    legend = ax.legend()
    ax.set_xlabel('Episode', fontsize = 14)
    ax.set_ylabel('Reward', fontsize = 14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend.prop.set_size(14) 
    for line in legend.legendHandles:
        line.set_linewidth(3.0)
    for text in legend.get_texts():
        text.set_fontsize(14)
    
    print("done 200_lr")


    plt.figure(2, figsize=(14, 10))
    plt.title("Running average reward over past 50 episodes vs. Iteration. "
              + "Grouped by Discount Factor. Using Conv 32", fontsize=16)
    # only use each label once to group by hyper-parameter
    labels_plotted = []
    for run in c32_dirs :
        lw = 1.5
        data = pd.read_csv(os.path.join(run, "results.csv"), header=None, skiprows=range(0, 51))
        # remove directory from label
        label = run.replace("results\\", "")
        params = label.split('_')

        # set label to be just the parameter to isolate
        for hyper_param in opt_list:
            if hyper_param in params:
                label = hyper_param
                break
        else:
            print(label)

        data.columns = ['Episode', 'Reward', 'Running Average Reward']
        colour = opt_colour_mapping.get(label)
        print(label, colour)
        if label not in labels_plotted :
            #sns.lineplot(data = data, x='X' ,y='Y', label='Y-values')
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label=label, color = colour, linestyle='-', linewidth=lw)
            labels_plotted.append(label)
        else: 
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label="_no_legend_", color = colour, linestyle='-', linewidth=lw)
    ax = plt.gca()
    legend = ax.legend()
    ax.set_xlabel('Episode', fontsize = 14)
    ax.set_ylabel('Reward', fontsize = 14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend.prop.set_size(14) 
    for line in legend.legendHandles:
        line.set_linewidth(3.0)
    for text in legend.get_texts():
        text.set_fontsize(14)
    
    print("conv32 done")

    plt.figure(3, figsize=(10, 8))
    plt.title("Running average reward over past 50 episodes vs. Iteration. "
              + "Grouped by Learning Rate. Conv 24")
    # only use each label once to group by hyper-parameter
    labels_plotted = []
    for run in c24_dirs :
        data = pd.read_csv(os.path.join(run, "results.csv"), header=None, skiprows=range(0, 51))
        # remove directory from label
        label = run.replace("results\\", "")
        params = label.split('_')

        # set label to be just the parameter to isolate
        for hyper_param in lr_list:
            if hyper_param in params:
                label = hyper_param
                break
        else:
            print(label)

        data.columns = ['Episode', 'Reward', 'Running Average Reward']
        colour = lr_colour_mapping.get(label)
        print(label, colour)
        if label not in labels_plotted :
            #sns.lineplot(data = data, x='X' ,y='Y', label='Y-values')
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label=label, color = colour, linestyle='-')
            labels_plotted.append(label)
        else: 
            sns.lineplot(data=data, x='Episode', y='Running Average Reward', label="_no_legend_", color = colour, linestyle='-')
    

    plt.figure(4, figsize=(12, 10))
    plt.title("Running average reward in Pong over past 50 episodes vs. Iteration. "
              + "Grouped by Network Architecture.", fontsize=16)
    labels_plotted = []
    for run in test_dirs:
        data = pd.read_csv(os.path.join(run, "results.csv"), header=None, skiprows=range(0, 51))
        # remove directory from label
        label = run.replace("results\\", "")
        params = label.split('_')
        for param in net_list:
            if param in params:
                label = param
                break
        else:
            print(label)
        
        data.columns = ['X', 'Reward', 'Running_Average']
        colour = net_colour_mapping.get(label)
        print(label, colour)
        if label not in labels_plotted :
            #sns.lineplot(data = data, x='X' ,y='Y', label='Y-values')
            sns.lineplot(data=data, x='X', y='Running_Average', label=label, color = colour, linestyle='-')
            #sns.lineplot(data=data, x='X', y='Reward', label="_no_legend_", color = colour, linestyle='-', alpha = 0.2)
            labels_plotted.append(label)
        else: 
            sns.lineplot(data=data, x='X', y='Running_Average', label="_no_legend_", color = colour, linestyle='-')
    ax = plt.gca()
    legend = ax.legend()
    ax.set_xlabel('Episode', fontsize = 14)
    ax.set_ylabel('Reward', fontsize = 14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend.prop.set_size(14) 
    for line in legend.legendHandles:
        line.set_linewidth(3.0)
    for text in legend.get_texts():
        text.set_fontsize(14)


plt.figure(5, figsize=(14, 10))
plt.title("Running average reward over past 50 episodes vs. Iteration. "
            + "Grouped by Learning Rate. Hidden 400", fontsize=16)
# only use each label once to group by hyper-parameter
labels_plotted = []
for run in hidden400_dirs :
    data = pd.read_csv(os.path.join(run, "results.csv"), header=None, skiprows=range(0, 51))
    # remove directory from label
    label = run.replace("results\\", "")
    params = label.split('_')

    # set label to be just the parameter to isolate
    for hyper_param in lr_list:
        if hyper_param in params:
            label = hyper_param
            break
    else:
        print(label)

    data.columns = ['Episode', 'Reward', 'Running Average Reward']
    colour = lr_colour_mapping.get(label)
    print(label, colour)
    if label not in labels_plotted :
        #sns.lineplot(data = data, x='X' ,y='Y', label='Y-values')
        sns.lineplot(data=data, x='Episode', y='Running Average Reward', label=label, color = colour, linestyle='-')
        labels_plotted.append(label)
    else: 
        sns.lineplot(data=data, x='Episode', y='Running Average Reward', label="_no_legend_", color = colour, linestyle='-')
plt.legend(fontsize='large')
ax = plt.gca()
legend = plt.gca().legend()
ax.set_xlabel('Episode', fontsize = 14)
ax.set_ylabel('Reward', fontsize = 14)
for line in legend.legendHandles:
    line.set_linewidth(3.0)
print("done 400_lr")    

plt.figure(5, figsize=(14, 10))
plt.title("Running average reward over past 50 episodes vs. Iteration. "
            + "Grouped by Network Architecture", fontsize=16)


plt.show()
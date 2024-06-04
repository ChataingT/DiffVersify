import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os

# Import necessary libraries
# Load the data from CSV
root = r""
file = os.path.join(root, r'xpR_F1_data.csv')
data = pd.read_csv(file)

trad_meth = {
    'bc':'DiffVersify',
    'binaps':'BiNaps',
    'diffnaps':'DiffNaps'
}
# Group the data by the 'db' column
grouped_db = data.groupby('db')

# Create a list to store all unique methods
methods = []

# Create a figure and axes with as many subplots as there are unique 'db' values
fig, axs = plt.subplots(1, len(grouped_db), figsize=(15, 4.5))

marker = {'bc':'+-r', 'binaps':'o-b', 'diffnaps':'x-g'}
# Iterate over each unique 'db' and plot all methods on the same plot
for ax, (db, group_db) in zip(axs, grouped_db):
    # Group the data by 'method' within the current 'db' group
    grouped_method = group_db.groupby('method')
    
    # Plot each method within the current 'db' group
    for method, group_method in grouped_method:
        ax.plot(group_method['x'], group_method['y'],  marker[method], label=trad_meth[method])

        if method not in methods:
            methods.append(method)
    
    # Add labels and title
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Coverage')
    ax.set_title(f'{db}')
    ax.grid(True)
    ax.set_ylim(0,1)

# lines = [Line2D([0], [0], color=plt.cm.viridis(i/len(methods)), lw=2) for i in range(len(methods))]
# fig.legend(lines, methods)
line1 = [Line2D([0], [0], marker=marker[m][0], color=marker[m][2], lw=2) for m in marker.keys()]

# plt.legend(handles=[scatter, line1, line2], loc='upper right', title='My Legend', fontsize='small')

fig.legend(line1, [trad_meth[m] for m in methods], loc='upper center', ncol=6, fancybox=True)

# Show plot
# plt.tight_layout()


plt.savefig(file[:-9]+'.jpeg', dpi=1000)
plt.savefig(file[:-9]+'.svg')
plt.savefig(file[:-9]+'.eps', dpi=1000)


plt.show()


"""
# Load the data from CSV
data = pd.read_csv('xpR_F1_data.csv')

# Group the data by the 'db' column
grouped_db = data.groupby('db')

# Iterate over each unique 'db' and plot all methods on the same plot
for db, group_db in grouped_db:
    # Initialize the plot
    plt.figure(figsize=(8, 6))
    
    # Group the data by 'method' within the current 'db' group
    grouped_method = group_db.groupby('method')
    
    # Plot each method within the current 'db' group
    for method, group_method in grouped_method:
        plt.plot(group_method['x'], group_method['y'], label=method)
    
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('softF1')
    plt.title(f'{db}')
    plt.legend(loc="best")
    plt.grid(True)
    
    # Show plot
    plt.show()
"""
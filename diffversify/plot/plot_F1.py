import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the style of seaborn
sns.set_style("whitegrid")

root = r""
df = pd.DataFrame()
# Load the data from CSV
for file, metric in zip(["xp1_F1_data.csv", "xp2_F1_data.csv", "xp3_F1_data.csv", "xp4_F1_data.csv"], ["nlines", "kclasses", "Anoise", "Dnoise"]):

    data = pd.read_csv(os.path.join(root, file))
    df = pd.concat([df, data])

# Plotting
plt.figure(figsize=(10, 6))

# Iterate through unique methods
for method in data['method'].unique():
    method_data = data[data['method'] == method]
    sns.lineplot(x=metric, y='mean', data=method_data, label=method, ci='sd')
    plt.fill_between(method_data[metric], method_data['mean'] - method_data['std'], method_data['mean'] + method_data['std'], alpha=0.2)

plt.xlabel(metric)
plt.ylabel('F1')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(root, file[:-9]+'.jpeg'))
plt.savefig(os.path.join(root, file[:-9]+'.svg'))

plt.show()

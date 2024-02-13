import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# this code aims to plot the comparison of SEV- with other local explanations methods based on the number of features changed in generating the counterfactural explanations

# Read the data
data_names = ["adult","compas","mimic","german","diabetes","fico"]
columns = ["SEV Minus","TreeSHAP","KernelSHAP","LIME","DiCE"]

for data_name in data_names:
    df = pd.read_excel("SEV Comparison.xlsx", sheet_name=data_name)
    df = df[columns]
    df = df.replace(0,np.nan).dropna(axis=0,how="all").fillna(0)

    fig, axes = plt.subplots(nrows=5, figsize=(10,12))
    cmap = sns.color_palette()  # Use seaborn default color palette

    for i, ax in enumerate(axes):
        ax.bar(df.index+1, df[columns[i]], color=cmap[i],alpha=0.7,label=columns[i])
        # ax.set_title(f"Plot {i+1}")
        ax.legend(loc="upper right",fontsize=28)
        ax.set_xticks(np.arange(1, np.max(df.index)+2)) 
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)

    ax.set_xlabel("Flip Numbers",fontsize=24)
    plt.tight_layout()
    plt.savefig("Experiment_C2/flip_%s.png"%data_name,dpi=500)
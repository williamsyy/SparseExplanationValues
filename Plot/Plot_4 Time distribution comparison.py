import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
data_names = ["adult","compas","mimic","german","diabetes","fico"]

for data_name in data_names:
    df = pd.read_csv("../Experiments/Results/Exp7/%s_1.csv"%data_name)
    kernelSHAP = df["kernelSHAP"]
    SEV = df["SEV"]
    print(data_name)
    print("kernelSHAP,%.4f,%.4f,%.4f,%.2f"%(np.mean(kernelSHAP<0.1),np.mean(kernelSHAP<0.5),np.mean(kernelSHAP<1),np.max(kernelSHAP)))
    print("SEV,%.4f,%.4f,%.4f,%.2f"%(np.mean(SEV<0.1),np.mean(SEV<0.5),np.mean(SEV<1),np.max(SEV)))
    sns.histplot(kernelSHAP,binwidth=0.05,label="KernelSHAP",alpha=0.5,binrange=(0,1))
    sns.histplot(SEV,binwidth=0.05,label="SEV",alpha=0.5,binrange=(0,1))
    plt.xlabel("Time Consumptions",fontsize=18)
    plt.ylabel("Number of Samples",fontsize=18  )
    plt.legend(fontsize=20)
    plt.savefig("Experiment 7/%s_1.png"%data_name,dpi=500)
    plt.cla()
    plt.close()
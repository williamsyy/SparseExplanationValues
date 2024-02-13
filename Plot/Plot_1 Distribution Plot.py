import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read the data
data_names = ["Adult","COMPAS","Diabetes","FICO","German","MIMIC"]

for data_name in data_names:
    # load the data
    data = pd.read_excel("Experiment D1 Data/"+data_name+" Sample.xlsx").set_index(data_name)

    # first plot the distribution of SEV+ in LR
    lr_columns_sev_plus = ["L2 LR","L1 LR","All-Opt LR", "Vol-Opt LR"]
    lr_columns_sev_minus = ["L2 LR","L1 LR","All-Opt LR"]
    sev_plus_columns = {"SEV+ = 1": 1, "SEV+ = 2": 2, "SEV+ = 3": 3, "SEV+ = 4": 4, "SEV+ = 5": 5, "SEV+ = 6": 6, "SEV+ > 6": 7}
    sev_minus_columns = {"SEV- = 1": 1, "SEV- = 2": 2, "SEV- = 3": 3, "SEV- = 4": 4, "SEV- = 5": 5, "SEV- = 6": 6, "SEV- > 6": 7}

    df_1 = data.loc[sev_plus_columns.keys(),lr_columns_sev_plus]

    plt.figure(figsize=(6,6))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(4,1,1)
    plt.bar(sev_plus_columns.values(), df_1["L2 LR"],color=colors_red)
    plt.ylabel("Counts",fontsize=16)
    plt.title("L2 LR: Mean SEV+ = %.2f"%data.loc["Mean SEV+","L2 LR"],fontsize=18)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(4,1,2)
    plt.bar(sev_plus_columns.values(), df_1["L1 LR"],color=colors_red)
    plt.ylabel("Counts",fontsize=16)
    plt.title("L1 LR: Mean SEV+ = %.2f"%data.loc["Mean SEV+","L1 LR"],fontsize=18)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(4,1,3)
    plt.bar(sev_minus_columns.values(), df_1["All-Opt LR"],color=colors_blue)
    plt.ylabel("Counts",fontsize=16)
    plt.title("AllOpt+ LR: Mean SEV+ = %.2f"%data.loc["Mean SEV+","All-Opt LR"],fontsize=18)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(4,1,4)
    plt.bar(sev_minus_columns.values(), df_1["Vol-Opt LR"],color=colors_blue)
    plt.ylabel("Counts",fontsize=16)
    plt.title("Vol-Opt LR: Mean SEV+ = %.2f"%data.loc["Mean SEV+","Vol-Opt LR"],fontsize=18)
    plt.xlabel("SEV+",fontsize=16)
    plt.xticks([1,2,3,4,5,6,7],labels=["1","2","3","4","5","6",">6"],fontsize=16)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_LR_Plus.png"%data_name,dpi=500)
    plt.close()

    # second plot the distribution of SEV- in LR
    lr_columns_sev_minus = ["L2 LR","L1 LR","All-Opt LR"]
    sev_minus_columns = {"SEV- = 1": 1, "SEV- = 2": 2, "SEV- = 3": 3, "SEV- = 4": 4, "SEV- = 5": 5, "SEV- = 6": 6, "SEV- > 6": 7}

    df_1 = data.loc[sev_minus_columns.keys(),lr_columns_sev_minus]

    plt.figure(figsize=(6,4.5))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(3,1,1)
    plt.bar(sev_plus_columns.values(), df_1["L2 LR"],color=colors_red)
    plt.ylabel("Counts",fontsize=16)
    plt.title("L2 LR: Mean SEV- = %.2f"%data.loc["Mean SEV-","L2 LR"],fontsize=18)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(3,1,2)
    plt.bar(sev_plus_columns.values(), df_1["L1 LR"],color=colors_red)
    plt.ylabel("Counts",fontsize=16)
    plt.title("L1 LR: Mean SEV- = %.2f"%data.loc["Mean SEV-","L1 LR"],fontsize=18)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(3,1,3)
    plt.bar(sev_minus_columns.values(), df_1["All-Opt LR"],color=colors_blue)
    plt.ylabel("Counts",fontsize=16)
    plt.title("AllOpt- LR: Mean SEV- = %.2f"%data.loc["Mean SEV-","All-Opt LR"],fontsize=18)
    plt.xticks([1,2,3,4,5,6,7],labels=["1","2","3","4","5","6",">6"],fontsize=16)
    plt.xlabel("SEV-",fontsize=16)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_LR_Minus.png"%data_name,dpi=500)
    plt.close()

    # third plot the distribution of SEV+ in MLP
    mlp_columns_sev_plus = ["MLP","All-Opt MLP"]
    sev_plus_columns = {"SEV+ = 1": 1, "SEV+ = 2": 2, "SEV+ = 3": 3, "SEV+ = 4": 4, "SEV+ = 5": 5, "SEV+ = 6": 6, "SEV+ > 6": 7}

    df_1 = data.loc[sev_plus_columns.keys(),mlp_columns_sev_plus]

    plt.figure(figsize=(6,3))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(2,1,1)
    plt.bar(sev_plus_columns.values(),df_1["MLP"],color=colors_red)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.title("MLP: Mean SEV+ = %.2f"%data.loc["Mean SEV+","MLP"],fontsize=20)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.bar(sev_plus_columns.values(),df_1["All-Opt MLP"],color=colors_blue)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.xlabel("SEV+",fontsize=18)
    plt.xticks([1,2,3,4,5,6,7],fontsize=16)
    plt.title("AllOpt+ MLP: Mean SEV+ = %.2f"%data.loc["Mean SEV+","All-Opt MLP"],fontsize=20)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_MLP_Plus.png"%data_name,dpi=500)
    plt.close()

    # fourth plot the distribution of SEV- in MLP
    mlp_columns_sev_plus = ["MLP","All-Opt MLP"]
    sev_plus_columns = {"SEV- = 1": 1, "SEV- = 2": 2, "SEV- = 3": 3, "SEV- = 4": 4, "SEV- = 5": 5, "SEV- = 6": 6, "SEV- > 6": 7}

    df_1 = data.loc[sev_plus_columns.keys(),mlp_columns_sev_plus]

    plt.figure(figsize=(6,3))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(2,1,1)
    plt.bar(sev_plus_columns.values(),df_1["MLP"],color=colors_red)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.title("MLP: Mean SEV- = %.2f"%data.loc["Mean SEV-","MLP"],fontsize=20)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.bar(sev_plus_columns.values(),df_1["All-Opt MLP"],color=colors_blue)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.xlabel("SEV-",fontsize=18)
    plt.xticks([1,2,3,4,5,6,7],fontsize=16)
    plt.title("AllOpt- MLP: Mean SEV- = %.2f"%data.loc["Mean SEV-","All-Opt MLP"],fontsize=20)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_MLP_Minus.png"%data_name,dpi=500)
    plt.close()

    # third plot the distribution of SEV+ in GBDT
    mlp_columns_sev_plus = ["GBDT","All-Opt GBDT"]
    sev_plus_columns = {"SEV+ = 1": 1, "SEV+ = 2": 2, "SEV+ = 3": 3, "SEV+ = 4": 4, "SEV+ = 5": 5, "SEV+ = 6": 6, "SEV+ > 6": 7}

    df_1 = data.loc[sev_plus_columns.keys(),mlp_columns_sev_plus]

    plt.figure(figsize=(6,3))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(2,1,1)
    plt.bar(sev_plus_columns.values(),df_1["GBDT"],color=colors_red)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.title("GBDT: Mean SEV+ = %.2f"%data.loc["Mean SEV+","GBDT"],fontsize=20)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.bar(sev_plus_columns.values(),df_1["All-Opt GBDT"],color=colors_blue)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.xlabel("SEV+",fontsize=18)
    plt.xticks([1,2,3,4,5,6,7],fontsize=16)
    plt.title("AllOpt+ GBDT: Mean SEV+ = %.2f"%data.loc["Mean SEV+","All-Opt GBDT"],fontsize=20)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_GBDT_Plus.png"%data_name,dpi=500)
    plt.close()

    # fourth plot the distribution of SEV- in MLP
    mlp_columns_sev_plus = ["GBDT","All-Opt GBDT"]
    sev_plus_columns = {"SEV- = 1": 1, "SEV- = 2": 2, "SEV- = 3": 3, "SEV- = 4": 4, "SEV- = 5": 5, "SEV- = 6": 6, "SEV- > 6": 7}

    df_1 = data.loc[sev_plus_columns.keys(),mlp_columns_sev_plus]

    plt.figure(figsize=(6,3))
    color_map = plt.get_cmap('Reds_r')
    colors_red = color_map(np.linspace(0.1,0.5,7))
    color_map = plt.get_cmap('Blues_r')
    colors_blue = color_map(np.linspace(0.1, 1, 7))
    plt.subplot(2,1,1)
    plt.bar(sev_plus_columns.values(),df_1["GBDT"],color=colors_red)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.title("GBDT: Mean SEV- = %.2f"%data.loc["Mean SEV-","GBDT"],fontsize=20)
    plt.xticks([])
    plt.tight_layout()
    plt.subplot(2,1,2)
    plt.bar(sev_plus_columns.values(),df_1["All-Opt GBDT"],color=colors_blue)
    plt.gca().yaxis.set_tick_params(labelsize=16)
    plt.ylabel("Counts",fontsize=16)
    plt.xlabel("SEV-",fontsize=18)
    plt.xticks([1,2,3,4,5,6,7],fontsize=16)
    plt.title("AllOpt- GBDT: Mean SEV- = %.2f"%data.loc["Mean SEV-","All-Opt GBDT"],fontsize=20)
    plt.tight_layout()
    plt.savefig("Experiment D1/%s_GBDT_Minus.png"%data_name,dpi=500)
    plt.close()
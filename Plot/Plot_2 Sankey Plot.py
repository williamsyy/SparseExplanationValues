from psankey.sankey import sankey
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_names = {"adult":"Adult","mimic":"MIMIC","diabetes":"Diabetes","compas":"COMPAS","fico":"FICO","german":"German"}

for data_name in data_names.keys():
    print(data_name)
    # load the original dataset
    df_l1_plus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_l1lr_plus_0.csv")["sev"]
    df_l2_plus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_l2lr_plus_0.csv")["sev"]
    df_l1_minus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_l1lr_minus_0.csv")["sev"]
    df_l2_minus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_l2lr_minus_0.csv")["sev"]
    df_mlp_plus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_mlp_plus_0.csv")["sev"]
    df_mlp_minus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_mlp_minus_0.csv")["sev"]
    df_gbdt_plus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_gbdt_plus_0.csv")["sev"]
    df_gbdt_minus = pd.read_csv("../Experiments/Results/Exp0_Baseline/data/"+data_name+"_gbdt_minus_0.csv")["sev"]

    # find the best parameters 
    df_all_lr_plus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV Penalty", "Positive+ Penalty"],["All-Opt LR"]].values.reshape(-1)
    df_vol_lr_plus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV Penalty", "Positive+ Penalty"],["Vol-Opt LR"]].values.reshape(-1)
    df_all_lr_minus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV- Penalty", "Positive- Penalty"],["All-Opt LR"]].values.reshape(-1)
    
    df_all_mlp_plus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV Penalty", "Positive+ Penalty"],["All-Opt MLP"]].values.reshape(-1)
    df_all_mlp_minus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV- Penalty", "Positive- Penalty"],["All-Opt MLP"]].values.reshape(-1)

    df_all_gbdt_plus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV Penalty", "Positive+ Penalty"],["All-Opt GBDT"]].values.reshape(-1)
    df_all_gbdt_minus_best = pd.read_excel("Experiment D1 Data/%s Sample.xlsx"%data_names[data_name]).set_index(data_names[data_name]).loc[["SEV- Penalty", "Positive- Penalty"],["All-Opt GBDT"]].values.reshape(-1)

    value_map = {1:1,0.1:0.1, 0.01:0.01, 10:10.0,0.0:0}

    # find the specific datasets
    optimized_df_lr_opt_plus = pd.read_csv("../Experiments/Results/Exp1/data/%s_lr_alloptplus_%s_%s.csv"%(data_name,value_map[df_all_lr_plus_best[0]],value_map[df_all_lr_plus_best[1]]))["SEV"]
    optimized_df_lr_opt_minus = pd.read_csv("../Experiments/Results/Exp1/data/%s_lr_alloptminus_%s_%s.csv"%(data_name,value_map[df_all_lr_minus_best[0]],value_map[df_all_lr_minus_best[1]]))["SEV"]
    optimized_df_lr_vol_opt_plus = pd.read_csv("../Experiments/Results/Exp1/data/%s_lr_volopt_%s_%s.csv"%(data_name,value_map[df_vol_lr_plus_best[0]],value_map[df_vol_lr_plus_best[1]]))["SEV"]
    
    optimized_df_mlp_opt_plus = pd.read_csv("../Experiments/Results/Exp1/data/%s_mlp_alloptplus_%s_%s.csv"%(data_name,value_map[df_all_mlp_plus_best[0]],value_map[df_all_mlp_plus_best[1]]))["SEV"]
    optimized_df_mlp_opt_minus = pd.read_csv("../Experiments/Results/Exp1/data/%s_mlp_alloptminus_%s_%s.csv"%(data_name,value_map[df_all_mlp_minus_best[0]],value_map[df_all_mlp_minus_best[1]]))["SEV"]

    
    optimized_df_gbdt_opt_plus = pd.read_csv("../Experiments/Results/Exp1/data/%s_gbdt_alloptplus_%s_%s.csv"%(data_name,value_map[df_all_gbdt_plus_best[0]],value_map[df_all_gbdt_plus_best[1]]))["SEV"]
    optimized_df_gbdt_opt_minus = pd.read_csv("../Experiments/Results/Exp1/data/%s_gbdt_alloptminus_%s_%s.csv"%(data_name,value_map[df_all_gbdt_minus_best[0]],value_map[df_all_gbdt_minus_best[1]]))["SEV"]

    # plot the results
    # Plot 1: plot l2_lr vs all_opt_lr
    df_plot = pd.DataFrame(df_l2_plus)
    df_plot["opt lr"] = optimized_df_lr_opt_plus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_allopt_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_allopt.png"%data_name,dpi=500)
    plt.close()

    # Plot 2: plot l2_lr vs vol_opt_lr
    df_plot = pd.DataFrame(df_l2_plus)
    df_plot["opt lr"] = optimized_df_lr_vol_opt_plus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_volopt_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_volopt.png"%data_name,dpi=500)
    plt.close()

    # Plot 3: plot l1_lr vs all_opt_lr 
    df_plot = pd.DataFrame(df_l1_plus)
    df_plot["opt lr"] = optimized_df_lr_opt_plus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l1lr_allopt_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l1lr_allopt.png"%data_name,dpi=500)
    plt.close()

    # Plot 4: plot l1_lr vs all_opt_lr in minus
    df_plot = pd.DataFrame(df_l1_minus)
    df_plot["opt lr"] = optimized_df_lr_opt_minus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l1lr_allopt_minus_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l1lr_allopt_minus.png"%data_name,dpi=500)
    plt.close()

    # Plot 5: plot l2_lr vs all_opt_lr in minus
    df_plot = pd.DataFrame(df_l2_minus)
    df_plot["opt lr"] = optimized_df_lr_opt_minus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_allopt_minus_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_l2lr_allopt_minus.png"%data_name,dpi=500)
    plt.close()

    # Plot 6: plot mlp vs all_opt_mlp in plus
    df_plot = pd.DataFrame(df_mlp_plus)
    df_plot["opt mlp"] = optimized_df_mlp_opt_plus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_mlp_allopt_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_mlp_allopt.png"%data_name,dpi=500)
    plt.close()

    # Plot 7: plot mlp vs all_opt_mlp in minus
    df_plot = pd.DataFrame(df_mlp_minus)
    df_plot["opt mlp"] = optimized_df_mlp_opt_minus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_mlp_allopt_minus_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_mlp_allopt_minus.png"%data_name,dpi=500)
    plt.close()

    # Plot 8: plot gbdt vs all_opt_gbdt in plus
    df_plot = pd.DataFrame(df_gbdt_plus)
    df_plot["opt gbdt"] = optimized_df_gbdt_opt_plus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_gbdt_allopt_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_gbdt_allopt.png"%data_name,dpi=500)
    plt.close()

    # Plot 9: plot gbdt vs all_opt_gbdt in minus
    df_plot = pd.DataFrame(df_gbdt_minus)
    df_plot["opt gbdt"] = optimized_df_gbdt_opt_minus
    df_plot.columns = ['source','target']
    df_plot = df_plot[(df_plot["source"] != 0)|(df_plot["target"]!=0)]
    df_plot["target"] = "Opt-SEV="+df_plot["target"].astype(str) +" Count:"
    df_plot["source"] = "SEV="+df_plot["source"].astype(str) + " Count:"
    df_final = df_plot.value_counts().reset_index()
    df_final.columns = ["source","target","value"]
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=True, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_gbdt_allopt_minus_number.png"%data_name,dpi=500)
    plt.close()
    nodes, fig, ax = sankey(df_final,aspect_ratio=0.75, nodelabels=False, linklabels=False, labelsize=9, nodecmap='RdBu', nodealpha=0.8)
    plt.tight_layout()
    plt.savefig("Experiment D2/%s_gbdt_allopt_minus.png"%data_name,dpi=500)
    plt.close()
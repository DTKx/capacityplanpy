import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

root_path_report="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\03_relatorio\\springer\\tables\\"
root_path_data="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\"

# # Open pickle dict
# file_name='var_1_results.pkl'
# path=root_path_data+file_name
# infile = open(path,'rb')
# results= pickle.load(infile)
# infile.close()
# results

# Open pickle dict Results Exec
file_name='v_0_exec.pkl'
path=root_path_data+file_name
infile = open(path,'rb')
ds_exec= pickle.load(infile)
infile.close()
# Open pickle dict Results Ind
file_name='v_0_id.pkl'
path=root_path_data+file_name
infile = open(path,'rb')
ds_ind= pickle.load(infile)
infile.close()

# for key,value in ds_exec.items():
#     print(key)
#     print(value)
#     print(value[0])
# df_exec=pd.DataFrame.from_dict(ds_exec,orient="index")
# best_pareto=pd.iloc[ix_max,1]
# best_pareto=df.at[ix_max,1]
# # df_exec.reset_index()
# # ds_exec[(0,0)][0]

values_ind=[]
# values_ind_test=[]
for key,value in ds_ind.items():
    # values_ind.append(value)
    values_ind.append(value[0])
    values_ind.append(value[1])
    # print(key)
    # print(value)
    # print(value[0])

df_ind=pd.DataFrame(values_ind)
# df_ind_2=pd.DataFrame(values_ind_test)
file_name='data_test.csv'
path=root_path_data+file_name
headers=["Num_exec","Solution","Total throughput [kg]", "Max total backlog [kg]", "Mean total backlog [kg]","std dev total backlog [kg]", "Median total backlog [kg]","Min total backlog [kg]", "P(total backlog ≤ 0 kg)","Max total inventory deficit [kg]", "Mean total inventory deficit [kg]","std dev inventory deficit [kg]", "Median total inventory deficit [kg]", "Min total inventory deficit [kg]","Batches [un]","Product label","Start of USP [date]","End of DSP [date]"]
df_ind.to_csv(path,header=headers)

# df_ind.to_csv(path)

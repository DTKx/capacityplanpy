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

for key,value in ds_exec.items():
    print(key)
    print(value)
    print(value[0])
df_exec=pd.DataFrame.from_dict(ds_exec,orient="index")
best_pareto=pd.iloc[ix_max,1]
best_pareto=df.at[ix_max,1]
# df_exec.reset_index()
# ds_exec[(0,0)][0]

values_ind=[]
for key,value in ds_ind.items():
    values_ind.append(value)
    # print(key)
    # print(value)
    # print(value[1])

df_ind=pd.DataFrame(values_ind[0])

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

root_path_report="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\03_relatorio\\springer\\tables\\"
root_path_data="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\"

# Import Data
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

# # Unpack Data Execution
# values_exec=[]
# for key,value in ds_exec.items():
#     # headers=["Execution","Variant","Hipervolume","Pareto Front"]
#     values_exec.append([key[1],value[0],value[1],value[2]])
# df_exec=pd.DataFrame(values_exec)
df_exec=pd.DataFrame(ds_exec)

# # Exports as csv
# file_name='df_exec.csv'
# path=root_path_data+file_name
# headers=["Execution","Variant","Hipervolume","Pareto Front"]
# # Overwrites
# df_exec.to_csv(path,header=headers)
# # # Appends
# # df_exec.to_csv(path,header=False,mode="a")

# # Exports as csv
# file_name='EXEC.csv'
# path=root_path_data+file_name
# # headers=["Execution","Variant","Hipervolume","Pareto Front"]
# # Overwrites
# df_exec.to_csv(path)
# # df_exec.to_csv(path,header=headers)
# # # Appends
# # df_exec.to_csv(path,header=False,mode="a")


# # Unpack Data per Ind
# values_ind=[]
# for key,value in ds_ind.items():
#     # values_ind.append(value)
#     values_ind.append(value[0])
#     values_ind.append(value[1])
# df_ind=pd.DataFrame(values_ind)
# values_ind=[]
# for i in range(0,len(ds_ind)):

df_ind=pd.DataFrame(ds_ind)

# # Exports as csv
# file_name='test.csv'
# path=root_path_data+file_name
# # headers=["Execution","Variation","Solution","Total throughput [kg]", "Max total backlog [kg]", "Mean total backlog [kg]","std dev total backlog [kg]", "Median total backlog [kg]","Min total backlog [kg]", "P(total backlog ≤ 0 kg)","Max total inventory deficit [kg]", "Mean total inventory deficit [kg]","std dev inventory deficit [kg]", "Median total inventory deficit [kg]", "Min total inventory deficit [kg]","Batches [un]","Product label","Start of USP [date]","End of DSP [date]"]
# # Overwrites
# # df_ind.to_csv(path,header=headers)
# df_ind.to_csv(path)

# # Exports as csv
# file_name='df_ind.csv'
# path=root_path_data+file_name
# headers=["Variation","Num_exec","Solution","Total throughput [kg]", "Max total backlog [kg]", "Mean total backlog [kg]","std dev total backlog [kg]", "Median total backlog [kg]","Min total backlog [kg]", "P(total backlog ≤ 0 kg)","Max total inventory deficit [kg]", "Mean total inventory deficit [kg]","std dev inventory deficit [kg]", "Median total inventory deficit [kg]", "Min total inventory deficit [kg]","Batches [un]","Product label","Start of USP [date]","End of DSP [date]"]
# # Overwrites
# df_ind.to_csv(path,header=headers)
# # # Appends
# # df_ind.to_csv(path,header=False,mode="a")
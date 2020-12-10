import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

root_path_report="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\03_relatorio\\springer\\tables\\"
root_path_data="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\"

# Open pickle dict
file_name='var_1_results.pkl'
path=root_path_data+file_name
infile = open(path,'rb')
results= pickle.load(infile)
infile.close()
results

for key,value in results.items():
    print(key)
    print(value)
    value[0]

results[(100, 1, 2, 0.6, (0.04, 0.61, 0.77, 0.47))]
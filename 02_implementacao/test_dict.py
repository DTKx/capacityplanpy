import numpy as np
chromossome=10
list_c=[[i] for i in range(0,chromossome)]
genes=5
list_g=[[j] for j in range(0,genes-1)]
batches=3
list_b=[[j] for j in range(0,batches-1)]
list_c.append(list_g)
list_c[0].append(list_g)
print(list_c[0])

list_c[0]=np.append([list_c[0]],list_g)
print(list_c)
print(list_c[0][0])
list_c[0][0]=np.append([list_c[0][0]],list_b)
import collections
import datetime

batches_end_date=collections.defaultdict(list)
batches_end_date_i=collections.defaultdict(list)
s = [(0,datetime.date(2016,12,1)),(1,datetime.date(2016,12,1)),(2,datetime.date(2016,12,1)),(3,datetime.date(2016,12,1)),(0,datetime.date(2016,12,1))]
for p,d in s:
    batches_end_date_i[p].append(d)
import pandas as pd

d=[i for i in range(0,5)]
batches_end_date_i[0]=np.append(batches_end_date_i[0],d)
batches_end_date_i[1]=np.append(batches_end_date_i[1],d)

# for batch in d:
#     batches_end_date_i[0].append(batch)
print(batches_end_date_i)

a=pd.Series(pd.to_datetime(pd.DataFrame(batches_end_date_i[0]).set_index(0).index)).resample("M",convention="start")
# a=pd.Series(pd.to_datetime(batches_end_date_i[0])).resample("M",convention="start")
# a=pd.Series(pd.to_datetime(batches_end_date_i[0])).resample("M",convention="start")

# a=pd.Series(batches_end_date_i[0]).resample("M")
print(a)
s[-1][1]
list_dicts_batches_end=[]
list_dicts_batches_end.append(batches_end_date_i)
list_dicts_batches_end[0][0]
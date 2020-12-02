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

batches_end_date=collections.defaultdict(list)
s = [(1,0,0,20),(1,0,1,20),(1,0,2,20),(1,0,2,20)]
for c,g,b,d in s:
    batches_end_date[c][g][b].append(d)
p=[i in range(0,4)]
batches_end_date_i=collections.defaultdict(list)
s = [(0,20),(1,20),(2,20),(3,20),(0,20)]
for p,d in s:
    batches_end_date_i[p].append(d)
batches_end_date_i[0].append(30)
s[-1][1]
list_dicts_batches_end=[]
list_dicts_batches_end.append(batches_end_date_i)
list_dicts_batches_end[0][0]
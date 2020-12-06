# import copy
# class FooInd():
#     def __init__(self):
#         self.a=1

# class Planning():
#     def foo(self,pop):
#         print(pop.a)

#     def main():
#         ind=FooInd()
#         print(ind.a)
#         Planning().foo(copy.deepcopy(ind))
# if __name__ == "__main__":
#     Planning.main()

import numpy as np
from functools import partial
# from collections import defaultdict
# from dateutil import relativedelta
# import datetime
from numba import cuda

# a=defaultdict(partial)

# b=np.zeros(shape=(1,2))
# c=np.ones(shape=(1,2))
# a[0]=(b,c)

# print("d")
# # d1=datetime.datetime(2018,10,5)
# # d2=datetime.datetime(2019,11,6)

# # r = relativedelta.relativedelta(d2, d1)
# # print(r.months)

# # print(12*r.years)
# target_0=[6.2,6.2,9.3,9.3,12.4,12.4,15.5,21.7,21.7,24.8,21.7,24.8,27.9,21.7,24.8,24.8,24.8,27.9,27.9,27.9,31,31,34.1,34.1,27.9,27.9,27.9,27.9,34.1,34.1,31,31,21.7,15.5,6.2,0]
# target_1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2]
# target_2=[0,4.9,9.8,9.8,9.8,9.8,19.6,19.6,14.7,19.6,19.6,19.6,14.7,19.6,19.6,14.7,14.7,19.6,19.6,9.8,19.6,19.6,19.6,19.6,24.5,34.3,24.5,29.4,39.2,39.2,29.4,19.6,19.6,14.7,4.9,0]
# target_3=[22,27.5,27.5,27.5,27.5,33,33,27.5,27.5,27.5,38.5,33,33,33,33,33,27.5,33,33,33,38.5,33,38.5,33,33,33,33,44,33,33,33,33,22,11,11,5.5]
# # a=np.array(target_0,target_1,target_2,target_3)
# a=np.column_stack([target_0,target_1,target_2,target_3])

# print(a.shape)

# a=np.ones(shape=(10,4))*9
# b=np.ones(shape=(10,4))*5
# f=np.ones(shape=(10,15))*7
# mask_valid=np.zeros(shape=(10,15),dtype=bool)
# # f[f<=2]=1
# # print(f)
# mask_valid[:,0:4]=np.ones(shape=(10,4),dtype=bool)
# print(f)
# for i in range(0,9,2):
#     mask=np.random.randint(2,size=(1,4))
#     mask_invert=mask^1
#     f[i][mask_valid[i]]=a[i]*mask+b[i]*mask_invert
#     f[mask_valid[i]]=a[i]*mask+b[i]*mask_invert
#     f[mask_valid[i+1]]=b[i]*mask+a[i]*mask_invert
#     print(f)


# chromossome=np.random.randint(0,4,size=15)
# range_max=4
# pmutp=0.3
# mask=np.random.randint(0,100,size=chromossome.shape)
# ix_mut=np.where(mask<=pmutp*100)
# print(chromossome)
# chromossome[ix_mut]=chromossome[ix_mut]+1
# print(chromossome)
# mask=np.random.randint(0,100,size=chromossome.shape)
# ix_mut=np.where(mask<=pmutp*100)
# print(chromossome)
# chromossome[ix_mut]=chromossome[ix_mut]-1
# print(chromossome)

# num_genes=20
# batches=np.ones(shape=(20,num_genes),dtype=int)
# batches_new=np.zeros(shape=(60,num_genes),dtype=int)
# batches=np.vstack((batches,batches_new))
# print(batches)
# print(batches.shape)
a=[i for i in range(0,10)]
# print(len(a))
# b=[5 for i in range(0,5)]
# for i in range(0,len(b)):
#     a.append(b[i])
# print(len(a))
# print(a)
b=np.array([0,5,7])
# print(a[list(b)])
print(list(map(a.__getitem__,list(b))))
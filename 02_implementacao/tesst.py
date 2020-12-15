import numpy as np

a=np.arange(0,10).reshape(5,-1)
mask=np.zeros(shape=a.shape,dtype=bool)
mask[:,1]=True
mask[0,0]=True
mask[0,1]=False
b=np.ones(shape=(5,2))*30
print(a)
a[mask]=b[mask].copy()
print(a)
a[mask][a[mask]<b[mask]]=b[mask][a[mask]<b[mask]].copy()
print(a)
a<b

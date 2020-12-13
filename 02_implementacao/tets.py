import numpy as np

a=np.random.rand(4,3)
print(a)
y=np.argsort(a[:,0],axis=0)
print(a[y])

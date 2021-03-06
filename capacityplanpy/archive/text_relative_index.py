# import numba as nb
# import numpy as np

# @nb.njit(fastmath=True)
# def isin(b):
#   for i in range(b.shape[0]):
#     res=False
#     if (b[i]==-1):
#       res=True
#     if (b[i]==1):
#       res=True
#   return res

# #Parallelized call to isin if the data is an array of shape (n,m)
# @nb.njit(fastmath=True,parallel=True)
# def isin_arr(b):
#   res=np.empty(b.shape[0],dtype=nb.boolean)
#   for i in nb.prange(b.shape[0]):
#     res[i]=isin(b[i,:])

#   return res

# A=(np.random.randn(10,3)-0.5)*5
# A=A.astype(np.int8)
# res=isin_arr(A)
# print(res)

from numba.cuda import test
import numpy as np
import numba as nb


def remove(ar):
    ar[0] = 0
    ar[2] = 0
    return ar


@nb.njit(fastmath=True)
def set_difference(array_a,array_b):
  print("pass")
  return set(array_a).difference(set(array_b))

@nb.njit(fastmath=True)
def diff(array_a,array_b):
  a=set_difference(array_a,array_b)
  a=np.array(a)
  print(a[0])
  return a

# import genetic
num_l = 10
ix = np.arange(num_l)
ix_set=np.arange(num_l)
num_loop = 4
classified_non_domin_ix = []
for l in range(num_loop):
    print("num", l)
    a = np.ones(num_l)
    print(a)
    print(a[ix])#Ix del
    print(a[ix_set])
    a[ix] = remove(a[ix])
    print(a)
    # print(a[ix])
    ix_one = np.where(a == 0)[0]
    classified_non_domin_ix.append(ix_one)
    ix_set = np.setdiff1d(ix, ix_one)
    print(ix_set)
    print(a[ix_set])
    ix_del=np.zeros(len(ix)-len(ix_one))
    g=0
    for j in ix:
      for k in ix_one:
        if j!=k:
          ix_del[g]=ix_del

    # ix_del = np.delete(ix, ix_one)
    # ix_del=np.array(ix_one[0])
    # ix_del=diff(ix, ix_one)
    # for k in ix_one:
    #     if k not in 


    ix_del=set(ix).difference(set(ix_one))#Op2
    ix_del=np.array(ix_del)
    print(type(ix_del))
    print(ix_del)
    # print(a[ix_del])
    ix = ix_set.copy()

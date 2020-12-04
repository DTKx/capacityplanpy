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
from collections import defaultdict
from dateutil import relativedelta
import datetime


a=defaultdict(partial)

b=np.zeros(shape=(1,2))
c=np.ones(shape=(1,2))
a[0]=(b,c)

print("d")
# d1=datetime.datetime(2018,10,5)
# d2=datetime.datetime(2019,11,6)

# r = relativedelta.relativedelta(d2, d1)
# print(r.months)

# print(12*r.years)

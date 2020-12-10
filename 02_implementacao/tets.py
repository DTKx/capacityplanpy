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
import pandas as pd
# from collections import defaultdict
# from dateutil import relativedelta
# import datetime
ix_to_delete=np.array([-10])
ix_falta_classificar=np.arange(1,2)
ix_to_delete=np.append(ix_to_delete,np.where(ix_falta_classificar==0))
ix_to_delete=np.delete(ix_to_delete,0)

print(ix_to_delete)

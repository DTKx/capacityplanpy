# import datetime
# # import timedelta
# from dateutil.relativedelta import *
# #  YYYY-MM-DD.
# x=datetime.date(2020, 5, 17)
# print(x)
# use_date = x+relativedelta(months=+1)
# print(use_date)
# use_date = x+relativedelta(days=+2)
# # use_date = x + datetime.timedelta(month=+1)
# print(use_date)

# print(x<use_date)

import numpy as np

a=np.arange(0,20).reshape(5,-1)

mask=np.zeros(a.shape, dtype=bool)
mask[:,0]=True
mask[1,1]=True
print(a.shape)

print(a[mask].shape)
print(a[mask])
a_list=[a]
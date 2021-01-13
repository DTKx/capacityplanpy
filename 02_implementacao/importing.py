import csv
import numpy as np
from ast import literal_eval

path="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\00_input\\demand_distribution.txt"
# a=np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
# reader = csv.reader(open(path, "rb"), delimiter=",")
# x = list(reader)
# result = numpy.array(reader).astype("float")

# a = np.genfromtxt(path, dtype=[('myint','i8'),('myfloat','f8')]skip_header=1)

with open(path,"r") as content:
    demand = content.read()

a=np.array(literal_eval(demand))
print(a)
print(type(a))
print(a[0])
print(a[0][0])
print(type(a[0][0]))

np.savetxt("C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\00_input\\TEST.txt", a, delimiter=",")

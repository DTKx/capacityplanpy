import numpy as np
import copy

print("Base")
a = np.zeros(shape=(2, 2, 2))
c = np.ones(shape=(2, 2, 2))
b = c
a[0, :, 0] = b[0, :, 0] + c[0, :, 0]
c[0, :, 0] = c[0, :, 0] * 100
print("After (b is mutated))")
print("a", a)
print("b", b)
print("c", c)

print("Test 1 using copy()")
a = np.zeros(shape=(2, 2, 2))
c = np.ones(shape=(2, 2, 2))
b = c
a[0, :, 0] = b[0, :, 0].copy() + c[0, :, 0]
c[0, :, 0] = c[0, :, 0] * 100
print("After (b is still mutated))")
print("a", a)
print("b", b)
print("c", c)

print("Test 2 properly using copy()")
a = np.zeros(shape=(2, 2, 2))
c = np.ones(shape=(2, 2, 2))
b = c.copy()
a[0, :, 0] = b[0, :, 0] + c[0, :, 0]
c[0, :, 0] = c[0, :, 0] * 100
print("After (b is not mutated))")
print("a", a)
print("b", b)
print("c", c)

print("Test 3 not using a copy but a slice.")
a = np.zeros(shape=(2, 2, 2))
c = np.ones(shape=(2, 2, 2))
a[0, :, 0] = c[0, :, 0] + c[0, :, 0]
c[0, :, 0] = c[0, :, 0] * 100
print("After (b is not mutated))")
print("a", a)
print("b", b)
print("c", c)

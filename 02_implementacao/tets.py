import numpy as np
import inspect 

print(inspect.isbuiltin(np.zeros)
source_code = inspect.getsource(np.zeros)
print(source_code)
import numpy as np
import pandas as pd

arrays = [np.random.randn(3,4) for _ in range(10)]
print(arrays)
a=np.stack(arrays,axis=0).shape
print(a)
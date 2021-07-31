import numpy as np

factors = np.random.randn(64, 5)
factor1, factor2 = np.split(factors,2)
y = (factor1 > factor2)[:, masks].astype(np.float32)
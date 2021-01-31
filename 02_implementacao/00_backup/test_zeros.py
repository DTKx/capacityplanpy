import numpy as np
from numba import jit
import time


@jit(nopython=True, nogil=True, fastmath=True)
def calc_distributions_monte_carlo(num_monte):
    distribution_sums_deficit = np.zeros(num_monte, dtype=np.float64)  # Stores deficit distributions
    distribution_sums_backlog = np.zeros(num_monte, dtype=np.float64)  # Stores backlog distributions
    print("hey")


if __name__ == "__main__":
    num_monte = 100
    calc_distributions_monte_carlo(num_monte)

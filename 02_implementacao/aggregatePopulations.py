# import concurrent.futures
# import copy
# import cProfile
# import csv
# import datetime
# import io
# import multiprocessing
# import pickle
# import pstats
# import random
# import time
# from ast import literal_eval
# from collections import defaultdict
# from datetime import timedelta
# from itertools import product
# from pstats import SortKey

# import numpy as np
# import pandas as pd
# from dateutil import relativedelta
# from dateutil.relativedelta import *
# import numba as nb

# # from numba import jit, prange, typeof
# from pygmo import hypervolume
# from scipy import stats

# # Local Modules
# # import sys
# # # insert at 1, 0 is the script path (or '' in REPL)
# # sys.path.insert(1,'C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\')
# # import genetico_permutacao as genetico
# import genetic as gn
# import population
# from planning import Planning


def selectFrontExtractMetrics(root_path, file_name, name_var, num_fronts):
    """Preprocess executions to generate a pareto front using the already found solutions and extracts metrics that were already exported to the pkl file. 
    """

    def export_obj(obj, path):
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(path):
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    # Variables
    ref_point = [2500, 2500]  # Hipervolume calculation
    volume_max = np.prod(ref_point)  # Maximum Volume

    # Lists store results
    result_execs = []
    result_ids = []

    pop_main = load_obj(root_path + file_name)
    # Removes the first dummy one chromossome
    print("Final Num Chromossomes", pop_main.fronts.shape)
    Planning.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
    # Front Classification
    pop_main.fronts = gn.AlgNsga2._fronts(pop_main.objectives_raw, num_fronts)
    print("fronts out", pop_main.fronts)
    # Select only front 0 with no violations or front 0
    ix_vio = np.where(pop_main.backlogs[:, 6] == 0)[0]
    ix_par = np.where(pop_main.fronts == 0)[0]
    ix_pareto_novio = np.intersect1d(ix_vio, ix_par)
    if len(ix_pareto_novio) > 0:
        print(
            "Found Solutions without violations and in pareto front",
            len(ix_pareto_novio),
            ix_pareto_novio,
        )
    else:
        print(
            "No solution without violations and in front 0, passing all in front 0.",
            ix_pareto_novio,
        )
        ix_pareto_novio = ix_par
    print("Fronts In select by index", pop_main.fronts)
    print("Backlog In select by index", pop_main.backlogs[:, 6])
    Planning.select_pop_by_index(pop_main, ix_pareto_novio)
    print("Fronts out select by index", pop_main.fronts)
    print("Backlog out select by index", pop_main.backlogs[:, 6])
    print("Objectives before metrics_inversion_violations", pop_main.objectives_raw)

    # Extract Metrics

    r_exec, r_ind = pop_main.metrics_inversion_violations(
        ref_point, volume_max, num_fronts, 0, name_var, pop_main.backlogs[:, 6],
    )

    result_execs.append(r_exec)
    result_ids.append(r_ind[0])  # X
    result_ids.append(r_ind[1])  # Y
    print("Objectives after metrics_inversion_violations", pop_main.objectives_raw)
    print("Backlog Out after metrics_inversion_violations", pop_main.backlogs[:, 6])
    export_obj(pop_main, root_path + file_name)
    name_var = "v_0"
    # Export Pickle
    file_name = name_var + "_exec.pkl"
    export_obj(result_execs, root_path + file_name)

    file_name = name_var + "_id.pkl"
    export_obj(result_ids, root_path + file_name)
    print("Finish Aggregation")

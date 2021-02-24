import concurrent.futures
import copy
import cProfile
import csv
import datetime
import io
import multiprocessing
import pickle
import pstats
import random
import time
from ast import literal_eval
from collections import defaultdict
from datetime import timedelta
from itertools import product
from pstats import SortKey

import numpy as np
import pandas as pd
from dateutil import relativedelta
from dateutil.relativedelta import *
import numba as nb

# from numba import jit, prange, typeof
from pygmo import hypervolume
from scipy import stats

# Local Modules
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1,'C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\')
# import genetico_permutacao as genetico
import genetic as gn
import population
from planning import Planning

def parse_incomplete():
    """Parse incomplete executions to generate a pareto front using the already found solutions that were already exported to the pkl file. 
    """
    def export_obj(obj, path):
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(path):
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    # Parameters

    # Number of genes
    num_genes = int(25)
    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    # Number of Months
    num_months = 36
    num_fronts = 3  # Number of fronts created
    qc_max_months = 4#Max number of months

    # Inversion val to convert maximization of throughput to minimization, using a value a little bit higher than the article max 630.4
    inversion_val_throughput = 2000
    ref_point = [inversion_val_throughput + 500, 2500]
    volume_max = np.prod(ref_point)  # Maximum Volume

    # Variables
    # Variant
    var = "front_nsga,tour_vio,rein_vio,vio_back,calc_montecarlo"

    # Number of Chromossomes
    nc = [100]
    # Number of Generations
    ng = [1000]
    # Number of tour
    nt = [2]
    # Crossover Probability
    # pcross = [0.11]
    pcross = [0.5]
    # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
    pmut = [(0.04, 0.61, 0.77, 0.47)]
    num_fronts = 3  # Number of fronts created


    root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
    # root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\"
    # List of variants
    list_vars = list(product(*[nc, ng, nt, pcross, pmut]))

    # Lists store results
    result_execs = []
    result_ids = []

    times = []
    # var=0
    for v_i in list_vars:
        file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
        name_var = f"{var},{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}"

        pop_main = load_obj(root_path + file_name)
        print("Shape", pop_main.objectives_raw.shape)

        # Removes the first dummy one chromossome
        Planning.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
        print("fronts in", pop_main.fronts)
        print("Number chromo in", pop_main.num_chromossomes)
        # Front Classification
        pop_main.fronts = gn.AlgNsga2._fronts(pop_main.objectives_raw, num_fronts)
        print("fronts out", pop_main.fronts)
        # Select only front 0 with no violations or front 0
        ix_vio = np.where(pop_main.backlogs[:, 6] == 0)[0]
        ix_par = np.where(pop_main.fronts == 0)[0]
        ix_pareto_novio = np.intersect1d(ix_vio, ix_par)
        if len(ix_pareto_novio)>0:
            var = var + "metrics_front0_wo_vio"
            print("Found Solutions without violations and in pareto front", len(ix_pareto_novio), ix_pareto_novio)
        else:
            print("No solution without violations and in front 0, passing all in front 0.")
            var = var + "metrics_front0_w_vio"
            ix_pareto_novio = ix_par
        print("Selected fronts", pop_main.fronts[ix_pareto_novio])
        print("Backlog In select by index", pop_main.backlogs[:, 6])
        Planning.select_pop_by_index(pop_main, ix_pareto_novio)
        print("After function", pop_main.fronts)
        print("Objectives before metrics_inversion_violations", pop_main.objectives_raw)

        # Extract Metrics

        r_exec, r_ind = pop_main.metrics_inversion_violations(
            ref_point,
            volume_max,
            inversion_val_throughput,
            num_fronts,
            0,
            name_var,
            pop_main.backlogs[:, 6],
        )

        result_execs.append(r_exec)
        result_ids.append(r_ind[0])  # X
        result_ids.append(r_ind[1])  # Y
        print("Objectives after metrics_inversion_violations", pop_main.objectives_raw)
        print("Backlog Out after metrics_inversion_violations", pop_main.backlogs[:, 6])

        # # Reinverts again the throughput, that was modified for minimization by addying a constant
        # pop_main.objectives_raw[:, 0] = self.inversion_val_throughput - pop_main.objectives_raw[:, 0]

        export_obj(pop_main, root_path + file_name)

        # var+=1
    name_var = "v_0"
    # name_var=f"exec{n_exec}_chr{nc}_ger{ng}_tour{nt}_cross{pcross}_mut{pmut}"
    file_name = name_var + "_results.csv"
    path = root_path + file_name
    # print(f"{tempo} tempo/exec{tempo/n_exec}")
    # Export times
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        try:
            writer.writerows(times)
        except:
            writer.writerow(times)

    # Export Pickle
    file_name = name_var + "_exec.pkl"
    export_obj(result_execs, root_path + file_name)

    file_name = name_var + "_id.pkl"
    export_obj(result_ids, root_path + file_name)

    print("Finish")


if __name__ == "__main__":
    parse_incomplete()

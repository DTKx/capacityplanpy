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


class Planning:
    # Class Variables

    # General Genetic Algorithms parameters

    # Number of genes
    num_genes = int(25)

    # Problem variables

    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # Number of Months
    num_months = 36
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    # Last day of manufacturing
    last_date = datetime.date(2019, 12, 1)  # YYYY-MM-DD.
    # # List of months
    # list_months = pd.date_range(start=start_date, end=end_date, freq="MS")[1:]
    # # First day of stock calculation
    # date_stock = list_months[0]

    # Number of Monte Carlo executions Article ==1000
    num_monte = 1000
    input_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\00_input\\"

    # Process Data

    products = [0, 1, 2, 3]
    inoculation_days = dict(
        zip(products, [20, 15, 20, 26])
    )  # Source: Thesis, Inoculation is a part of the USP
    seed_production_days = dict(
        zip(products, [25, 21, 25, 23])
    )  # Source: Thesis, Seed and production is a part of the USP
    usp_days = dict(zip(products, [45, 36, 45, 49]))  # total USP days
    dsp_days = dict(zip(products, [7, 11, 7, 7]))  # total USP days
    qc_days = dict(zip(products, [90, 90, 90, 90]))  # total QC days
    qc_max_months = 4
    yield_kg_batch = dict(zip(products, [3.1, 6.2, 4.9, 5.5]))
    yield_kg_batch_ar = np.array([3.1, 6.2, 4.9, 5.5])
    # initial_stock=dict(zip(products,[18.6,0,19.6,33]))
    initial_stock = np.array([18.6, 0, 19.6, 33])
    min_batch = dict(zip(products, [2, 2, 2, 3]))
    max_batch = dict(zip(products, [50, 50, 50, 30]))
    batch_multiples = dict(zip(products, [1, 1, 1, 3]))

    target_stock = np.loadtxt(
        input_path + "target_stock.csv", delimiter=",", skiprows=1
    )  # Target Stock

    s0 = [0, 10, 16, 20]  # Setup Time
    s1 = [16, 0, 16, 20]
    s2 = [16, 10, 0, 20]
    s3 = [18, 10, 18, 0]
    setup_key_to_subkey = [{0: a, 1: b, 2: c, 3: d} for a, b, c, d in zip(s0, s1, s2, s3)]

    # Inversion val to convert maximization of throughput to minimization, using a value a little bit higher than the article max 630.4
    inversion_val_throughput = 2000

    with open(input_path + "demand_distribution.txt", "r") as content:
        demand_distribution = np.array(literal_eval(content.read()))

    # Monte Carlo

    ix_not0 = np.where(demand_distribution != 0)  # Index of values that are not zeros
    tr_demand = np.loadtxt(
        input_path + "triangular_demand.txt", delimiter=","
    )  # 1D array with only not zeros demand_distribution

    with open(input_path + "demand_montecarlo.pkl", "rb") as reader:
        demand_montecarlo = pickle.load(reader)  # Pre Calculated Monte Carlo Simulations option

    # NSGA Variables

    num_fronts = 3  # Number of fronts created
    big_dummy = 10 ** 5  # Big Dummy for crowding distance computation

    # Hypervolume parameters

    # Hypervolume Reference point
    ref_point = [inversion_val_throughput + 500, 2500]
    volume_max = np.prod(ref_point)  # Maximum Volume

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def metrics_dist_deficit(distribution_sums_deficit):
        metrics = np.array(
            [
                np.amax(distribution_sums_deficit),
                np.mean(distribution_sums_deficit),
                np.std(distribution_sums_deficit),
                np.median(distribution_sums_deficit),
                np.amin(distribution_sums_deficit),
                np.sum(distribution_sums_deficit),
            ]
        )
        return metrics

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def metrics_dist_backlog(distribution_sums_backlog, num_monte):
        metrics = np.array(
            [
                np.amax(distribution_sums_backlog),  # 0)Max total backlog months and products
                np.mean(distribution_sums_backlog),  # 1)Mean total backlog months and products
                np.std(distribution_sums_backlog),  # 2)Std Dev total backlog months and products
                np.median(distribution_sums_backlog),  # 3)Median total backlog months and products
                np.amin(distribution_sums_backlog),  # 4)Min total backlog months and products
                np.sum(distribution_sums_backlog <= 0)
                / num_monte,  # 5)Probability of Total Backlog <=0 P(total backlog<=0)
                np.median(distribution_sums_backlog),  # 6)Backlog violations
            ]
        )  # Stores backlogs and metrics
        return metrics

    def calc_inventory_objectives(self, pop):
        """Calculates Inventory levels returning the backlog and calculates the objectives the total deficit and total throughput addying to the pop attribute

        Args:
            pop (class object): Population object to calculate Inventory levels
        """
        # Creates a vector for batch/kg por the products
        pop_yield = np.vectorize(self.yield_kg_batch.__getitem__)(
            pop.products_raw
        )  # Equivalentt to         # pop_yield=np.array(list(map(self.yield_kg_batch.__getitem__,pop_products)))

        # Loop per Chromossome
        for i in range(0, len(pop.products_raw)):
            # if any(pop.batches_raw[i][pop.masks[i]] == 0):
            #     raise Exception("Invalid number of batches (0).")
            pop.objectives_raw[i, 0] = self.inversion_val_throughput - np.dot(
                pop.batches_raw[i][pop.masks[i]], pop_yield[i][pop.masks[i]]
            )  # Inversion of the Throughput by a fixed value to generate a minimization problem
            pop.objectives_raw[i, 1] = self.calc_median_deficit_backlog(
                pop, i
            )  # Adds median_deficit_i
            # a=np.sum(produced_i)
            # if pop.objectives_raw[i,0]-a>1:
            #     raise Exception("Error in Objective 1")
        return pop

    @staticmethod
    def tournament_restrictions(fronts, crowding_dist, n_parents, n_tour, violations):
        """Tournament with replacement for selection to crossover, considering those criteria:       
        1)Lowest number of constraints: 1)Lowest median Total BacklogIf draw then: 
        2)Best pareto front, if draw then:
        3)Highest Crowding Distance

        Args:
            pop (object): Population object that contains the individuals to select
            n_parents (int): Number of chromossomes to select
            n_tour (int): Number of individuals to compete during each tournament

        Returns:
            array: Array with indexes of selected individuals
        """
        # # Calculates number of violated constraints
        num_chromossomes = len(violations)

        # Arrays representing the indexes
        idx_population = np.arange(0, num_chromossomes)
        # Indexes of winners
        idx_winners = np.empty(shape=(n_parents,), dtype=int)

        # Selection all participants
        idx_for_tournament = np.random.choice(idx_population, size=n_tour * n_parents, replace=True)
        j = 0
        violations_tour = violations[idx_for_tournament].reshape(-1, n_tour)
        violations_min = np.amin(violations_tour, axis=1)
        fronts_tour = fronts[idx_for_tournament].reshape(-1, n_tour)
        fronts_min = np.amin(fronts_tour, axis=1)
        crowding_dist_tour = crowding_dist[idx_for_tournament].reshape(-1, n_tour)
        idx_for_tournament = idx_for_tournament.reshape(-1, n_tour)
        for j in range(0, n_parents):
            # Criteria
            ix_lowest_vio = np.where(
                violations_tour[j, :] == violations_min[j]
            )  # 1) Lowest Restrictions
            if len(ix_lowest_vio[0]) == 1:
                idx_winners[j] = idx_for_tournament[j][ix_lowest_vio][0]
            else:
                ix_lowest_fronts = np.where(
                    fronts_tour[j, :] == fronts_min[j]
                )  # 2)Lowest Pareto Front
                if len(ix_lowest_fronts[0]) == 1:
                    idx_winners[j] = idx_for_tournament[j][ix_lowest_fronts][0]
                else:  # 3)Highest Crowding Distance
                    ix_lowest_crowd = np.argmax(crowding_dist_tour[j, :])
                    idx_winners[j] = idx_for_tournament[j][
                        ix_lowest_crowd
                    ]  # In case of equal selects the first
        return idx_winners

    def mutation_processes(self, new_product, new_batches, new_mask, pmut):
        """Mutation Processes:
            1. To mutate a product label with a rate of pMutP. 
            2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            3. To add a new random gene to the end of the chromosome (un- conditionally).
            4. To swap two genes within the same chromosome once with a rate of pSwap .

        Args:
            new_product (array of int): Number of Products
            new_batches (array of int): Number of Batches
            new_mask (array of bools): Mask of active genes
            pmut (tuple): Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)

        Returns:
            [type]: [description]
        """

        # if (new_product>=self.num_products).any():
        #     raise Exception("Error in labels of products, labels superior than maximum defined.")
        # Active genes per chromossome
        genes_per_chromo = np.sum(new_mask, axis=1, dtype=int)
        # Loop per chromossome
        for i in range(0, len(new_product)):
            # if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")
            # # print(new_batches[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # 1. To mutate a product label with a rate of pMutP.
            # print("In label",new_product[i])
            new_product[i, 0 : genes_per_chromo[i]] = gn.Mutations._label_mutation(
                new_product[i, 0 : genes_per_chromo[i]], self.num_products, pmut[0]
            )
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # print(new_product[i])
            # 2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            # print("In add_subtract",new_batches[i])
            new_batches[i, 0 : genes_per_chromo[i]] = gn.Mutations._add_subtract_mutation(
                new_batches[i, 0 : genes_per_chromo[i]], pmut[1], pmut[2]
            )
            # print(new_batches[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # 3. To add a new random gene to the end of the chromosome (un- conditionally).
            # print(new_product[i])
            # print("In new gene",new_batches[i])
            # print(new_mask[i])
            new_product[i, genes_per_chromo[i]] = random.randint(0, self.num_products - 1)
            new_batches[i, genes_per_chromo[i]] = 1
            new_mask[i, genes_per_chromo[i]] = True
            genes_per_chromo[i] = genes_per_chromo[i] + 1
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # print(new_product[i])
            # print(new_batches[i])
            # print(new_mask[i])
            # 4. To swap two genes within the same chromosome once with a rate of pSwap .
            # print("In Swap",new_product[i])
            (
                new_product[i, 0 : genes_per_chromo[i]],
                new_batches[i, 0 : genes_per_chromo[i]],
            ) = gn.Mutations._swap_mutation(
                new_product[i, 0 : genes_per_chromo[i]],
                new_batches[i, 0 : genes_per_chromo[i]],
                pmut[3],
            )
            # print(new_product[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")

        # if (new_product>=self.num_products).any():
        #     raise Exception("Error in labels of products, labels superior than maximum defined.")
        return new_product, new_batches, new_mask

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def agg_product_batch(products, batches, masks, genes_per_chromo):
        """Aggregates product batches in case of neighbours products.
        Fix process constraints of batch min, max and multiple.
            If Batch<Min then Batch=Min, 
            If Batch>Max then Batch=Max, 
            If Batch Multiple !=Multiple then Batch round to closest given not within Min and Max

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks
        """
        # # Active genes per chromossome
        # genes_per_chromo=np.sum(masks,axis=1,dtype=int)
        if (genes_per_chromo > 1).any():
            # Loop per chromossome in population
            for j in np.arange(0, len(genes_per_chromo)):
                # if np.sum(masks[j,genes_per_chromo[j]:])>0:
                #     raise Exception("Invalid bool after number of active genes.")
                if genes_per_chromo[j] > 1:
                    # Loop per gene i in chromossome
                    i = 0
                    while i < genes_per_chromo[j] - 1:
                        if products[j, i] == products[j, i + 1]:  # Requires aggregation
                            # print(batches[j])
                            batches[j, i] = batches[j, i] + batches[j, i + 1]
                            # print(batches[j])
                            # Brings the sequence forward and sets the last value as 0
                            temp_ar = batches[j, i + 2 :].copy()
                            batches[j, i + 1 : -1] = temp_ar
                            batches[j, -1] = 0
                            # print(batches[j])
                            # print(products[j])
                            # Brings the sequence forward and sets the last value as 0
                            temp_ar = products[j, i + 2 :].copy()
                            products[j, i + 1 : -1] = temp_ar
                            products[j, -1] = 0
                            # print(products[j])
                            # print(masks[j])
                            masks[j, genes_per_chromo[j] - 1] = False
                            genes_per_chromo[j] = genes_per_chromo[j] - 1
                            # print(masks[j])
                        else:
                            i += 1
        return products, batches, masks

    def fix_batch_violations(self, products, batches):
        """Aggregates product batches in case of neighbours products.
        Fix process constraints of batch min, max and multiple.
            If Batch<Min then Batch=Min, 
            If Batch>Max then Batch=Max, 
            If Batch Multiple !=Multiple then Batch round to closest given not within Min and Max

        Args:
            products (array): Array of products
            batches (array): Array of batches
        """
        min_batch_raw = np.vectorize(self.min_batch.__getitem__)(products)
        max_batch_raw = np.vectorize(self.max_batch.__getitem__)(products)
        batch_multiples_raw = np.vectorize(self.batch_multiples.__getitem__)(products)

        # # 1)Minimum number of batches,
        mask_min = (batches < min_batch_raw) & (batches != 0)
        batches[mask_min] = min_batch_raw[mask_min].copy()
        # # 2)Maximum number of batches,
        mask_max = batches > max_batch_raw
        batches[mask_max] = max_batch_raw[mask_max].copy()
        # # 3)Multiples of number of batches
        remainder = np.remainder(batches, batch_multiples_raw)
        mask_remainder = (remainder != 0).copy()
        multiple = remainder + batches
        batches[mask_remainder] = multiple[mask_remainder].copy()
        # Max always respects the remainder, therefore no need to correct again
        return products, batches

    def fix_aggregation_batches(self, products, batches, masks):
        """Fixes Aggregation of products and maximum, minimum and multiples of batches.

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks

        Returns:
            Arrays: Fixed product, batches and masks array
        """
        genes_per_chromo = np.sum(masks, axis=1, dtype=int)  # Updates Active genes per chromossome

        products, batches, masks = self.agg_product_batch(
            products, batches, masks, genes_per_chromo
        )  # Fix Aggregation of products

        products, batches = self.fix_batch_violations(products, batches)  # Fix number of batches
        return products, batches, masks

    @staticmethod
    def merge_pop_with_offspring(pop, pop_new):
        """Appends the offspring population to the Current population.

        Args:
            pop (class object): Current Population object
            pop_new (class object): Offspring population object
        """

        pop.batches_raw = np.vstack((pop.batches_raw, pop_new.batches_raw))  # Batches
        pop.num_chromossomes = len(pop.batches_raw)
        pop.products_raw = np.vstack((pop.products_raw, pop_new.products_raw))  # Products
        pop.masks = np.vstack((pop.masks, pop_new.masks))  # Masks
        pop.start_raw = np.vstack(
            (pop.start_raw, pop_new.start_raw)
        )  # Time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        pop.end_raw = np.vstack((pop.end_raw, pop_new.end_raw))
        pop.backlogs = np.vstack((pop.backlogs, pop_new.backlogs))  # Stock backlog_i
        pop.deficit = np.vstack((pop.deficit, pop_new.deficit))  # Stock Deficit_i
        pop.objectives_raw = np.vstack(
            (pop.objectives_raw, pop_new.objectives_raw)
        )  # Objectives throughput_i,deficit_strat_i
        pop.genes_per_chromo = np.sum(
            pop.masks, axis=1, dtype=int
        )  # Genes per chromossome (Number of active campaigns per solution)
        pop.produced_month_product_individual = np.concatenate(
            (pop.produced_month_product_individual, pop_new.produced_month_product_individual),
            axis=2,
        )

        pop.fronts = np.empty(
            shape=(pop.num_chromossomes, 1), dtype=int
        )  # NSGA 2 Creates an array of fronts and crowding distance
        pop.crowding_dist = np.empty(shape=(pop.num_chromossomes, 1), dtype=int)
        return pop

    def select_pop_by_index(self, pop, ix_reinsert):
        """Selects chromossomes to maintain in pop class object, updating the class atributes given the index.

        Args:
            pop (class object): Population Class Object to reduce based on selected index ix_reinsert
            ix_reinsert (array): Indexes selected to maintain in the population class object.
        """
        pop.num_chromossomes = len(ix_reinsert)

        # Batches
        pop.batches_raw = pop.batches_raw[ix_reinsert]

        # Products
        pop.products_raw = pop.products_raw[ix_reinsert]

        # Masks
        pop.masks = pop.masks[ix_reinsert]

        # Time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        pop.start_raw = pop.start_raw[ix_reinsert]
        pop.end_raw = pop.end_raw[ix_reinsert]

        # Stock backlog_i
        pop.backlogs = pop.backlogs[ix_reinsert]

        # Stock Deficit_i
        pop.deficit = pop.deficit[ix_reinsert]

        # Objectives throughput_i,deficit_strat_i
        pop.objectives_raw = pop.objectives_raw[ix_reinsert]

        # Genes per chromossome (Number of active campaigns per solution)
        pop.genes_per_chromo = np.sum(pop.masks, axis=1, dtype=int)

        # List of dictionaries with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        pop.produced_month_product_individual = pop.produced_month_product_individual[
            :, :, ix_reinsert
        ]

        # NSGA2
        # Creates an array of fronts and crowding distance
        pop.fronts = pop.fronts[ix_reinsert]
        pop.crowding_dist = pop.crowding_dist[ix_reinsert]

    def export_obj(self, obj, path):
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, path):
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    def parse_incomplete(self):
        """Parse incomplete executions to generate a pareto front using the already found solutions that were already exported to the pkl file. 
        """
        # Parameters

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

            pop_main = self.load_obj(root_path + file_name)
            print("Shape", pop_main.objectives_raw.shape)

            # Removes the first dummy one chromossome
            self.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
            print("fronts in", pop_main.fronts)
            print("Number chromo in", pop_main.num_chromossomes)
            # Front Classification
            pop_main.fronts = gn.AlgNsga2._fronts(pop_main.objectives_raw, self.num_fronts)
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
            self.select_pop_by_index(pop_main, ix_pareto_novio)
            print("After function", pop_main.fronts)
            print("Objectives before metrics_inversion_violations", pop_main.objectives_raw)

            # Extract Metrics

            r_exec, r_ind = pop_main.metrics_inversion_violations(
                self.ref_point,
                self.volume_max,
                self.inversion_val_throughput,
                self.num_fronts,
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

            self.export_obj(pop_main, root_path + file_name)

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
        self.export_obj(result_execs, root_path + file_name)

        file_name = name_var + "_id.pkl"
        self.export_obj(result_ids, root_path + file_name)

        print("Finish")


if __name__ == "__main__":
    Planning().parse_incomplete()

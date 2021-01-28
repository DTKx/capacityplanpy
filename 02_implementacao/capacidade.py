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

# from genetic import gn.AlgNsga2, gn.Crossovers, gn.Mutations

# gn.AlgNsga2._crossover_uniform,gn.AlgNsga2._fronts,_crowding_distance


class Population:
    """Stores population attributes and methods
    """

    # Metrics per backlog deficit
    # 0)Max total months and products, 1)Mean total months and products,
    # 2)Std Dev total months and products, 3)Median total months and products,
    # 4)Min total months and products 5)Sum total months and products
    # 6)Backlog violations

    num_metrics = 7

    def __init__(
        self,
        num_genes,
        num_chromossomes,
        num_products,
        num_objectives,
        start_date,
        qc_max_months,
        num_months,
    ):
        """Initiates the current population, with a batch population,product population and a mask.
        batch population contains the number of batches, initially with only one batch
        product population contains the product being produced related to the batch number of the batch population,r randolmly assigned across different number of products
        mask dictates the current population in place, supporting a variable length structure

        Args:
            num_genes (int): Number of genes in a chromossome
            num_chromossomes (int): Number of chromossomes in population
            num_products (int): Number of products available to compose the product propulation
            num_objectives (int): Number of objectives being evaluated
            start_date (datetime): Start Date of planning
            qc_max_months (array): Additional number of months to finish quality control.
            num_months (int): Number of months of planning
        """
        self.name_variation = "-"
        self.num_chromossomes = num_chromossomes
        self.num_genes = num_genes

        # Initializes Batch with 1 batch
        self.batches_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
        self.batches_raw[:, 0] = int(1)

        # Initializes products with random allocation of products
        self.products_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
        self.products_raw[:, 0] = np.random.randint(low=0, high=num_products, size=num_chromossomes)

        # Initialize Mask of active items with only one gene
        self.masks = np.zeros(shape=(num_chromossomes, num_genes), dtype=bool)
        self.masks[:, 0] = True

        # Initializes a time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        self.start_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype="datetime64[D]")
        # self.start_raw[:,0]=start_date
        self.end_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype="datetime64[D]")

        # Initializes Stock backlog_i [kg]

        # 0)Max total backlog months and products, 1)Mean total backlog months and products,
        # 2)Std Dev total backlog months and products, 3)Median total backlog months and products,
        # 4)Min total backlog months and products 5)Sum total backlog months and products
        # 6)Backlog violations
        self.backlogs = np.zeros(shape=(num_chromossomes, self.num_metrics), dtype=float)

        # Initializes Inventory deficit per month (Objective 1, but with breakdown per month) [kg]
        # 0)Max total months and products, 1)Mean total months and products,
        # 2)Std Dev total months and products, 3)Median total months and products,
        # 4)Min total months and products 5)Sum total months and products
        self.deficit = np.zeros(shape=(num_chromossomes, self.num_metrics - 1), dtype=float)

        # Initializes the objectives throughput_i,deficit_strat_i
        self.objectives_raw = np.zeros(shape=(num_chromossomes, num_objectives), dtype=float)

        # Initializes genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo = np.sum(self.masks, axis=1, dtype=int)

        # Initialize 3d array with produced (month,product,individual)
        self.produced_month_product_individual = np.zeros(
            shape=(num_months + qc_max_months, num_products, num_genes)
        )

        # NSGA2
        # Creates an array of fronts and crowding distance
        self.fronts = np.empty(shape=(num_chromossomes, 1), dtype=int)
        self.crowding_dist = np.empty(shape=(num_chromossomes, 1), dtype=int)

    def update_genes_per_chromo(self):
        """ Updates genes per chromossome (Number of active campaigns per solution)
        """
        self.genes_per_chromo = np.sum(self.masks, axis=1, dtype=int)

    def update_new_population(self, new_products, new_batches, new_mask):
        """Updates the values of the new offspring population in the class object.

        Args:
            new_products (Array of ints): Population of product labels
            new_batches (Array of ints): Population of number of batches
            new_mask (Array of booleans): Population of active genes
        """
        # Updates new Batches values
        self.batches_raw = copy.deepcopy(new_batches)

        # Updates new Products
        if isinstance(self.products_raw[0][0], np.int32) == False:
            raise ValueError("Not int")

        self.products_raw = copy.deepcopy(new_products)

        if isinstance(self.products_raw[0][0], np.int32) == False:
            raise ValueError("Not int")

        # Updates Mask of active items with only one gene
        self.masks = copy.deepcopy(new_mask)
        self.update_genes_per_chromo()

    def extract_metrics(self, ix, num_fronts, num_exec, id_solution, name_var, ix_pareto):
        """Extract Metrics

        Args:
            ix (int): Index of the solution to verify metrics

        Returns:
            list: List with the metrics Total throughput [kg] Max total backlog [kg] Mean total backlog [kg] Median total backlog [kg] a Min total backlog [kg] P(total backlog ≤ 0 kg) 
                Max total inventory deficit [kg] Mean total inventory deficit [kg] a Median total inventory deficit [kg] Min total inventory deficit [kg]
        """
        metrics = [num_exec, name_var, id_solution]
        # Total throughput [kg]
        metrics.append(self.objectives_raw[:, 0][ix_pareto][ix])
        # Max total backlog [kg]
        metrics.append(self.backlogs[:, 0][ix_pareto][ix])
        # Mean total backlog [kg] +1stdev
        metrics.append(self.backlogs[:, 1][ix_pareto][ix])
        # Standard Dev
        metrics.append(self.backlogs[:, 2][ix_pareto][ix])
        # Median total backlog [kg]
        metrics.append(self.backlogs[:, 3][ix_pareto][ix])
        # Min total backlog [kg]
        metrics.append(self.backlogs[:, 4][ix_pareto][ix])
        # P(total backlog ≤ 0 kg)
        metrics.append(self.backlogs[:, 5][ix_pareto][ix])
        # DeltaXY (total backlog) [kg]

        # Max total inventory deficit [kg]
        metrics.append(self.deficit[:, 0][ix_pareto][ix])
        # Mean total inventory deficit [kg] +1stdev
        metrics.append(self.deficit[:, 1][ix_pareto][ix])
        # Standard Dev
        metrics.append(self.deficit[:, 2][ix_pareto][ix])
        # Median total inventory deficit [kg]
        metrics.append(self.deficit[:, 3][ix_pareto][ix])
        # Min total inventory deficit [kg]
        metrics.append(self.deficit[:, 4][ix_pareto][ix])
        # Total Deficit
        metrics.append(self.objectives_raw[:, 1][ix_pareto][ix])

        # Extra Metrics for plotting
        # Batches
        metrics.append(self.batches_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # Products
        metrics.append(self.products_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # Start of USP
        metrics.append(self.start_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # End of DSP
        metrics.append(self.end_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])

        return metrics

    def metrics_inversion_violations(
        self,
        ref_point,
        volume_max,
        inversion_val_throughput,
        num_fronts,
        num_exec,
        name_var,
        violations,
    ):
        """Extract the metrics only from the pareto front, inverts the inversion made to convert form maximization to minimization, organizes metrics and data for visualization.

        Returns:
            list: Array with metrics:
                "Hypervolume"
                Solution X "X Total throughput [kg]", "X Max total backlog [kg]", "X Mean total backlog [kg]", "X Median total backlog [kg]","X Min total backlog [kg]", "X P(total backlog ≤ 0 kg)","X Max total inventory deficit [kg]", "X Mean total inventory deficit [kg]", "X Median total inventory deficit [kg]", "X Min total inventory deficit [kg]" 
                Solution Y "Y Total throughput [kg]", "Y Max total backlog [kg]", "Y Mean total backlog [kg]", "Y Median total backlog [kg]","Y Min total backlog [kg]", "Y P(total backlog ≤ 0 kg)","Y Max total inventory deficit [kg]", "Y Mean total inventory deficit [kg]", "Y Median total inventory deficit [kg]", "Y Min total inventory deficit [kg]" Pareto Front
        """
        # Indexes
        try:
            ix_vio = np.where(violations == 0)[0]
            ix_par = np.where(self.fronts == 0)[0]
            ix_pareto = np.intersect(ix_vio, ix_par)
        except:
            ix_pareto = np.where(self.fronts == 0)[0]

        # Calculates hypervolume
        try:
            hv = hypervolume(points=self.objectives_raw[ix_pareto])
            hv_vol_norma = hv.compute(ref_point) / volume_max
        except ValueError:
            hv_vol_norma = 0
        metrics_exec = [num_exec, name_var, hv_vol_norma]
        # data_plot=[]

        # Reinverts again the throughput, that was modified for minimization by addying a constant
        self.objectives_raw[:, 0] = inversion_val_throughput - self.objectives_raw[:, 0]
        # Metrics
        ix_best_min = np.argmin(self.objectives_raw[:, 0][ix_pareto])
        ix_best_max = np.argmax(self.objectives_raw[:, 0][ix_pareto])

        metrics_id = [
            self.extract_metrics(ix_best_min, num_fronts, num_exec, "X", name_var, ix_pareto)
        ]
        metrics_id.append(
            self.extract_metrics(ix_best_max, num_fronts, num_exec, "Y", name_var, ix_pareto)
        )

        # Plot Data
        metrics_exec.append(self.objectives_raw[ix_pareto])
        return metrics_exec, metrics_id


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

    target_stock = np.loadtxt(input_path + "target_stock.csv", delimiter=",", skiprows=1)  # Target Stock

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
    tr_demand = np.loadtxt(input_path + "triangular_demand.txt", delimiter=",")  # 1D array with only not zeros demand_distribution

    with open(input_path + "demand_montecarlo.pkl", "rb") as reader:
        demand_montecarlo = pickle.load(reader)  # Pre Calculated Monte Carlo Simulations option

    # NSGA Variables

    num_fronts = 3  # Number of fronts created
    big_dummy = 10 ** 5  # Big Dummy for crowding distance computation

    # Hypervolume parameters

    # Hypervolume Reference point
    ref_point = [inversion_val_throughput + 500, 2500]
    volume_max = np.prod(ref_point)  # Maximum Volume

    def create_export_demand_not_null(self):
        with open(self.input_path + "demand_distribution.txt", "r") as content:
            demand_distribution = np.array(literal_eval(content.read()))
        # Length of rows to calculate triangular
        tr_len = len(demand_distribution[self.ix_not0])
        # Generates tr_demand
        tr_demand = np.zeros(shape=(tr_len, 3))
        for i in range(0, tr_len):
            tr_demand[i] = np.array(demand_distribution[self.ix_not0][i], dtype=np.float64)
        np.savetxt(self.input_path + "triangular_demand.txt", tr_demand, delimiter=",")

    def calc_start_end(self, pop_obj):
        """Calculates start and end dates of batch manufacturing, as well as generates (dicts_batches_month_kg) a list of dictionaries (List index = Chromossome, key=Number of products and date values os release from QC) per chromossome with release dates of each batch per product. 

        Args:
            pop_obj (Class object): Class Object of the population to be analized
        """
        if isinstance(pop_obj.products_raw[0][0], np.int32) == False:
            raise ValueError("Not int")
        # Initialize by addying the first date
        pop_obj.start_raw[:, 0] = self.start_date

        produced_i = np.zeros(
            shape=(
                self.num_months + self.qc_max_months,
                self.num_products,
                pop_obj.num_chromossomes,
            ),
            dtype=int,
        )  # Produced Month 0 is the first month of inventory batches

        for i in range(0, pop_obj.num_chromossomes):  # Loop per chromossome i
            if isinstance(pop_obj.products_raw[0][0], np.int32) == False:
                raise ValueError("Not int")
            j = 0  # Evaluates gene/Campaign zero
            qa_days = self.qc_days[pop_obj.products_raw[i][j]]
            end_date = self.start_date + timedelta(
                days=self.usp_days[pop_obj.products_raw[i, j]]
                + self.dsp_days[pop_obj.products_raw[i, j]]
            )  # End first batch=USP+DSP
            release_date = end_date + timedelta(
                days=qa_days
            )  # Release date from QA batch=USP+DSP+QA
            m = (release_date.year - self.start_date.year) * 12 + (
                release_date.month - self.start_date.month
            )
            produced_i[
                m, pop_obj.products_raw[i][j], i
            ] += 1  # Updates the month with the number of batches produced

            for n_b in range(0, pop_obj.batches_raw[i][j]):  # loop in number of batches per gene
                if isinstance(pop_obj.products_raw[0][0], np.int32) == False:
                    raise ValueError("Not int")
                end_date = end_date + timedelta(
                    days=self.dsp_days[pop_obj.products_raw[i, j]]
                )  # end_date=previous_end+DSP
                release_date = end_date + timedelta(
                    days=qa_days
                )  # Release date from QA batch=USP+DSP+QA
                m = (release_date.year - self.start_date.year) * 12 + (
                    release_date.month - self.start_date.month
                )
                produced_i[
                    m, pop_obj.products_raw[i][j], i
                ] += 1  # Updates the month with the number of batches produced

            pop_obj.end_raw[i][j] = end_date  # Add end date of DSP for the first gene

            j += 1  # Evaluates further genes
            while j < pop_obj.genes_per_chromo[i]:  # Loop per gene j starting from second gene
                if isinstance(pop_obj.products_raw[0][0], np.int32) == False:
                    raise ValueError("Not int")
                # Eval First batch
                previous_end_date = end_date  # Updates end date
                pop_obj.start_raw[i, j] = previous_end_date
                +timedelta(
                    days=self.setup_key_to_subkey[pop_obj.products_raw[i, j]][
                        pop_obj.products_raw[i, j - 1]
                    ]
                )
                -timedelta(
                    days=self.usp_days[pop_obj.products_raw[i, j]]
                )  # Add a Start Date=Previous End Date + Change Over Time - USP

                qa_days = self.qc_days[pop_obj.products_raw[i][j]]
                end_date = previous_end_date + timedelta(
                    days=self.usp_days[pop_obj.products_raw[i, j]]
                    + self.dsp_days[pop_obj.products_raw[i, j]]
                )  # End first batch=Previous enddate+USP+DSP

                if (
                    end_date > self.last_date
                ):  # Verifies if End day<Last Day ok else delete, Fix and inactivates current gene and next ones
                    end_date = previous_end_date  # Return end date to the previous for breaking
                    pop_obj.masks[i][j : pop_obj.genes_per_chromo[i]] = False
                    pop_obj.batches_raw[i][j : pop_obj.genes_per_chromo[i]] = 0
                    pop_obj.genes_per_chromo[i] = np.sum(pop_obj.masks[i])
                    break  # Break the while loop, goes to produced_i = produced_i * self.yield_kg_batch_ar
                else:  # Continues
                    release_date = end_date + timedelta(
                        days=int(qa_days)
                    )  # Release date from QA batch=USP+DSP+QA or enddate+QA
                    m = (release_date.year - self.start_date.year) * 12 + (
                        release_date.month - self.start_date.month
                    )
                    produced_i[
                        m, pop_obj.products_raw[i][j], i
                    ] += 1  # Updates the month with the number of batches produced
                    for n_b in range(
                        1, pop_obj.batches_raw[i][j]
                    ):  # loop in subsequent batches per gene
                        previous_end_date = end_date  # Updates value of previous end date
                        end_date = previous_end_date + timedelta(
                            days=self.dsp_days[pop_obj.products_raw[i, j]]
                        )  # end_date=previous_end+DSP
                        if end_date > self.last_date:  # Verifies if End day<Last Day ok else delete
                            pop_obj.batches_raw[i][j] = (
                                n_b - 1
                            )  # Stops the number of batches till the last possible
                            end_date = (
                                previous_end_date  # Return end date to the previous for breaking
                            )
                            if (
                                pop_obj.batches_raw[i][j] == 0
                            ):  # Fix and inactivates current gene and next ones
                                pop_obj.masks[i][j : pop_obj.genes_per_chromo[i]] = False
                                pop_obj.batches_raw[i][j : pop_obj.genes_per_chromo[i]] = 0
                                pop_obj.genes_per_chromo[i] = np.sum(pop_obj.masks[i])
                            elif (
                                pop_obj.masks[i][j + 1] == True
                            ):  # Next gene j+1 is active, must be inactivated
                                pop_obj.masks[i][j + 1 : pop_obj.genes_per_chromo[i]] = False
                                pop_obj.batches_raw[i][j + 1 : pop_obj.genes_per_chromo[i]] = 0
                                pop_obj.genes_per_chromo[i] = np.sum(pop_obj.masks[i])
                            break  # Break the for loop, goes to pop_obj.end_raw[i][j] = end_date
                        else:
                            release_date = end_date + timedelta(
                                days=int(qa_days)
                            )  # Release date from QA batch=USP+DSP+QA
                            m = (release_date.year - self.start_date.year) * 12 + (
                                release_date.month - self.start_date.month
                            )
                            produced_i[
                                m, pop_obj.products_raw[i][j], i
                            ] += 1  # Updates the month with the number of batches produced
                    pop_obj.end_raw[i][j] = end_date  # Add end date of first gene

                    j += 1
            produced_i[:, :, i] = (
                produced_i[:, :, i] * self.yield_kg_batch_ar
            )  # Conversion batches to kg
            if isinstance(pop_obj.products_raw[0][0], np.int32) == False:
                raise ValueError("Not int")

        pop_obj.produced_month_product_individual = produced_i  # Overwrites the old array
        pop_obj.update_genes_per_chromo()  # Updates Genes per Chromo
        return pop_obj

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_triangular_dist(demand_distribution, num_monte):
        return np.random.triangular(
            demand_distribution[0], demand_distribution[1], demand_distribution[2], size=num_monte,
        )

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_median_triangular_dist(demand_distribution, num_monte):
        n = len(demand_distribution)
        demand_i = np.zeros(shape=(n,))
        # demand_i=np.median(np.random.triangular(demand_distribution[:][0],demand_distribution[:][1],demand_distribution[:][2],size=num_monte))
        for i in np.arange(0, n):  # Loop per month
            demand_i[i] = np.median(
                np.random.triangular(
                    demand_distribution[i][0],
                    demand_distribution[i][1],
                    demand_distribution[i][2],
                    size=num_monte,
                )
            )
        return demand_i

    def calc_demand_montecarlo_to_external_file(self, n_exec_demand):
        """Performs a Montecarlo Simulation to define the Demand of products, uses a demand_distribution for containing either 0 as expected or a triangular distribution (minimum, mode (most likely),maximum) values in kg

        Args:
            n_exec_demand ([type]): Number of Executions of demand calculations
        """
        demand_dict = {}
        for i in range(0, n_exec_demand):
            demand_dict[(i)] = self.calc_triangular_dist(self.tr_demand, self.num_monte)
            print(i)
        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        # Export Pickle
        file_name = "demand_montecarlo.pkl"
        path = root_path + file_name
        file_pkl = open(path, "wb")
        pickle.dump(demand_dict, file_pkl)
        file_pkl.close()

    def load_demand_montecarlo(self, line, col):
        """Loads random demand profile generated by Monte Carlo Simulation.
        Args:
        """
        i = random.randint(0, self.num_demands - 1)
        # demand_i=np.zeros(shape=(line,col))
        # demand_i[self.ix_not0]=self.demand_montecarlo[i]
        return self.demand_montecarlo[i]

    def calc_demand_montecarlo(self):
        """Performs a Montecarlo Simulation to define the Demand of products, uses a demand_distribution for containing either 0 as expected or a triangular distribution (minimum, mode (most likely),maximum) values in kg

        Args:
        """
        demand_i = np.zeros(shape=(self.num_months, self.num_products))
        demand_i[self.ix_not0] = self.calc_median_triangular_dist(self.tr_demand, self.num_monte)
        return demand_i

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_stock(available_i, stock_i, produced_i, demand_i, backlog_i, num_months):
        """Calculates Stock per month along (over num_months) Stock=Available-Demand if any<0 Stock=0 & Back<0 = else.

        Args:
            available_i (array of int): Number of available batches per month (each column represents a month). Available=Previous Stock+Produced this month
            stock_i (array of int): Stock Available. Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            produced_i (array of int): Produced per month
            demand_i (array of int): Demand on month
            backlog_i (array of int): Backlog in the month
            num_months (int): Number of months to evaluate each column represents a month
        """
        # Loop per Months starting through 1
        for j in np.arange(1, num_months):
            # Available=Previous Stock+Produced this month
            available_i[j] = stock_i[j - 1] + produced_i[j]

            # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            stock_i[j] = available_i[j] - demand_i[j]
            # Corrects negative values
            ix_neg = np.where(stock_i[j] < 0)[0]
            if len(ix_neg) > 0:
                # Adds negative values to backlog
                # print(f"backlog in {backlog_i[j]}")
                backlog_i[j][ix_neg] = stock_i[j][ix_neg] * (int(-1))
                # print(f"backlog out {backlog_i[j]}")
                # Corrects if Stock is negative
                stock_i[j][ix_neg] = int(0)
                # print(f"backlog {backlog_i[j][ix_neg]} check if mutated after assignement of stock")
        return stock_i, backlog_i

    @staticmethod
    @nb.jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_distributions_monte_carlo(
        produced, demand_j, num_monte, num_months, num_products, target_stock, initial_stock
    ):
        """Calculates the Deficit and Backlog distributions using Monte Carlo Simulation.
        Each monte Carlo Simulation generates a total backlog (sum of all backlogs month and product of the simulation) and total deficit(sum of all deficits=Target Stock-Actual Stock month and product of the simulation), which is stored in an array.

        Args:
            produced (array of floats): Produced values per month (rows) and products (columns)
            demand_j (Array of floats): num_monte demand scenarios are created using Monte Carlo, some values of demand are not simulated, are static instead.
            num_monte (Array of floats): Number of Monte Carlo Simulations
            num_months (int): Number of months to be simulated (rows)
            num_products (int): Number of products (columns)
            target_stock (Array of floats): Target stock strategically defined. (Available should be similar to target stock)
            initial_stock (Array of floats): Stock(Month 0)

        Returns:
            [array of floats]: Returns distribution of deficit and backlog.
        """
        target_stock_copy = target_stock.copy()
        available = np.zeros(shape=(num_months, num_products), dtype=np.float64)
        available[0, :] = (
            initial_stock + produced[0, :]
        )  # Evaluates stock for Initial Month (0) Available=Previous Stock+Produced this month
        distribution_sums_deficit = np.zeros(
            num_monte, dtype=np.float64
        )  # Stores deficit distributions
        distribution_sums_backlog = np.zeros(
            num_monte, dtype=np.float64
        )  # Stores backlog distributions

        for j in nb.prange(num_monte):  # Loop per number of monte carlo simulations
            produced_j = produced.copy()  # Produced Month 0 is the first month of inventory batches
            available_j = available.copy()
            stock_j = np.zeros(
                shape=(num_months, num_products), dtype=np.float64
            )  # Stores deficit distributions
            backlog_j = np.zeros(
                shape=(num_months, num_products), dtype=np.float64
            )  # Stores backlog distributions
            deficit_strat_j = np.zeros(
                shape=(num_months, num_products), dtype=np.float64
            )  # Stores deficit distributions

            stock_j[0, :] = (
                available_j[0, :] - demand_j[0, :, j]
            )  # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            ix_neg = np.where(stock_j[0, :] < 0)
            num_neg = len(ix_neg[0])
            if num_neg > 0:  # Corrects negative values
                backlog_j[0, :][ix_neg] = (stock_j[0, :][ix_neg]) * (
                    -1
                )  # Adds negative values to backlog
                # print("backlog", backlog_j)
                # print("stock_i", stock_j)
                for ix in nb.prange(num_neg):
                    stock_j[0, ix_neg[0][ix]] = 0.0  # Corrects if Stock is negative
                # print("backlog", backlog_j)
                # print("stock_i", stock_j)

            for k in nb.prange(
                1, num_months
            ):  # Calculates for the rest of months Stock Loop per Months starting through 1
                available_j[k] = (
                    stock_j[k - 1] + produced_j[k]
                )  # Available=Previous Stock+Produced this month

                stock_j[k] = (
                    available_j[k] - demand_j[k, :, j]
                )  # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else

                ix_neg = np.where(stock_j[k] < 0)
                num_neg = len(ix_neg[0])
                if num_neg > 0:  # Corrects negative values
                    # Adds negative values to backlog
                    # print("backlog in",backlog_j[k])
                    # print("STOCK in",stock_j[k])
                    backlog_j[k][ix_neg] = (stock_j[k][ix_neg]) * (int(-1))
                    # Corrects if Stock is negative
                    for n in nb.prange(num_neg):
                        stock_j[k][ix_neg[0][n]] = 0.0
                    # print("backlog out",backlog_j[k])
                    # print("STOCK out",stock_j[k])
            deficit_strat_j = (
                stock_j - target_stock_copy
            )  # Minimise the median total inventory deicit, i.e. cumulative ◦ Maximise the total production throughput. differences between the monthly product inventory levels and the strategic inventory targets.
            # Cumulative sum of the differences be- tween the product inventory levels and the corresponding strategic monthly targets whenever the latter are greater than the former.
            distribution_sums_backlog[j] = np.sum(backlog_j)
            ix_neg = np.where(deficit_strat_j < 0)
            num_neg = len(ix_neg[0])
            sum_deficit = 0.0
            if num_neg > 0:  # Sums negative numbers
                for n in nb.prange(num_neg):
                    sum_deficit += deficit_strat_j[ix_neg[0][n], ix_neg[1][n]]
                # print("backlog out",backlog_j[k])
                # print("STOCK out",stock_j[k])
            distribution_sums_deficit[j] = sum_deficit * (-1)
            # distribution_sums_deficit[j] = -1.0 * np.sum(
            #     deficit_strat_j[np.where(deficit_strat_j < 0.0)]
            # )
        return distribution_sums_backlog, distribution_sums_deficit

    def calc_median_deficit_backlog(self, pop, i):
        """Calculates the Objective Deficit and backlog of distribution considering a Monte Carlo Simulation of demand.

        Args:
            pop (Population Class): Population object from class Population
            i (int): Index of individual being evaluated

        Returns:
            float: Median of objective deficit
        """
        n_tr_distributions = len(
            self.tr_demand
        )  # number of different simulations needed to calculate one deficit

        row, col = self.demand_distribution.shape

        demand_j = np.zeros(shape=(row, col, self.num_monte), dtype=float)
        for k in np.arange(0, n_tr_distributions):  # Loop per triangular distributions to simulate
            demand_j[self.ix_not0[0][k], self.ix_not0[1][k]] = self.calc_triangular_dist(
                self.demand_distribution[self.ix_not0][k], self.num_monte
            )

        distribution_sums_backlog, distribution_sums_deficit = self.calc_distributions_monte_carlo(
            pop.produced_month_product_individual[
                :, :, i
            ],  # Produced Month 0 is the first month of inventory batches
            demand_j,
            self.num_monte,
            self.num_months,
            self.num_products,
            self.target_stock,
            self.initial_stock,
        )

        pop.backlogs[i] = self.metrics_dist_backlog(
            distribution_sums_backlog, self.num_monte
        )  # Stores backlogs and metrics
        pop.deficit[i] = self.metrics_dist_deficit(
            distribution_sums_deficit
        )  # Stores deficit metrics

        return pop.deficit[i][3]  # MedianDeficit

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

    def main(self, num_exec, num_chromossomes, num_geracoes, n_tour, perc_crossover, pmut):
        print("START Exec number:", num_exec)
        # 1) Random parent population is initialized with its attributes
        pop = Population(
            self.num_genes,
            num_chromossomes,
            self.num_products,
            self.num_objectives,
            self.start_date,
            self.qc_max_months,
            self.num_months,
        )
        # 1.1) Initializes class object for Offspring Population
        # Number of chromossomes for crossover, guarantees an even number
        n_parents = int(num_chromossomes * perc_crossover)
        if n_parents % 2 == 1:
            n_parents = n_parents + 1
        pop_offspring = Population(
            self.num_genes,
            n_parents,
            self.num_products,
            self.num_objectives,
            self.start_date,
            self.qc_max_months,
            self.num_months,
        )
        # 1.2) Creates start and end date from schedule assures only batches with End date<Last day of manufacturing

        # 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        pop = self.calc_start_end(pop)

        # 3)Calculate inventory levels and objectives
        pop = self.calc_inventory_objectives(pop)
        # if (pop.objectives_raw<0).any():
        #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")

        # print(
        #     "Metrics backlog all population: amax",
        #     np.amax(pop.backlogs[:, 0]),  # 0)Max total backlog months and products
        #     " mean",
        #     np.mean(pop.backlogs[:, 1]),  # 1)Mean total backlog months and products
        #     " median",
        #     np.median(pop.backlogs[:, 3]),  # 3)Median total backlog months and products
        #     " min",
        #     np.amin(pop.backlogs[:, 4]),  # 4)Min total backlog months and products
        #     " median",
        #     np.median(pop.backlogs[:, 6]),  # 6)Backlog violations
        # )

        # 4)Front Classification
        objectives_raw_copy = pop.objectives_raw.copy()
        pop.fronts = gn.AlgNsga2._fronts(objectives_raw_copy, self.num_fronts)

        # 5) Crowding Distance
        objectives_raw_copy = pop.objectives_raw.copy()
        fronts_copy = pop.fronts.copy()
        pop.crowding_dist = gn.AlgNsga2._crowding_distance(
            objectives_raw_copy, fronts_copy, self.big_dummy
        )
        for i_gen in range(0, num_geracoes):
            print("Generation ", i_gen)

            #     for i in range(0,len(pop.products_raw)):
            #         if any(pop.batches_raw[i][pop.masks[i]]==0):
            #             raise Exception("Invalid number of batches (0).")
            #         if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #             raise Exception("Invalid bool after number of active genes.")
            # 6)Selection for Crossover Tournament
            backlogs_copy = pop.backlogs[:, 6].copy()
            fronts_copy = pop.fronts.copy()
            crowding_dist_copy = pop.crowding_dist.copy()
            ix_to_crossover = self.tournament_restrictions(
                fronts_copy, crowding_dist_copy, n_parents, n_tour, backlogs_copy
            )

            if isinstance(pop.products_raw[0][0], np.int32) == False:
                raise ValueError("Not int")
            # 7)Crossover
            # 7.1 Sorts Selected by number of genes
            genes_per_chromo_copy = pop.genes_per_chromo.copy()
            ix_to_crossover = ix_to_crossover[np.argsort(genes_per_chromo_copy[ix_to_crossover])]
            # 7.2 Creates a new population for offspring population crossover and calls uniform crossover
            # new_products,new_batches,new_mask=gn.Crossovers._crossover_uniform(copy.deepcopy(pop.products_raw[ix_to_crossover]),copy.deepcopy(pop.batches_raw[ix_to_crossover]),copy.deepcopy(pop.masks[ix_to_crossover]),copy.deepcopy(pop.genes_per_chromo),perc_crossover)
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")
            products_raw_copy = pop.products_raw.copy()
            batches_raw_copy = pop.batches_raw.copy()
            masks_copy = pop.masks.copy()
            if isinstance(products_raw_copy[0][0], np.int32) == False:
                raise ValueError("Not int")
            new_products, new_batches, new_mask = gn.Crossovers._crossover_uniform(
                products_raw_copy[ix_to_crossover],
                batches_raw_copy[ix_to_crossover],
                masks_copy[ix_to_crossover],
                perc_crossover,
            )

            # 8)Mutation
            if isinstance(pop.products_raw[0][0], np.int32) == False:
                raise ValueError("Not int")
            if isinstance(new_products[0][0], np.int32) == False:
                raise ValueError("Not int")
            new_products, new_batches, new_mask = self.mutation_processes(
                new_products, new_batches, new_mask, pmut
            )
            if isinstance(new_products[0][0], np.int32) == False:
                raise ValueError("Not int")
            if isinstance(pop.products_raw[0][0], np.int32) == False:
                raise ValueError("Not int")

            # 9)Aggregate batches with same product neighbours
            new_products, new_batches, new_mask = self.fix_aggregation_batches(
                new_products, new_batches, new_mask
            )
            for i in range(0, len(new_products)):
                if any(new_batches[i][new_mask[i]] == 0):
                    raise Exception("Invalid number of batches (0).")
                if np.sum(new_mask[i][~new_mask[i]]) > 0:
                    raise Exception("Invalid bool after number of active genes.")

            # 10) Merge populations Current and Offspring
            pop_offspring.update_new_population(new_products, new_batches, new_mask)
            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 11) 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
            pop_offspring = self.calc_start_end(pop_offspring)
            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")
            # print("Backlog before calc_inventory offspring",pop.backlogs[:,6])

            # 12) 3)Calculate inventory levels and objectives
            pop_offspring = self.calc_inventory_objectives(pop_offspring)

            # print(
            #     "Metrics backlog all population offspring: amax",
            #     np.amax(pop_offspring.backlogs[:, 0]),  # 0)Max total backlog months and products
            #     " mean",
            #     np.mean(pop_offspring.backlogs[:, 1]),  # 1)Mean total backlog months and products
            #     " median",
            #     np.median(
            #         pop_offspring.backlogs[:, 3]
            #     ),  # 3)Median total backlog months and products
            #     " min",
            #     np.amin(pop_offspring.backlogs[:, 4]),  # 4)Min total backlog months and products
            #     " median",
            #     np.median(pop_offspring.backlogs[:, 6]),  # 6)Backlog violations
            # )

            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # if (pop_offspring.objectives_raw<0).any():
            #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")
            # 13) Merge Current Pop with Offspring
            # pop_offspring_copy=copy.deepcopy(pop_offspring)
            pop = self.merge_pop_with_offspring(pop, pop_offspring)
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")
            # if (pop.objectives_raw<0).any():
            #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")
            # print("Backlog after merging offspring",pop.backlogs[:,6])

            # print(
            #     "Metrics backlog all population after merge : amax",
            #     np.amax(pop.backlogs[:, 0]),  # 0)Max total backlog months and products
            #     " mean",
            #     np.mean(pop.backlogs[:, 1]),  # 1)Mean total backlog months and products
            #     " median",
            #     np.median(pop.backlogs[:, 3]),  # 3)Median total backlog months and products
            #     " min",
            #     np.amin(pop.backlogs[:, 4]),  # 4)Min total backlog months and products
            #     " median",
            #     np.median(pop.backlogs[:, 6]),  # 6)Backlog violations
            # )

            # 14) 4)Front Classification
            objectives_raw_copy = pop.objectives_raw.copy()
            pop.fronts = gn.AlgNsga2._fronts(objectives_raw_copy, self.num_fronts)

            # 15) 5) Crowding Distance
            objectives_copy = pop.objectives_raw.copy()
            fronts_copy = pop.fronts.copy()
            pop.crowding_dist = gn.AlgNsga2._crowding_distance(
                objectives_copy, fronts_copy, self.big_dummy
            )

            # 16) Linear Reinsertion

            # 16.1) Selects indexes to maintain
            # Calculates number of violated constraints
            backlogs_copy = np.copy(pop.backlogs[:, 6])
            crowding_dist_copy = np.copy(pop.crowding_dist)
            fronts_copy = np.copy(pop.fronts)
            ix_reinsert = gn.AlgNsga2._index_linear_reinsertion_nsga_constraints(
                backlogs_copy, crowding_dist_copy, fronts_copy, num_chromossomes,
            )

            # 16.2) Remove non reinserted chromossomes from pop
            ix_reinsert_copy = np.copy(ix_reinsert)
            self.select_pop_by_index(pop, ix_reinsert_copy)
        return pop

    def export_obj(self, obj, path):
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, path):
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    def run_parallel(self):
        """Runs with Multiprocessing.
        """
        # Parameters
        # Number of executions
        n_exec = 2
        n_exec_ite = range(0, n_exec)

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
        pcross = [0.11]
        # pcross=[0.5]
        # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
        pmut = [(0.04, 0.61, 0.77, 0.47)]

        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"

        # List of variants
        list_vars = list(product(*[nc, ng, nt, pcross, pmut]))

        # Lists store results
        result_execs = []
        result_ids = []

        times = []
        # var=0
        for v_i in list_vars:
            name_var = f"{var},{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}"
            # Creates a dummy pop with one chromossome to concatenate results
            pop_main = Population(
                self.num_genes,
                1,
                self.num_products,
                self.num_objectives,
                self.start_date,
                self.qc_max_months,
                self.num_months,
            )
            pop_main.name_variation = name_var

            t0 = time.perf_counter()
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                for pop_exec in executor.map(
                    Planning().main,
                    n_exec_ite,
                    [v_i[0]] * n_exec,
                    [v_i[1]] * n_exec,
                    [v_i[2]] * n_exec,
                    [v_i[3]] * n_exec,
                    [v_i[4]] * n_exec,
                ):
                    print("In merge pop exec", pop_exec.fronts)
                    print("Backlog In merge", pop_exec.backlogs[:, 6])

                    print("In merge pop main", pop_main.fronts)
                    pop_main = self.merge_pop_with_offspring(pop_main, pop_exec)
                    print("Out merge pop main", pop_main.fronts)
                    print("Backlog Out merge", pop_main.backlogs[:, 6])

                    file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
                    self.export_obj(pop_main, root_path + file_name)

            # Removes the first dummy one chromossome
            self.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
            print("fronts in", pop_main.fronts)
            # Front Classification
            pop_main.fronts = gn.AlgNsga2._fronts(pop_main.objectives_raw, self.num_fronts)
            print("fronts out", pop_main.fronts)
            # Select only front 0 with no violations or front 0
            try:
                ix_vio = np.where(pop_main.backlogs[:, 6] == 0)[0]
                ix_par = np.where(pop_main.fronts == 0)[0]
                ix_pareto_novio = np.intersect(ix_vio, ix_par)
                var = var + "metrics_front0_wo_vio"
            except:
                print("No solution without violations, passing all in front 0.")
                var = var + "metrics_front0_w_vio"
                ix_pareto_novio = np.where(pop_main.fronts == 0)[0]
            print("Selected fronts", pop_main.fronts[ix_pareto_novio])
            print("Backlog In select by index", pop_main.backlogs[:, 6])
            self.select_pop_by_index(pop_main, ix_pareto_novio)
            print("After function", pop_main.fronts)
            print("Objectives before metrics_inversion_violations", pop_main.objectives_raw)

            # Extract Metrics

            # # Reinverts again the throughput, that was modified for minimization by addying a constant
            # self.objectives_raw[:, 0] = inversion_val_throughput - self.objectives_raw[:, 0]

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

            file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
            self.export_obj(pop_main, root_path + file_name)

            tf = time.perf_counter()
            delta_t = tf - t0
            print("Total time ", delta_t, "Per execution", delta_t / n_exec)
            times.append([v_i, delta_t, delta_t / n_exec])

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

    def run_cprofile():
        """Runs without multiprocessing.
        """
        num_exec = 1
        num_chromossomes = 100
        num_geracoes = 100
        n_tour = 2
        pcross = 0.50
        # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
        pmut = (0.04, 0.61, 0.77, 0.47)
        t0 = time.perf_counter()

        # pop_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)
        # cProfile.runctx("results,num_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)", globals(), locals())

        pr = cProfile.Profile()
        pr.enable()
        pr.runctx(
            "pop_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)",
            globals(),
            locals(),
        )
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        file_name = "cprofile.txt"
        path = root_path + file_name
        ps.print_stats()
        with open(path, "w+") as f:
            f.write(s.getvalue())
        tf = time.perf_counter()
        delta_t = tf - t0
        print("Total time ", delta_t)


if __name__ == "__main__":
    # Planning.run_cprofile()
    Planning().run_parallel()
    # Saves Monte Carlo Simulations
    # Planning().calc_demand_montecarlo_to_external_file(5000)

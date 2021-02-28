import concurrent.futures
import cProfile
import csv
import datetime
import io
import pickle
import pstats
import random
from time import perf_counter
from ast import literal_eval
from datetime import timedelta
from itertools import product
from pstats import SortKey
import numpy as np
from numba import jit, prange, typeof
from pygmo import hypervolume

# from scipy import stats
import tracemalloc


# Local Modules
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1,'C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\')
from genetic import Helpers, Mutations, Crossovers, AlgNsga2
from population import Population
import logging
import os
from errors import CountError, InvalidValuesError
import gc

LOG_FILENAME = "planning.log"
filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), LOG_FILENAME)
logging.basicConfig(
    filename=filepath, filemode="w", level=logging.ERROR
)  # Defines the path and level of log file


class Planning:
    def __init__(
        self,
        num_genes,
        num_products,
        num_objectives,
        # start_date,
        qc_max_months,
        num_months,
        num_fronts,
    ):
        """Initializes Planning class.

        Args:
            num_genes (int): Number of genes per chromossome
            num_products (int): Number of different products
            num_objectives (int): Number of different objectives
            # start_date (datetime): Start date of planning
            qc_max_months (int): Number of maximum months
            num_months (int): Number of months
            num_fronts (int): Number of fronts
        """
        self.num_genes = num_genes
        self.num_products = num_products
        self.num_objectives = num_objectives
        # self.start_date = start_date
        self.qc_max_months = qc_max_months
        self.num_months = num_months
        self.num_fronts = num_fronts

    # Class Variables
    # General Genetic Algorithms parameters
    # # Number of genes
    # num_genes = int(25)

    # Problem variables

    # # Number of products
    # num_products = int(4)
    # # Number of Objectives
    # num_objectives = 2
    # # Number of Months
    # num_months = 36
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    # Last day of manufacturing
    last_date = datetime.date(2019, 12, 1)  # YYYY-MM-DD.

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
    # qc_max_months = 4
    yield_kg_batch = dict(zip(products, [3.1, 6.2, 4.9, 5.5]))
    yield_kg_batch_ar = np.array([3.1, 6.2, 4.9, 5.5])
    # initial_stock=dict(zip(products,[18.6,0,19.6,33]))
    initial_stock = np.array([18.6, 0, 19.6, 33])
    min_batch = (2, 2, 2, 3)
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

    # num_fronts = 3  # Number of fronts created
    big_dummy = 10 ** 5  # Big Dummy for crowding distance computation

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
        Start USP=End Date USP (last batch) + Changeover(opt)
        End

        Args:
            pop_obj (Class object): Class Object of the population to be analized
        """
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

            for n_b in range(1, pop_obj.batches_raw[i][j]):  # loop in number of batches per gene
                # if n_b==14:
                #     print("hey")
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
                # Eval First batch
                previous_end_date = end_date  # Updates end date
                start_usp = previous_end_date + timedelta(
                    days=self.setup_key_to_subkey[pop_obj.products_raw[i, j]][
                        pop_obj.products_raw[i, j - 1]
                    ]
                    - self.usp_days[pop_obj.products_raw[i, j]]
                )  # Add a Start Date=Previous End Date + Change Over Time - USP

                pop_obj.start_raw[i, j] = start_usp
                qa_days = self.qc_days[pop_obj.products_raw[i][j]]
                end_date = start_usp + timedelta(
                    days=self.usp_days[pop_obj.products_raw[i, j]]
                    + self.dsp_days[pop_obj.products_raw[i, j]]
                )  # End first batch=USP Start Date+USP+DSP

                if (
                    end_date > self.last_date
                ):  # Verifies if End day<Last Day ok else delete, Fix and inactivates current gene and next ones
                    end_date = previous_end_date  # Return end date to the previous for breaking
                    pop_obj.masks[i][j : pop_obj.genes_per_chromo[i]] = False
                    pop_obj.batches_raw[i][j : pop_obj.genes_per_chromo[i]] = 0
                    pop_obj.products_raw[i][j : pop_obj.genes_per_chromo[i]] = 0
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

        pop_obj.produced_month_product_individual = produced_i  # Overwrites the old array
        pop_obj.update_genes_per_chromo()  # Updates Genes per Chromo
        return pop_obj

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_triangular_dist(demand_distribution, num_monte):
        return np.random.triangular(
            demand_distribution[0], demand_distribution[1], demand_distribution[2], size=num_monte,
        )

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def calc_median_triangular_dist(demand_distribution, num_monte):
        n = len(demand_distribution)
        demand_i = np.zeros(shape=(n,))
        # demand_i=np.median(np.random.triangular(demand_distribution[:][0],demand_distribution[:][1],demand_distribution[:][2],size=num_monte))
        for i in prange(0, n):  # Loop per month
            demand_i[i] = np.median(
                np.random.triangular(
                    demand_distribution[i][0],
                    demand_distribution[i][1],
                    demand_distribution[i][2],
                    size=num_monte,
                )
            )
        return demand_i

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
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
        for j in prange(1, num_months):
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
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
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

        for j in prange(num_monte):  # Loop per number of monte carlo simulations
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

            sum_backlog_j = 0.0  # Stores sum of backlog per individual
            sum_deficit_j = 0.0  # Stores sum of deficit per individual

            stock_j[0, :] = (
                available_j[0, :] - demand_j[0, :, j]
            )  # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else

            # Corrects stock negative values and adds to backlog for month 0
            for p in prange(num_products):  # Loop per product
                if stock_j[0, p] < 0:
                    backlog_j[0, p] = stock_j[0, p] * (-1.0)  # Adds negative values to backlog
                    sum_backlog_j += backlog_j[0, p]
                    stock_j[0, p] = 0.0  # Corrects if Stock is negative

            for k in prange(
                1, num_months
            ):  # Calculates for the rest of months Stock Loop per Months starting through 1
                available_j[k] = (
                    stock_j[k - 1] + produced_j[k]
                )  # Available=Previous Stock+Produced this month

                stock_j[k] = (
                    available_j[k] - demand_j[k, :, j]
                )  # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else

                # Corrects stock negative values and adds to backlog
                for p in prange(num_products):  # Loop per product
                    if stock_j[k, p] < 0:
                        backlog_j[k, p] = stock_j[k, p] * (-1.0)  # Adds negative values to backlog
                        sum_backlog_j += backlog_j[k, p]
                        stock_j[k, p] = 0.0  # Corrects if Stock is negative

                    deficit_strat_j[k, p] = (
                        stock_j[k, p] - target_stock_copy[k, p]
                    )  # Minimise the median total inventory deicit, i.e. cumulative ◦ Maximise the total production throughput. differences between the monthly product inventory levels and the strategic inventory targets.
                    if deficit_strat_j[k, p] < 0:  # Sums negative numbers
                        sum_deficit_j += deficit_strat_j[k, p] * (-1)

            # Cumulative sum of the differences be- tween the product inventory levels and the corresponding strategic monthly targets whenever the latter are greater than the former.
            distribution_sums_backlog[j] = sum_backlog_j  # total backlog (amount of missed orders)
            distribution_sums_deficit[
                j
            ] = sum_deficit_j  # Total Deficit (Difference between Stock-Target)
        return distribution_sums_backlog, distribution_sums_deficit

    def create_demand_montecarlo(self):
        """Create a demand array using triangular distribution.

        Returns:
            [type]: [description]
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
        return demand_j

    def calc_median_deficit_backlog(self, pop, i):
        """Calculates the Objective Deficit and backlog of distribution considering a Monte Carlo Simulation of demand.

        Args:
            pop (Population Class): Population object from class Population
            i (int): Index of individual being evaluated

        Returns:
            float: Median of objective deficit
        """
        demand_j = self.create_demand_montecarlo()
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
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
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
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
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
            if any(pop.batches_raw[i][pop.masks[i]] == 0):
                expression = f"any({pop.batches_raw[i][pop.masks[i]] == 0})"
                e = f"Invalid number of batches (0).\n Batches: {pop.batches_raw[i]} \n Masks  {pop.masks[i]} \n Position {i})"
                logging.error(
                    InvalidValuesError(expression, e), exc_info=True
                )  # Adds Exception to log file
                raise InvalidValuesError(expression, e)  # Raise

            produced = np.dot(pop.batches_raw[i][pop.masks[i]], pop_yield[i][pop.masks[i]])
            pop.objectives_raw[i, 0] = produced * (
                -1.0
            )  # Inversion of the Throughput by a fixed value to generate a minimization problem

            pop.objectives_raw[i, 1] = self.calc_median_deficit_backlog(
                pop, i
            )  # Adds median_deficit_i

            if pop.objectives_raw[i, 0] > 0:
                expression = f"{pop.objectives_raw[i, 0] < 0}"
                e = f"Invalid value of Objective 1 (Positive value).\n Produced: {produced} \n Index {i} \n objective {pop.objectives_raw[i, 0]})"
                logging.error(
                    InvalidValuesError(expression, e), exc_info=True
                )  # Adds Exception to log file
                raise InvalidValuesError(expression, e)  # Raise

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

        # Active genes per chromossome
        genes_per_chromo = np.sum(new_mask, axis=1, dtype=int)
        # Loop per chromossome
        for i in range(0, len(new_product)):
            # 1. To mutate a product label with a rate of pMutP.
            # print("In label",new_product[i])
            new_product[i, 0 : genes_per_chromo[i]] = Mutations._label_mutation(
                new_product[i, 0 : genes_per_chromo[i]], self.num_products, pmut[0]
            )
            # print(new_product[i])
            # 2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            # print("In add_subtract",new_batches[i])
            (
                new_batches[i],
                new_product[i],
                new_mask[i],
                genes_per_chromo[i],
            ) = Mutations._add_subtract_mutation(
                new_batches[i], new_product[i], new_mask[i], genes_per_chromo[i], pmut[1], pmut[2]
            )
            # print(new_batches[i])
            # 3. To add a new random gene to the end of the chromosome (un- conditionally).
            # print(new_product[i])
            # print("In new gene",new_batches[i])
            # print(new_mask[i])
            new_product[i, genes_per_chromo[i]] = random.randint(0, self.num_products - 1)
            new_batches[i, genes_per_chromo[i]] = 1
            new_mask[i, genes_per_chromo[i]] = True
            genes_per_chromo[i] = genes_per_chromo[i] + 1
            # print(new_product[i])
            # print(new_batches[i])
            # print(new_mask[i])
            # 4. To swap two genes within the same chromosome once with a rate of pSwap .
            # print("In Swap",new_product[i])
            (
                new_product[i, 0 : genes_per_chromo[i]],
                new_batches[i, 0 : genes_per_chromo[i]],
            ) = Mutations._swap_mutation(
                new_product[i, 0 : genes_per_chromo[i]],
                new_batches[i, 0 : genes_per_chromo[i]],
                pmut[3],
            )
            # print(new_product[i])

        return new_product, new_batches, new_mask

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def agg_product_batch(products, batches, masks, genes_per_chromo):
        """Aggregates product batches in case of neighbours products.

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks
        """
        # Loop per chromossome in population
        for j in prange(0, len(genes_per_chromo)):
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

    def fix_batch_violations(self, products, batches, masks):
        """Aggregates product batches in case of neighbours products.
        Fix process constraints of batch min, max and multiple.
            If Batch<Min then Batch=>Inactivate or set to minimum, 
            If Batch>Max then Batch=Max, 
            If Batch Multiple !=Multiple then Batch round to closest given not within Min and Max

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks with state of activation
        """
        # # 1)Multiples of number of batches
        batch_multiples_raw = np.vectorize(self.batch_multiples.__getitem__)(products)
        remainder = np.remainder(batches, batch_multiples_raw)
        mask_remainder = (remainder != 0).copy()
        multiple = remainder + batches
        batches[mask_remainder] = multiple[mask_remainder].copy()

        # # 2)Maximum number of batches,
        max_batch_raw = np.vectorize(self.max_batch.__getitem__)(products)
        mask_max = batches > max_batch_raw
        batches[mask_max] = max_batch_raw[mask_max].copy()
        # # 3)Minimum number of batches
        batches, masks, products = self.removeBelowMinimumBatches(
            batches, masks, self.min_batch, products
        )
        return products, batches, masks

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def removeBelowMinimumBatches(batches, masks, min_batch_raw, products):
        """May remove batches that are below the minimum, if a random number is below threshhold, batch is removed (as long as the number of genes is higher than 1). Otherwise the number of batches is set to the minimum.
        The threshold allows for inactivation or activation of the batch.

        Args:
            batches (Array of ints): Array containing number of batches
            products (Array of ints): Array containing product label
            masks (Array of bools): Contains activation state
            min_batch_raw (tuple of ints): Minimum accepted value of batches per product (0,1,2,..)

        Returns:
            [type]: [description]
        """
        removalThreshold = 0.5
        genesPerChromo = np.sum(masks, axis=1)
        for j in prange(len(genesPerChromo)):  # Loop per individual
            i = 0
            while i < genesPerChromo[j]:  # Loop per genes
                k = 0  # Correct index
                # print("i",i)
                if batches[j, i] < min_batch_raw[products[j, i]]:  # Value below minimum
                    probaRemove = (
                        np.random.rand()
                    )  # Probability of keeping the minimum value, If <50% deactivate gene
                    if (probaRemove < removalThreshold) | (
                        genesPerChromo[j] == 1
                    ):  # Set to minimum value, if below threshold of there is only one gene
                        batches[j, i] = min_batch_raw[products[j, i]]
                    else:  # Remove
                        # print("In batches",batches[j])
                        # print(batches[j][masks[j]])
                        temp_ar = batches[j, i + 1 :].copy()
                        batches[j, i:-1] = temp_ar  # Brings the sequence forward and sets the
                        batches[j, -1] = 0  # last value as 0
                        # print(batches[j])
                        # print(products[j])
                        temp_ar = products[j, i + 1 :].copy()  # Adjust Products
                        products[j, i:-1] = temp_ar
                        products[j, -1] = 0
                        # print(products[j])
                        # print(masks[j])
                        masks[j, (genesPerChromo[j] - 1)] = False
                        genesPerChromo[j] -= 1
                        # print(masks[j])
                        # print(batches[j][masks[j]])
                        k = 1
                i += 1 - k  # Corrects if a batch was removed
            for k in prange(genesPerChromo[j]):  # Verifies Invalid number of batches
                if batches[j, k] == 0:
                    raise Exception("Invalid number of batches active(0).")
        return batches, masks, products

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

        products, batches, masks = self.fix_batch_violations(
            products, batches, masks
        )  # Fix number of batches
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

    @staticmethod
    def select_pop_by_index(pop, ix_reinsert):
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

    def main(
        self,
        num_exec,
        num_chromossomes,
        num_geracoes,
        n_tour,
        perc_crossover,
        pmut,
        root_path,
        file_name,
    ):
        print("START Exec number:", num_exec)
        t0 = perf_counter()
        # 1) Random parent population is initialized with its attributes
        pop = Population(
            self.num_genes,
            num_chromossomes,
            self.num_products,
            self.num_objectives,
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
            self.qc_max_months,
            self.num_months,
        )
        # 1.2) Creates start and end date from schedule assures only batches with End date<Last day of manufacturing

        # 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        pop = self.calc_start_end(pop)

        # 3)Calculate inventory levels and objectives
        pop = self.calc_inventory_objectives(pop)

        # 4)Front Classification
        objectives_raw_copy = pop.objectives_raw.copy()
        pop.fronts = AlgNsga2._fronts(objectives_raw_copy, self.num_fronts)

        # 5) Crowding Distance
        objectives_raw_copy = pop.objectives_raw.copy()
        fronts_copy = pop.fronts.copy()
        pop.crowding_dist = AlgNsga2._crowding_distance(
            objectives_raw_copy, fronts_copy, self.big_dummy
        )
        for i_gen in range(0, num_geracoes):
            # print("Generation ", i_gen)

            # 6)Selection for Crossover Tournament
            backlogs_copy = pop.backlogs[:, 6].copy()
            fronts_copy = pop.fronts.copy()
            crowding_dist_copy = pop.crowding_dist.copy()
            ix_to_crossover = self.tournament_restrictions(
                fronts_copy, crowding_dist_copy, n_parents, n_tour, backlogs_copy
            )

            # 7)Crossover
            # 7.1 Sorts Selected by number of genes
            genes_per_chromo_copy = pop.genes_per_chromo.copy()
            ix_to_crossover = ix_to_crossover[np.argsort(genes_per_chromo_copy[ix_to_crossover])]
            # 7.2 Creates a new population for offspring population crossover and calls uniform crossover
            products_raw_copy = pop.products_raw.copy()
            batches_raw_copy = pop.batches_raw.copy()
            masks_copy = pop.masks.copy()
            new_products, new_batches, new_mask = Crossovers._crossover_uniform(
                products_raw_copy[ix_to_crossover],
                batches_raw_copy[ix_to_crossover],
                masks_copy[ix_to_crossover],
                perc_crossover,
            )

            # 8)Mutation
            new_products, new_batches, new_mask = self.mutation_processes(
                new_products, new_batches, new_mask, pmut
            )

            # 9)Aggregate batches with same product neighbours
            new_products, new_batches, new_mask = self.fix_aggregation_batches(
                new_products, new_batches, new_mask
            )

            # 10) Merge populations Current and Offspring
            pop_offspring.update_new_population(new_products, new_batches, new_mask)

            # 11) 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
            pop_offspring = self.calc_start_end(pop_offspring)

            # 12) 3)Calculate inventory levels and objectives
            pop_offspring = self.calc_inventory_objectives(pop_offspring)
            # 13) Merge Current Pop with Offspring
            pop = Planning.merge_pop_with_offspring(pop, pop_offspring)

            # 14) 4)Front Classification
            objectives_raw_copy = pop.objectives_raw.copy()
            pop.fronts = AlgNsga2._fronts(objectives_raw_copy, self.num_fronts)

            # 15) 5) Crowding Distance
            objectives_copy = pop.objectives_raw.copy()
            fronts_copy = pop.fronts.copy()
            pop.crowding_dist = AlgNsga2._crowding_distance(
                objectives_copy, fronts_copy, self.big_dummy
            )

            # 16) Linear Reinsertion

            # 16.1) Selects indexes to maintain
            # Calculates number of violated constraints
            backlogs_copy = np.copy(pop.backlogs[:, 6])
            crowding_dist_copy = np.copy(pop.crowding_dist)
            fronts_copy = np.copy(pop.fronts)
            ix_reinsert = AlgNsga2._index_linear_reinsertion_nsga_constraints(
                backlogs_copy, crowding_dist_copy, fronts_copy, num_chromossomes,
            )

            # 16.2) Remove non reinserted chromossomes from pop
            ix_reinsert_copy = np.copy(ix_reinsert)
            self.select_pop_by_index(pop, ix_reinsert_copy)

        pop_main = Helpers._load_obj(root_path + file_name)
        print("In merge num_chromossomes", pop_main.num_chromossomes)
        pop_main = Planning.merge_pop_with_offspring(pop_main, pop)
        print("Out merge num_chromossomes", pop_main.num_chromossomes)

        Helpers._export_obj(pop_main, root_path + file_name)
        # del pop_main
        # del pop
        t1 = perf_counter()
        print("Exec", num_exec, "Time", t1 - t0)
        gc.collect()
        return t1 - t0


def selectFrontExtractMetrics(root_path, file_name, name_var, num_fronts):
    """Preprocess executions to generate a pareto front using the already found solutions and extracts metrics that were already exported to the pkl file. 
    """
    # Variables
    ref_point = [2500, 2500]  # Hipervolume calculation
    volume_max = np.prod(ref_point)  # Maximum Volume

    # Lists store results
    result_execs = []
    result_ids = []

    pop_main = Helpers._load_obj(root_path + file_name)
    # Removes the first dummy one chromossome
    print("Final Num Chromossomes", pop_main.fronts.shape)
    Planning.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
    # Front Classification
    pop_main.fronts = AlgNsga2._fronts(pop_main.objectives_raw, num_fronts)
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
    Helpers._export_obj(pop_main, root_path + file_name)
    name_var = "v_0"
    # Export Pickle
    file_name = name_var + "_exec.pkl"
    Helpers._export_obj(result_execs, root_path + file_name)

    file_name = name_var + "_id.pkl"
    Helpers._export_obj(result_ids, root_path + file_name)
    print("Finish Aggregation")


def run_parallel(numExec, numGenerations, maxWorkers):
    """Run main function using Multiprocessing.
    """
    # Parameters
    n_exec_ite = range(0, numExec)

    # Number of genes
    num_genes = int(25)
    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # # Start date of manufacturing
    # start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    qc_max_months = 4  # Max number of months
    # Number of Months
    num_months = 36
    num_fronts = 3  # Number of fronts created

    # # Inversion val to convert maximization of throughput to minimization, using a value a little bit higher than the article max 630.4
    # ref_point = [2500, 2500]
    # volume_max = np.prod(ref_point)  # Maximum Volume

    # Variables
    # Variant
    var = "front_nsga,tour_vio,rein_vio,vio_back,calc_montecarlo"

    # Number of Chromossomes
    nc = [100]
    ng = [numGenerations]  # Number of Generations

    # Number of tour
    nt = [2]
    # Crossover Probability
    # pcross = [0.11]
    pcross = [0.5]
    # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
    pmut = [(0.04, 0.61, 0.77, 0.47)]

    root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"

    # List of variants
    list_vars = list(product(*[nc, ng, nt, pcross, pmut]))

    # Lists store results
    times = []
    for v_i in list_vars:
        name_var = f"{var},{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}"
        # Creates a dummy pop with one chromossome to concatenate results
        pop_main = Population(
            # num_genes, 1, num_products, num_objectives, start_date, qc_max_months, num_months,
            num_genes,
            1,
            num_products,
            num_objectives,
            qc_max_months,
            num_months,
        )
        pop_main.name_variation = name_var
        file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
        Helpers._export_obj(pop_main, root_path + file_name)
        del pop_main  # 1)Is it a best practice delete an object after exporting and then load, export and del again?

        t0 = perf_counter()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        print("Entering")
        timeExecution = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            for time in executor.map(
                Planning(
                    num_genes,
                    num_products,
                    num_objectives,
                    # start_date,
                    qc_max_months,
                    num_months,
                    num_fronts,
                ).main,
                n_exec_ite,
                [v_i[0]] * numExec,
                [v_i[1]] * numExec,
                [v_i[2]] * numExec,
                [v_i[3]] * numExec,
                [v_i[4]] * numExec,
                [root_path] * numExec,
                [file_name] * numExec,
            ):
                timeExecution.append(time)

        tf = perf_counter()
        delta_t = tf - t0
        print("Total time ", delta_t, "Per execution", delta_t / numExec)
        times.append([v_i, delta_t, delta_t / numExec, timeExecution])
        selectFrontExtractMetrics(
            root_path, file_name, name_var, num_fronts
        )  # Makes the fronts calculation and extracts metricts
    name_var = "v_0"
    # name_var=f"exec{numExec}_chr{nc}_ger{ng}_tour{nt}_cross{pcross}_mut{pmut}"
    file_name = name_var + "_results.csv"
    path = root_path + file_name
    # print(f"{tempo} tempo/exec{tempo/numExec}")
    # Export times
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        try:
            writer.writerows(times)
        except:
            writer.writerow(times)


def run_cprofile(numExec, numGenerations, maxWorkers):
    """Runs without multiprocessing.
    """
    t0 = perf_counter()

    pr = cProfile.Profile()
    pr.enable()
    pr.runctx(
        "run_parallel(numExec,numGenerations,maxWorkers)", globals(), locals(),
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
    tf = perf_counter()
    delta_t = tf - t0
    print("Total time ", delta_t)
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)


def runCprofileMain(numExec, num_geracoes):
    """Runs without multiprocessing.
    """
    root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
    file_name = "cProfileMain.pkl"

    # Number of genes
    num_genes = int(25)
    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    qc_max_months = 4  # Max number of months
    # Number of Months
    num_months = 36
    num_fronts = 3  # Number of fronts created

    numExec = 4
    n_exec_ite = range(0, numExec)

    num_chromossomes = 100
    n_tour = 2
    pcross = 0.50
    # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
    pmut = (0.04, 0.61, 0.77, 0.47)
    t0 = perf_counter()
    # Creates a dummy pop with one chromossome to concatenate results
    pop_main = Population(
        # num_genes, 1, num_products, num_objectives, start_date, qc_max_months, num_months,
        num_genes,
        1,
        num_products,
        num_objectives,
        qc_max_months,
        num_months,
    )
    file_name = "cProfileMain.pkl"
    pop_main.name_variation = file_name
    Helpers._export_obj(pop_main, root_path + file_name)
    del pop_main  # 1)Is it a best practice delete an object after exporting and then load, export and del again?
    # myPlan = Planning(
    #     num_genes, num_products, num_objectives, start_date, qc_max_months, num_months, num_fronts
    # )
    # for time in map(myPlan.main,n_exec_ite,[num_chromossomes] * numExec,[num_geracoes] * numExec,[n_tour] * numExec,[pcross] * numExec,[pmut] * numExec,[root_path]*numExec,[file_name]*numExec):
    #     print(time)

    pr = cProfile.Profile()
    pr.enable()
    myPlan = Planning(
        num_genes, num_products, num_objectives, start_date, qc_max_months, num_months, num_fronts,
    )

    pr.runctx(
        "for time in map(myPlan.main,n_exec_ite,[num_chromossomes] * numExec,[num_geracoes] * numExec,[n_tour] * numExec,[pcross] * numExec,[pmut] * numExec,[root_path]* numExec,[file_name]* numExec):print(time)",
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
    tf = perf_counter()
    delta_t = tf - t0
    print("Total time ", delta_t)
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)


if __name__ == "__main__":
    # Planning().run_cprofile()
    numExec = 2  # Number of executions
    numGenerations = 1000  # Number of executions
    maxWorkers = 1  # Local parallelization Maximum number of threads
    # run_parallel(numExec, numGenerations, maxWorkers)
    run_cprofile(numExec, numGenerations, maxWorkers)
    # runCprofileMain(numExec, numGenerations)

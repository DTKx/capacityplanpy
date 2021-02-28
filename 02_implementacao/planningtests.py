import unittest
import numpy as np
import copy
import pickle
from planning import Planning
from population import Population
import datetime
import genetic as gn
import matplotlib.pyplot as plt
from scipy.stats import mode
import random


def load_obj(path):
    with open(path, "rb") as input:
        obj = pickle.load(input)
    return obj


class PlanningTests(unittest.TestCase):
    path_data = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\03_testing\\"

    @classmethod
    def setUpClass(cls):
        print("setupClass")

    def setUp(self):
        print("setUp")

    def test_Population__init__(self):
        """Tests generation of random values __init__ of Population class.
        """
        print("Population__init__")
        num_chromossomes = 1000
        num_products = 4
        num_tests = 1000
        count_products = {x: 0 for x in range(4)}
        for i in range(num_tests):
            pop = Population(1, num_chromossomes, num_products, 1, 2, 2)
            for j in range(num_products):
                count_products[j] += np.sum(pop.products_raw[:, 0] == j)
            del pop  # Deletes object
        total = sum(count_products.values(), 0.0)
        count_products = {key: val / total for key, val in count_products.items()}
        # print(count_products.values())
        delta = 0.1
        message = "First and second are not almost equal."  # error message in case if test case got failed
        for prob in count_products.values():
            self.assertAlmostEqual(prob, 1.0 / float(num_products), None, message, delta)

    def tearDown(self):
        print("tearDown")

    # def setUp(self):
    #     print("setUp")

    # def test_create_demand_montecarlo(self):
    #     """Tests function create_demand_montecarlo, evaluates if demand generated:
    #     1) Equal to demand expected.
    #     2) Generates distributions within the expected triangular distribution range.
    #     """
    #     print("create_demand_montecarlo")
    #     num_tests = 1000
    #     n_tr_distributions = len(
    #         Planning().tr_demand
    #     )  # number of different values with triangular distributions
    #     demand_distribution = Planning().demand_distribution
    #     ix_not0 = Planning().ix_not0

    #     for i in range(num_tests):
    #         demand_j = Planning().create_demand_montecarlo()
    #         ix_0 = np.where(demand_distribution == 0)
    #         self.assertTrue(
    #             (demand_j[ix_0] == 0).all()
    #         )  # Assures demand have the expected value==0, for each value and Monte Carlo Simulation

    #     for k in np.arange(
    #         n_tr_distributions
    #     ):  # Loop per triangular distributions assures values in expected range of triangular distribution.
    #         triangular_params = demand_distribution[ix_not0[0][k], ix_not0[1][k]]
    #         self.assertTrue(
    #             (demand_j[ix_not0[0][k], ix_not0[1][k]] >= triangular_params[0]).all()
    #             & (demand_j[ix_not0[0][k], ix_not0[1][k]] <= triangular_params[2]).all()
    #         )
    #         min_dist = min(demand_j[ix_not0[0][k], ix_not0[1][k]])
    #         mode_dist = mode(np.around(demand_j[ix_not0[0][k], ix_not0[1][k]], 2))[0][0]
    #         max_dist = max(demand_j[ix_not0[0][k], ix_not0[1][k]])
    #         delta = 0.75 * abs(triangular_params[0] - triangular_params[2])
    #         # delta = abs(
    #         #     0.25
    #         #     * (
    #         #         max(
    #         #             abs(triangular_params[0] - triangular_params[1]),
    #         #             abs(triangular_params[2] - triangular_params[1]),
    #         #         )
    #         #     )
    #         # )
    #         message = "Triangular Distribution is outside expected ranges."  # error message in case if test case got failed
    #         self.assertAlmostEqual(triangular_params[0], min_dist, None, message, delta)
    #         self.assertAlmostEqual(triangular_params[1], mode_dist, None, message, delta)
    #         self.assertAlmostEqual(triangular_params[2], max_dist, None, message, delta)

    # def tearDown(self):
    #     print("tearDown")

    def setUp(self):
        print("setUp")

    def test_calc_start_end(self):
        """Tests start and end of production batches, by comparing with a calculated example.
        """
        print("calc_start_end")
        num_chromossomes = 1

        # Number of genes
        num_genes = int(25)
        # Number of products
        num_products = int(4)
        # Number of Objectives
        num_objectives = 2
        # Start date of manufacturing
        start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
        qc_max_months = 4#Max number of months
        # Number of Months
        num_months = 36
        num_fronts = 3  # Number of fronts created

        myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

        def parser_calc_start_end(path):
            with open(path) as f:
                lines = f.readlines()
                test = f.read().splitlines()
                num_lines = len(lines)
                data_input = np.zeros(shape=(num_lines - 1, 2))  # -1 Given label
                data_output = np.zeros(
                    shape=(num_lines - 1, 2), dtype="datetime64[D]"
                )  # -1 Given label
                for i in range(0, num_lines - 1):
                    batch, prod, start, end = lines[i + 1].split(",")
                    data_input[i, 0] = int(batch)
                    data_input[i, 1] = int(prod)
                    data_output[i, 0] = datetime.date(
                        int(start.split("/")[2]), int(start.split("/")[0]), int(start.split("/")[1])
                    )  # YYYY-MM-DD.
                    data_output[i, 1] = datetime.date(
                        int(end.split("/")[2]), int(end.split("/")[0]), int(end.split("/")[1])
                    )  # YYYY-MM-DD. Avoids endlines
            return data_input, data_output

        data_input, data_output = parser_calc_start_end(self.path_data + "calc_start_end.csv")
        num_data = len(data_input)

        # Loads Solution to population
        pop = Population(
            num_genes,
            num_chromossomes,
            num_products,
            num_objectives,
            qc_max_months,
            num_months,
        )
        pop.batches_raw[0, 0:num_data] = data_input[:, 0]
        pop.products_raw[0, 0:num_data] = data_input[:, 1]
        pop.masks[0, 0:num_data] = True
        pop.update_genes_per_chromo()

        pop = myPlan.calc_start_end(pop)  # Call function
        pop.start_raw
        pop.end_raw

        self.assertTrue((pop.start_raw[0][0:num_data] == data_output[:, 0]).all())  # Start
        self.assertTrue((pop.end_raw[0][0:num_data] == data_output[:, 1]).all())  # End

    def tearDown(self):
        print("tearDown")

    # def setUp(self):
    #     print("setUp")

    # def test_calc_distributions_monte_carlo_cuda(self):
    #     """Tests function calc_distributions_monte_carlo_cuda, comparing to a manually calculated result.
    #     """
    #     print("calc_distributions_monte_carlo")
    #     produced = np.loadtxt(
    #         self.path_data + "distributions_monte_carlo_produced.csv", delimiter=",", skiprows=1
    #     )
    #     demand_val = np.loadtxt(
    #         self.path_data + "distributions_monte_carlo_demand.csv", delimiter=","
    #     )
    #     demand = np.zeros(shape=(demand_val.shape[0], demand_val.shape[1], 2))
    #     demand[:, :, 0] = demand_val.copy()
    #     demand[:, :, 1] = demand_val
    #     distribution_sums_backlog_solution = 0
    #     distribution_sums_deficit_solution = 413.32

    #     # Number of genes
    #     num_genes = int(25)
    #     # Number of products
    #     num_products = int(4)
    #     # Number of Objectives
    #     num_objectives = 2
    #     # Start date of manufacturing
    #     start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    #     qc_max_months = 4#Max number of months
    #     # Number of Months
    #     num_months = 36
    #     num_fronts = 3  # Number of fronts created

    #     myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

    #     (
    #         distribution_sums_backlog,
    #         distribution_sums_deficit,
    #     ) = myPlan.calc_distributions_monte_carlo_cuda(
    #         produced,  # Produced Month 0 is the first month of inventory batches
    #         demand,
    #         2,
    #         num_months,
    #         num_products,
    #         myPlan.target_stock,
    #         myPlan.initial_stock,
    #     )
    #     delta = 0.1
    #     message = "First and second backlog are not almost equal."  # error message in case if test case got failed
    #     self.assertAlmostEqual(
    #         distribution_sums_backlog_solution, distribution_sums_backlog[0], None, message, delta
    #     )
    #     message = "First and second deficit values are not almost equal."  # error message in case if test case got failed
    #     print(distribution_sums_deficit[0])
    #     self.assertAlmostEqual(
    #         distribution_sums_deficit_solution, distribution_sums_deficit[0], None, message, delta
    #     )

    # def tearDown(self):
    #     print("tearDown")


    def setUp(self):
        print("setUp")

    def test_calc_distributions_monte_carlo(self):
        """Tests function calc_distributions_monte_carlo, comparing to a manually calculated result.
        """
        print("calc_distributions_monte_carlo")
        produced = np.loadtxt(
            self.path_data + "distributions_monte_carlo_produced.csv", delimiter=",", skiprows=1
        )
        demand_val = np.loadtxt(
            self.path_data + "distributions_monte_carlo_demand.csv", delimiter=","
        )
        demand = np.zeros(shape=(demand_val.shape[0], demand_val.shape[1], 2))
        demand[:, :, 0] = demand_val.copy()
        demand[:, :, 1] = demand_val
        distribution_sums_backlog_solution = 0
        distribution_sums_deficit_solution = 413.32

        # Number of genes
        num_genes = int(25)
        # Number of products
        num_products = int(4)
        # Number of Objectives
        num_objectives = 2
        # Start date of manufacturing
        start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
        qc_max_months = 4#Max number of months
        # Number of Months
        num_months = 36
        num_fronts = 3  # Number of fronts created

        myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

        (
            distribution_sums_backlog,
            distribution_sums_deficit,
        ) = myPlan.calc_distributions_monte_carlo(
            produced,  # Produced Month 0 is the first month of inventory batches
            demand,
            2,
            num_months,
            num_products,
            myPlan.target_stock,
            myPlan.initial_stock,
        )
        delta = 0.1
        message = "First and second backlog are not almost equal."  # error message in case if test case got failed
        self.assertAlmostEqual(
            distribution_sums_backlog_solution, distribution_sums_backlog[0], None, message, delta
        )
        message = "First and second deficit values are not almost equal."  # error message in case if test case got failed
        self.assertAlmostEqual(
            distribution_sums_deficit_solution, distribution_sums_deficit[0], None, message, delta
        )

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test_tournament_restrictions(self):
        """Tests the tournament function to evaluate if the mean of the violations is decreasing after the tournament. In other words, verify if the lowest violations individuals are indeed being selected.
        """
        print("tournament_restrictions")
        violations = np.loadtxt(
            self.path_data + "tournament_restrictions_violations.txt", delimiter=","
        )
        crowd_dist = np.loadtxt(
            self.path_data + "tournament_restrictions_crowding_dist_copy.txt", delimiter=","
        )
        fronts = np.loadtxt(
            self.path_data + "tournament_restrictions_fronts_copy.txt", delimiter=","
        )
        n_tests = 10  # Number of tests
        n_parents = len(fronts) // 2
        n_tour = 2
        mean_violations = np.mean(violations)

        # Number of genes
        num_genes = int(25)
        # Number of products
        num_products = int(4)
        # Number of Objectives
        num_objectives = 2
        # Start date of manufacturing
        start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
        qc_max_months = 4#Max number of months
        # Number of Months
        num_months = 36
        num_fronts = 3  # Number of fronts created

        myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

        for i in range(n_tests):
            violations_copy = violations.copy()
            crowding_dist_copy = crowd_dist.copy()
            fronts_copy = fronts.copy()
            ix_to_crossover = myPlan.tournament_restrictions(
                fronts_copy, crowding_dist_copy, n_parents, n_tour, violations_copy
            )
            self.assertLess(np.mean(violations[ix_to_crossover]), mean_violations)

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test_mutation_processes(self):
        """Tests the mutation_processes.
        """
        print("mutation_processes")
        num_tests = 3
        num_products = 4
        pmut_list = [
            (random.random(), random.random(), random.random(), random.random())
            for i in range(num_tests)
        ]
        pmut_list.append((0.0, 0.0, 0.0, 0.0))
        pmut_list.append((1.0, 1.0, 1.0, 1.0))

        # Number of genes
        num_genes = int(25)
        # Number of products
        num_products = int(4)
        # Number of Objectives
        num_objectives = 2
        # Start date of manufacturing
        start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
        qc_max_months = 4#Max number of months
        # Number of Months
        num_months = 36
        num_fronts = 3  # Number of fronts created
        myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

        for pmut in pmut_list:  # Loops for different mutation rates
            num_genes = random.randint(2, 25)
            num_chromossomes = random.randint(50, 100)
            batches_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
            products_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
            masks = np.zeros(shape=(num_chromossomes, num_genes), dtype=bool)
            genes_per_chromo = np.zeros(shape=(num_chromossomes,), dtype=int)
            for i in range(num_chromossomes):  # Initializes random arrays
                genes_per_chromo[i] = random.randint(
                    0, num_genes - 1
                )  # Number of active genes per chromo
                batches_raw[i, 0 : genes_per_chromo[i]] = np.random.randint(
                    1, 50, size=genes_per_chromo[i]
                )
                products_raw[:, 0 : genes_per_chromo[i]] = np.random.randint(
                    low=0, high=num_products, size=genes_per_chromo[i]
                )
                masks[i, 0 : genes_per_chromo[i]] = True
                self.assertFalse(
                    (masks[i, genes_per_chromo[i] :] != False).any()
                )  # If true Invalid bool after number of active genes.
                self.assertFalse(
                    (batches_raw[i, genes_per_chromo[i] :] > 0).any()
                )  # If true Invalid number of batches (0).

            products_raw, batches_raw, masks = myPlan.mutation_processes(
                products_raw, batches_raw, masks, pmut
            )
            genes_per_chromo = np.sum(masks, axis=1, dtype=int)

            for i in range(
                num_chromossomes
            ):  # Verify if mutation is causing any unexpected behaviour
                self.assertFalse(
                    np.sum(masks[i, genes_per_chromo[i] :] != False) > 0
                )  # If true Invalid bool after number of active genes.
                self.assertFalse(
                    np.sum(batches_raw[i, genes_per_chromo[i] :]) > 0
                )  # If true Invalid number of batches (0).
            self.assertFalse(
                (products_raw >= num_products).any()
            )  # If true Error in labels of products, labels superior than maximum defined.

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test_select_pop_by_index(self):
        """Tests if:
        1)selection of pop by index is indeed reducing population violations data in n_tests.
        2)Verify invalid number of batches
        3)Verify invalid value of active genes
        """
        # Number of genes
        num_genes = int(25)
        # Number of products
        num_products = int(4)
        # Number of Objectives
        num_objectives = 2
        # Start date of manufacturing
        start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
        qc_max_months = 4#Max number of months
        # Number of Months
        num_months = 36
        num_fronts = 3  # Number of fronts created

        myPlan=Planning(num_genes,num_products,num_objectives,qc_max_months,num_months,num_fronts)

        n_tests = 100  # Number of tests
        for i in range(n_tests):
            pop = load_obj(self.path_data + "select_pop_by_index_pop.pkl")
            violations_copy = np.copy(pop.backlogs[:, 6])
            crowding_dist_copy = np.copy(pop.crowding_dist)
            fronts_copy = np.copy(pop.fronts)

            mean_violations = np.mean(violations_copy)

            ix_sel = gn.AlgNsga2._index_linear_reinsertion_nsga_constraints(
                violations_copy, crowding_dist_copy, fronts_copy, len(fronts_copy) // 2
            )  # Input Violations,Crowding Distance, Fronts
            self.assertLess(
                np.mean(violations_copy[ix_sel]), mean_violations
            )  # Verifies if selected indexes indeed reduce violations mean
            myPlan.select_pop_by_index(pop, ix_sel)
            self.assertLess(
                np.mean(pop.backlogs[:, 6]), mean_violations
            )  # Verifies if selected individuals indeed reduce violations mean
            for i in range(0, len(pop.products_raw)):
                self.assertFalse(
                    any(pop.batches_raw[i][pop.masks[i]] == 0)
                )  # Verify invalid number of batches (0)
                self.assertFalse(
                    np.sum(pop.masks[i][pop.genes_per_chromo[i] :]) > 0
                )  # Verify invalid value of active genes (If true Invalid bool after number of active genes.)

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test_merge_pop_with_offspring(self):
        """Tests shape and content of merged populations.
        """
        print("merge_pop_with_offspring")
        
        pop = load_obj(self.path_data + "merge_pop_with_offspring_pop_1.pkl")
        pop_copy = load_obj(self.path_data + "merge_pop_with_offspring_pop_1.pkl")
        pop2 = load_obj(self.path_data + "merge_pop_with_offspring_pop_2.pkl")

        pop_chromo=pop.num_chromossomes
        pop_genes=pop.num_genes
        num_metrics=pop.backlogs.shape[1]
        num_objectives=pop.objectives_raw.shape[1]
        max_months,num_products=pop.produced_month_product_individual.shape[:2]
        pop2_chromo=pop2.num_chromossomes

        Planning.merge_pop_with_offspring(pop,pop2)

        #Batches
        self.assertEqual(pop.batches_raw.shape,(pop_chromo+pop2_chromo,pop_genes))
        self.assertEqual((pop.batches_raw[0:pop_chromo]).all(),(pop_copy.batches_raw).all())
        self.assertEqual((pop.batches_raw[pop_chromo:]).all(),(pop2.batches_raw).all())
        #Chromossomes
        self.assertEqual(pop.num_chromossomes,pop_chromo+pop2_chromo)
        #Products
        self.assertEqual(pop.products_raw.shape,(pop_chromo+pop2_chromo,pop_genes))
        self.assertEqual((pop.products_raw[0:pop_chromo]).all(),(pop_copy.products_raw).all())
        self.assertEqual((pop.products_raw[pop_chromo:]).all(),(pop2.products_raw).all())

        self.assertEqual(pop.masks.shape,(pop_chromo+pop2_chromo,pop_genes))# Masks
        self.assertEqual((pop.masks[0:pop_chromo]).all(),(pop_copy.masks).all())
        self.assertEqual((pop.masks[pop_chromo:]).all(),(pop2.masks).all())

        self.assertEqual(pop.start_raw.shape,(pop_chromo+pop2_chromo,pop_genes))# Start Raw
        self.assertTrue(np.array_equal(pop.start_raw[0:pop_chromo],pop_copy.start_raw))
        self.assertTrue(np.array_equal(pop.start_raw[pop_chromo:],pop2.start_raw))
        

        self.assertEqual(pop.end_raw.shape,(pop_chromo+pop2_chromo,pop_genes))# End Raw
        self.assertTrue(np.array_equal(pop.end_raw[0:pop_chromo],pop_copy.end_raw))
        self.assertTrue(np.array_equal(pop.end_raw[pop_chromo:],pop2.end_raw))

        self.assertEqual(pop.backlogs.shape,(pop_chromo+pop2_chromo,num_metrics))# Stock backlog_i
        self.assertEqual((pop.backlogs[0:pop_chromo]).all(),(pop_copy.backlogs).all())
        self.assertEqual((pop.backlogs[pop_chromo:]).all(),(pop2.backlogs).all())
        
        self.assertEqual(pop.deficit.shape,(pop_chromo+pop2_chromo,num_metrics-1))# Stock Deficit_i
        self.assertEqual((pop.deficit[0:pop_chromo]).all(),(pop_copy.deficit).all())
        self.assertEqual((pop.deficit[pop_chromo:]).all(),(pop2.deficit).all())

        self.assertEqual(pop.objectives_raw.shape,(pop_chromo+pop2_chromo,num_objectives))# Stock Deficit_i
        self.assertEqual((pop.objectives_raw[0:pop_chromo]).all(),(pop_copy.objectives_raw).all())
        self.assertEqual((pop.objectives_raw[pop_chromo:]).all(),(pop2.objectives_raw).all())

        self.assertEqual(pop.genes_per_chromo.shape,(pop_chromo+pop2_chromo,))# Genes per chromossome (Number of active campaigns per solution)
        self.assertEqual((pop.genes_per_chromo[0:pop_chromo]).all(),(pop_copy.genes_per_chromo).all())
        self.assertEqual((pop.genes_per_chromo[pop_chromo:]).all(),(pop2.genes_per_chromo).all())

        self.assertEqual(pop.produced_month_product_individual.shape,(max_months,num_products,pop_chromo+pop2_chromo))# Genes per chromossome (Number of active campaigns per solution)
        self.assertEqual((pop.produced_month_product_individual[0:pop_chromo]).all(),(pop_copy.produced_month_product_individual).all())
        self.assertEqual((pop.produced_month_product_individual[pop_chromo:]).all(),(pop2.produced_month_product_individual).all())
        
        self.assertEqual(pop.fronts.shape,(pop_chromo+pop2_chromo,1))# fronts
        self.assertEqual(pop.crowding_dist.shape,(pop_chromo+pop2_chromo,1))# crowding_dist

    def tearDown(self):
        print("tearDown")


    @classmethod
    def tearDownClass(cls):
        print("teardownClass")


if __name__ == "__main__":
    unittest.main(verbosity=2)

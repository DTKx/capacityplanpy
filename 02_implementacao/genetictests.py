import unittest
import numpy as np
import copy
import pickle
from population import Population
import genetic as gn
import random


def load_obj(path):
    with open(path, "rb") as input:
        obj = pickle.load(input)
    return obj


class GeneticTests(unittest.TestCase):
    path_data = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\03_testing\\"

    @classmethod
    def setUpClass(cls):
        print("setupClass")


    def setUp(self):
        print("setUp")

    def test__fronts(self):
        """Tests _fronts, that classifies solutions(Each row is a solution with different objectives values) into pareto fronts.
        1) Tests using data manually classified to 3 fronts.
        """
        print("_fronts")
        num_fronts = 3
        objectives_fn = np.loadtxt(
            self.path_data + "_fronts.csv", delimiter=",", skiprows=1
        )  # Index(['f0', 'f1', 'f2', 'f3', 'front'], dtype='object')
        front_calc = gn.AlgNsga2._fronts(objectives_fn[:, 0:3], num_fronts)
        self.assertEqual(front_calc.all(), objectives_fn[:, 4].all())

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test__crowding_distance(self):
        """Tests evaluation of crowding distance for a manually calculated crowding distance.
        """
        print("_crowding_distance")
        big_dummy = 10 ** 5
        objectives_fn = np.loadtxt(
            self.path_data + "_crowding_distance.csv", delimiter=",", skiprows=1
        )  # Index(['f0', 'f1', 'f2', 'f3', 'front','dcrowd'], dtype='object')
        crowding_dist = gn.AlgNsga2._crowding_distance(
            objectives_fn[:, 0:4], objectives_fn[:, 4], big_dummy
        )
        delta = 0.0001
        message = "First and second are not almost equal."  # error message in case if test case got failed
        self.assertAlmostEqual(crowding_dist.all(), objectives_fn[:, 5].all(), None, message, delta)
        # npt.assert_almost_equal(crowding_dist, objectives_fn[:, 5], decimal=3)

    def tearDown(self):
        print("tearDown")


    def setUp(self):
        print("setUp")

    def test__index_linear_reinsertion_nsga_constraints(self):
        """Tests the selection of individuals for crossover to verify whether the mean of the number of violations is t1<=t0. If the lowest violations individuals are indeed being selected.
        """
        print("_index_linear_reinsertion_nsga_constraints")
        violations = np.loadtxt(
            self.path_data + "_index_linear_reinsertion_nsga_constraints_violations.txt",
            delimiter=",",
        )
        crowd_dist = np.loadtxt(
            self.path_data + "_index_linear_reinsertion_nsga_constraints_crowding_dist_copy.txt",
            delimiter=",",
        )
        fronts = np.loadtxt(
            self.path_data + "_index_linear_reinsertion_nsga_constraints_fronts_copy.txt",
            delimiter=",",
        )
        n_tests = 10  # Number of tests
        mean_violations = np.mean(violations)
        for i in range(n_tests):
            violations_copy = violations.copy()
            crowd_dist_copy = crowd_dist.copy()
            fronts_copy = fronts.copy()
            ix_sel = gn.AlgNsga2._index_linear_reinsertion_nsga_constraints(
                violations_copy, crowd_dist_copy, fronts_copy, len(fronts_copy) // 2
            )
            self.assertLess(np.mean(violations[ix_sel]), mean_violations)

    def tearDown(self):
        print("tearDown")

    def setUp(self):
        print("setUp")

    def test__crossover_uniform(self):
        """Test Crossover with 3 distinct probabilities (1,0,random)
        1) Probabilities 1 and 0:
            - Tests manual calculated solutions.
            - Tests for invalid inumber of batches 0
        2) For Probability random
            - Tests for invalid inumber of batches 0
        """
        cross_probabilities = [1, 0, random.random()]
        data_pop = ["_crossover_uniform_pop.pkl", "_crossover_uniform_pop2.pkl"]
        for pop_name in data_pop:
            for perc_crossover in cross_probabilities:
                pop = load_obj(self.path_data + pop_name)
                num_ind = len(pop.products_raw)

                n_parents = num_ind // 2

                genes_per_chromo_copy = pop.genes_per_chromo.copy()
                ix_to_crossover = np.random.randint(0, num_ind, size=n_parents)
                ix_to_crossover = ix_to_crossover[
                    np.argsort(genes_per_chromo_copy[ix_to_crossover])
                ]
                products_raw_copy = pop.products_raw[ix_to_crossover].copy()
                batches_raw_copy = pop.batches_raw[ix_to_crossover].copy()
                masks_copy = pop.masks[ix_to_crossover].copy()
                genes_per_chromo_copy = pop.genes_per_chromo[ix_to_crossover].copy()

                new_products, new_batches, new_mask = gn.Crossovers._crossover_uniform(
                    pop.products_raw[ix_to_crossover],
                    pop.batches_raw[ix_to_crossover],
                    pop.masks[ix_to_crossover],
                    perc_crossover,
                )
                if perc_crossover == 1:
                    swap_ix = np.arange(n_parents)  # Creates the solutions index
                    ix_higher_three = np.where((genes_per_chromo_copy >= 3) & (swap_ix % 2 == 0))[
                        0
                    ]  # First index higher than 3 and pair genes_per_chromo[i] >= 3:  # Condition for crossover
                    if len(ix_higher_three) > 0:
                        ix_higher_three = ix_higher_three[0]
                        swap_ix[ix_higher_three] = ix_higher_three + 1
                        swap_ix[ix_higher_three + 1] = ix_higher_three
                        for i in range(ix_higher_three + 2, n_parents):
                            swap_ix[i] = swap_ix[i - 2] + 2

                        delta = 0.0001
                        message = "First and second are not almost equal, for probability:" + str(
                            perc_crossover
                        )  # error message in case if test case got failed
                        self.assertAlmostEqual(
                            new_products.all(),
                            products_raw_copy[swap_ix].all(),
                            None,
                            message,
                            delta,
                        )
                        self.assertAlmostEqual(
                            new_batches.all(), batches_raw_copy[swap_ix].all(), None, message, delta
                        )
                        self.assertAlmostEqual(
                            new_mask.all(), masks_copy[swap_ix].all(), None, message, delta
                        )
                    else:
                        pass
                elif perc_crossover == 0:
                    delta = 0.0001
                    message = "First and second are not almost equal, for probability:" + str(
                        perc_crossover
                    )  # error message in case if test case got failed
                    self.assertAlmostEqual(
                        new_products.all(), products_raw_copy.all(), None, message, delta
                    )
                    self.assertAlmostEqual(
                        new_batches.all(), batches_raw_copy.all(), None, message, delta
                    )
                    self.assertAlmostEqual(new_mask.all(), masks_copy.all(), None, message, delta)
                else:
                    pass
                for i in range(0, n_parents):
                    self.assertFalse(
                        any(new_batches[i][new_mask[i]] == 0)
                    )  # Verify invalid number of batches (0)
                    self.assertFalse(
                        np.sum(new_mask[i][~new_mask[i]]) > 0
                    )  # Verify invalid number of batches (0)

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("teardownClass")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import numpy as np
import pandas as pd
import copy
import pickle
import capacidade as ca
from capacidade import Population
import genetic as gn

# import numpy.testing as npt


def load_obj(path):
    with open(path, "rb") as input:
        obj = pickle.load(input)
    return obj


class TestGenetic(unittest.TestCase):
    path_data = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\03_testing\\"

    @classmethod
    def setUpClass(cls):
        print("setupClass")

    @classmethod
    def tearDownClass(cls):
        print("teardownClass")

    def test__fronts(self):
        """Tests _fronts, that classifies solutions(Each row is a solution with different objectives values) into pareto fronts.
        1) Tests using data manually classified to 3 fronts.
        """
        print("_fronts")
        num_fronts = 3
        objectives_fn = np.loadtxt(
            open(self.path_data + "_fronts.csv", "rb"), delimiter=",", skiprows=1
        )  # Index(['f0', 'f1', 'f2', 'f3', 'front'], dtype='object')
        front_calc = gn.AlgNsga2._fronts(objectives_fn[:, 0:3], num_fronts)
        self.assertEqual(front_calc.all(), objectives_fn[:, 4].all())

    def test__crowding_distance(self):
        """Tests evaluation of crowding distance for a manually calculated crowding distance.
        """
        print("_crowding_distance")
        big_dummy = 10 ** 5
        objectives_fn = np.loadtxt(
            open(self.path_data + "_crowding_distance.csv", "rb"), delimiter=",", skiprows=1
        )  # Index(['f0', 'f1', 'f2', 'f3', 'front','dcrowd'], dtype='object')
        crowding_dist = gn.AlgNsga2._crowding_distance(
            objectives_fn[:, 0:4], objectives_fn[:, 4], big_dummy
        )
        delta = 0.0001
        message = "First and second are not almost equal."  # error message in case if test case got failed
        self.assertAlmostEqual(crowding_dist.all(), objectives_fn[:, 5].all(), None, message, delta)
        # npt.assert_almost_equal(crowding_dist, objectives_fn[:, 5], decimal=3)

    def test__index_linear_reinsertion_nsga_constraints(self):
        """Tests the selection of individuals for crossover to verify whether the mean of the number of violations is t1<=t0. If the lowest violations individuals are indeed being selected.
        """
        print("_index_linear_reinsertion_nsga_constraints")
        violations = np.loadtxt(
            open(
                self.path_data + "_index_linear_reinsertion_nsga_constraints_violations.txt", "rb"
            ),
            delimiter=",",
        )
        crowd_dist = np.loadtxt(
            open(
                self.path_data
                + "_index_linear_reinsertion_nsga_constraints_crowding_dist_copy.txt",
                "rb",
            ),
            delimiter=",",
        )
        fronts = np.loadtxt(
            open(
                self.path_data + "_index_linear_reinsertion_nsga_constraints_fronts_copy.txt", "rb"
            ),
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

    def test_select_pop_by_index(self):
        """Tests if:
        1)selection of pop by index is indeed reducing population violations data in n_tests.
        2)Verify invalid number of batches
        3)Verify invalid value of active genes
        """
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
            ca.Planning().select_pop_by_index(pop, ix_sel)
            self.assertLess(
                np.mean(pop.backlogs[:, 6]), mean_violations
            )  # Verifies if selected individuals indeed reduce violations mean
            for i in range(0,len(pop.products_raw)):
                self.assertFalse(any(pop.batches_raw[i][pop.masks[i]]==0))#Verify invalid number of batches (0)
                self.assertFalse(np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0)#Verify invalid value of active genes (If true Invalid bool after number of active genes.)

if __name__ == "__main__":
    unittest.main()

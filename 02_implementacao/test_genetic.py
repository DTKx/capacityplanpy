import unittest
import numpy as np
import pandas as pd
import genetic as gn


class TestGenetic(unittest.TestCase):
    path_data="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\03_testing\\"
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
        num_fronts=3
        objectives_fn=np.loadtxt(open(self.path_data+"_fronts.csv", "rb"), delimiter=",", skiprows=1)#Index(['f0', 'f1', 'f2', 'f3', 'front'], dtype='object')
        front_calc=gn.AlgNsga2._fronts(objectives_fn[:,0:3], num_fronts)
        self.assertEqual(front_calc.all(), objectives_fn[:,4].all())

if __name__ == "__main__":
    unittest.main()

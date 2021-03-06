import numpy as np
from collections import defaultdict

# import cPickle as pickle
import pickle


class Population:
    num_chromossomes = 10

    def __init__(self, num_genes):
        self.products_raw = np.zeros(shape=(self.num_chromossomes, num_genes), dtype=int)
        self.batches_raw = defaultdict(list)
        self.num_genes = num_genes


class Planning:
    def export_obj(obj, path):
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(path):
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    def main():
        # 1)Trying to open an object created from my current file
        pop = Population(5)  # Initiates object
        print(pop.num_genes)
        path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop.pkl"
        Planning.export_obj(pop, path)
        del pop
        try:
            print(pop.num_genes)
        except:
            pass
        path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop.pkl"
        pop = Planning.load_obj(path)
        print(pop.num_genes)  # Data types works well output an int
        print(pop.batches_raw)  # Data types works well output an dict
        print(pop.products_raw)  # Data types works well output an array
        # 2)Trying to open an object created from another file
        path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_other_file.pkl"
        pop_1 = Planning.load_obj(path)
        print(pop_1.num_genes)  # Expected output an int
        print(pop_1.batches_raw)  # Expected output an array
        print(pop_1.products_raw)  # Expected output an dict

        # 3)Trying to open an object created from my actual implementation
        # path="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_100,2,2,0.5,(0.04, 0.61, 0.77, 0.47).pkl"
        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\01_implementation\\"
        file_path = "15-112\\pop_100,1000,2,0.11,(0.04, 0.61, 0.77, 0.47).pkl"

        pop_actual = Planning.load_obj(root_path + file_path)
        print(pop_actual.name_variation)  # Exception has occurred: AttributeError'list' object has no attribute 'name_variation'
        print(pop_actual.batches_raw)  # Expected output an array
        print(pop_actual.products_raw)  # Expected output an array

        # 4)Trying to open an object created from
        path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_2.pkl"
        pop_2 = Planning.load_obj(path)
        print(pop_2.name_variation)  # Exception has occurred: AttributeError'list' object has no attribute 'name_variation'
        print(pop_2.batches_raw)  # Expected output an array
        print(pop_2.products_raw)  # Expected output an array


if __name__ == "__main__":
    Planning.main()

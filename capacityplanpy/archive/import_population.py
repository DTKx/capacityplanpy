import numpy as np
from collections import defaultdict

# import cPickle as pickle
import pickle
from capacidade import Population
def export_obj(obj, path):
    with open(path, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as input:
        obj = pickle.load(input)
    return obj


def main():
    # 3)Trying to open an object created from my actual implementation
    # path="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_100,2,2,0.5,(0.04, 0.61, 0.77, 0.47).pkl"
    root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\02_analysis\\01_implementation\\"
    file_path = "15-112\\pop_100,1000,2,0.11,(0.04, 0.61, 0.77, 0.47).pkl"

    pop_actual = load_obj(root_path + file_path)
    print("Name ", pop_actual.name_variation)  
    print("Batches ", pop_actual.batches_raw)  
    print("Products ", pop_actual.products_raw)
    print("Shape ", pop_actual.products_raw.shape)
    print("Objectives ", pop_actual.objectives_raw)


if __name__ == "__main__":
    main()

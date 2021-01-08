import numpy as np
from collections import defaultdict
# import cPickle as pickle
import pickle

class Population:
    num_chromossomes=10
    def __init__(self,num_genes):
        self.products_raw=defaultdict(list)
        self.batches_raw=np.zeros(shape=(self.num_chromossomes,num_genes),dtype=int)
        self.num_genes=num_genes

class Planning:
    def export_obj(obj,path):
        with open(path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    def load_obj(path):
        with open(path, 'rb') as input:
            obj = pickle.load(input)
        return obj

    def main():
        pop=Population(5) #Initiates object
        print(pop.num_genes)
        path="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_other_file.pkl"
        Planning.export_obj(pop,path)


if __name__=="__main__":
    Planning.main()

import numpy as np
from collections import defaultdict
# import cPickle as pickle
import pickle
import datetime
class Population():
    """Stores population attributes and methods
    """
    # Metrics per backlog deficit
        # 0)Max total months and products, 1)Mean total months and products, 
        # 2)Std Dev total months and products, 3)Median total months and products,
        # 4)Min total months and products 5)Sum total months and products
        # 6)Backlog violations

    num_metrics=7
    def __init__(self,num_genes,num_chromossomes,num_products,num_objectives,start_date,initial_stock,num_months):
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
            initial_stock (array): Initial Stock of products
            num_months (int): Number of months of planning
        """
        self.name_variation="-"
        self.num_chromossomes=num_chromossomes
        self.num_genes=num_genes

        # Initializes Batch with 1 batch
        self.batches_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype=int)
        self.batches_raw[:,0]=int(1)

        # Initializes products with random allocation of products 
        self.products_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype=int)
        self.products_raw[:,0]=np.random.randint(low=0,high=num_products,size=num_chromossomes)

        # Initialize Mask of active items with only one gene
        self.masks=np.zeros(shape=(num_chromossomes,num_genes),dtype=bool)
        self.masks[:,0]=True

        # Initializes a time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        self.start_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[D]')
        # self.start_raw[:,0]=start_date
        self.end_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[D]')

        # Initializes Stock backlog_i [kg] 

        # 0)Max total backlog months and products, 1)Mean total backlog months and products, 
        # 2)Std Dev total backlog months and products, 3)Median total backlog months and products,
        # 4)Min total backlog months and products 5)Sum total backlog months and products
        # 6)Backlog violations
        self.backlogs=np.zeros(shape=(num_chromossomes,self.num_metrics),dtype=float)

        # Initializes Inventory deficit per month (Objective 1, but with breakdown per month) [kg]
        # 0)Max total months and products, 1)Mean total months and products, 
        # 2)Std Dev total months and products, 3)Median total months and products,
        # 4)Min total months and products 5)Sum total months and products
        self.deficit=np.zeros(shape=(num_chromossomes,self.num_metrics-1),dtype=float)

        # Initializes the objectives throughput_i,deficit_strat_i
        self.objectives_raw=np.zeros(shape=(num_chromossomes,num_objectives),dtype=float)

        # Initializes genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo=np.sum(self.masks,axis=1,dtype=int)

        # Initialize list of dictionaries with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        self.dicts_batches_end_dsp=[]

        # NSGA2
        # Creates an array of fronts and crowding distance
        self.fronts=np.empty(shape=(num_chromossomes,1),dtype=int)
        self.crowding_dist=np.empty(shape=(num_chromossomes,1),dtype=int)


class Planning:
    # Number of genes
    num_genes=int(15)
    # Number of products
    num_products=int(4)
    # Number of Objectives
    num_objectives=2
    # Number of Months
    num_months=36
    # Start date of manufacturing
    start_date=datetime.date(2016,12,1)#  YYYY-MM-DD.
    initial_stock=np.array([18.6,0,19.6,33])

    def export_obj(obj,path):
        with open(path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    def load_obj(path):
        with open(path, 'rb') as input:
            obj = pickle.load(input)
        return obj

    def main(self):
        pop_main=Population(self.num_genes,1,self.num_products,self.num_objectives,self.start_date,self.initial_stock,self.num_months)
        path="C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\pop_2.pkl"
        Planning.export_obj(pop_main,path)


if __name__=="__main__":
    Planning().main()

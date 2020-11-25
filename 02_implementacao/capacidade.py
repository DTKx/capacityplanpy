import numpy as np
import random

"""Pseudo Code
1)Initial Population Chromossome=[Product [int],Num_batches [int]] 
    Random products with a batch of 1
    Idea) 
        My planning horizon is 36 months and my product manufacturing time is at least 35 days, so my worst case scenario regarding chromossome length is when I have changes in product almost every month. So i can use a fixed length chromossome of 37 (36+1 (Extension gene in case of mutation)) along with a boolean mask, therefore I can leverage from either Numba, or tensor libraries more easily.
        I can use 2 arrays one for the number of batches and one for the product label

"""

# class Current():
#     """Stores current population
#     """

class CurrentPop():
    """Stores current population and its methods
    """
    def __init__(self,num_genes,num_chromossomes,num_products):
        """Initiates the current population, with a batch population,product population and a mask.
        batch population contains the number of batches, initially with only one batch
        product population contains the product being produced related to the batch number of the batch population,r randolmly assigned across different number of products
        mask dictates the current population in place, supporting a variable length structure

        Args:
            num_genes (int): Number of genes in a chromossome
            num_chromossomes (int): Number of chromossomes in population
            num_products (int): Number of products available to compose the product propulation
        """
        self.batches_raw=np.zeros(shape=(num_chromossomes,num_genes))
        # Initializes with 1 batch
        self.batches_raw[:,0]=int(1)
        self.products_raw=np.zeros(shape=(num_chromossomes,num_genes))
        # Initializes with random allocation of products
        self.batches_raw[:,0]=np.random.randint(low=0,high=num_products,size=num_chromossomes)
        self.masks=np.zeros(shape=(num_chromossomes,num_genes),dtype=bool)
        # Initializes with only one gene
        self.masks[:,0]=True
        # The real population must be returned with the mask
        self.batches=self.batches_raw[self.masks]
        self.products=self.products_raw[self.masks]

    def agg_product_batch(self):
        """Aggregates product batches in case of neighbours products
        """
        for 



class Planning():
    # Class Variables

    # Number of genes
    num_genes=int(37)
    # # Number of chromossomes
    # num_chromossomes=int(100)
    # Number of products
    num_products=int(3)
    def calc_objectives(pop_batches,pop_products):

    def main(num_chromossomes,num_geracoes,n_tour,perc_crossover):
        # 1) Random parent population is created
        pop=CurrentPop(Planning.num_genes,num_chromossomes,Planning.num_products)
        print(pop.batches.shape)
        print(pop.products.shape)
        print(pop.masks.shape)
        # 2) Avaliar objetivos
        pop_batch,pop_product=
    
    def run_cprofile():
        num_chromossomes=100
        num_geracoes=200
        n_tour=2
        perc_crossover=0.6
        Planning.main(num_chromossomes,num_geracoes,n_tour,perc_crossover)



if __name__=="__main__":
    # Planning.main()
    Planning.run_cprofile()

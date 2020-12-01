import numpy as np
import random
import copy
import timeit
import datetime
from dateutil.relativedelta import *

"""Pseudo Code
1)Initial Population Chromossome=[Product [int],Num_batches [int]] 
    Random products with a batch of 1
    Idea) 
        My planning horizon is 36 months and my product manufacturing time is at least 35 days, so my worst case scenario regarding chromossome length is when I have changes in product almost every month. So i can use a fixed length chromossome of 37 (36+1 (Extension gene in case of mutation)) along with a boolean mask, therefore I can leverage from either Numba, or tensor libraries more easily.
        I can use 2 arrays one for the number of batches and one for the product label

"""

class CurrentPop():
    """Stores current population and its atributes and methods
    """
    def __init__(self,num_genes,num_chromossomes,num_products,num_objectives,start_date,initial_stock):
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
        """
        # Initializes Batch with 1 batch
        self.batches_raw=np.zeros(shape=(num_chromossomes,num_genes))
        self.batches_raw[:,0]=int(1)

        # Initializes products with random allocation of products 
        self.products_raw=np.zeros(shape=(num_chromossomes,num_genes))
        self.products_raw[:,0]=np.random.randint(low=0,high=num_products,size=num_chromossomes)

        # Initializes a time vector Start and end of campaign Starting with the first date
        self.start_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[s]')
        self.start_raw[:,0]=start_date
        self.end_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[s]')

        # Initializes Stock
        self.stock_raw=np.zeros(shape=(num_chromossomes,num_products))

        # Initialize Mask of active items with only one gene
        self.masks=np.zeros(shape=(num_chromossomes,num_genes),dtype=bool)
        self.masks[:,0]=True

        # Initializes the objectives
        self.objectives_raw=np.zeros(shape=(num_chromossomes,num_objectives))

        # Initializes genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo=np.sum(self.masks,axis=0)

        # # The real population must be returned with the mask
        # self.batches=self.batches_raw[self.masks]
        # self.products=self.products_raw[self.masks]
        # # self.timeline=self.timeline_raw[self.masks]

    def count_genes_per_chromossomes(self):
        """Counts number of active genes per chromossome

        Returns:
            array: Count of active genes per chromossome
        """
        count=np.sum(self.masks,axis=0)
        return count

    def agg_product_batch(self):
        """Aggregates product batches in case of neighbours products
        """
        # Loop per chromossome in population
        # for list_batches,list_products,list_masks,list_genes_chromo in zip(self.batches,self.products,self.masks,self.genes_per_chromo):
        for j in range(0,len(self.genes_per_chromo)):
            # Loop per gene i in chromossome
            i=0
            while i<np.sum(self.masks[j],axis=0)-1:
                if self.products[j][i]==self.products[j][i+1]:
                    # Sum next
                    print(self.batches[j])
                    self.batches[j][i]=self.batches[j][i]+self.batches[j][i+1]
                    print(self.batches[j])
                    # Deletes [i+a] and insert a value in the last
                    print(self.batches[j])
                    self.batches[j]=np.insert(np.delete(self.batches[j],i),-1,0)
                    print(self.batches[j])
                    print(self.products[j])
                    self.products[j]=np.insert(np.delete(self.products[j],i),-1,0)
                    print(self.products[j])
                    print(self.masks[j])
                    self.masks[j]=np.insert(np.delete(self.masks[j],i),-1,False)
                    print(self.masks[j])
                    self.start_raw[j]=np.insert(np.delete(self.start_raw[j],i),-1,0)
                    self.end_raw[j]=np.insert(np.delete(self.end_raw[j],i),-1,0)
                    self.stock_raw[j]=np.insert(np.delete(self.stock_raw[j],i),-1,0)
                    i+=1
                else:
                    i+=1
        # Updates
        self.genes_per_chromo=np.sum(self.masks,axis=0)


class Planning():
    # Class Variables

    # Number of genes
    num_genes=int(37)
    # # Number of chromossomes
    # num_chromossomes=int(100)
    # Number of products
    num_products=int(3)
    # Number of Objectives
    num_objectives=2
    # Number of Months
    num_months=36
    start_date=datetime.date(2016,12,1)#  YYYY-MM-DD.
    # use_date = x+relativedelta(months=+1)

    # Process Data 
    products = [0,1,2,3]
    usp_days=dict(zip(products,[45,36,45,49]))
    dsp_days=dict(zip(products,[7,11,7,7]))
    qc_days=dict(zip(products,[90,90,90,90]))
    yield_kg_batch=dict(zip(products,[3.1,6.2,4.9,5.5]))
    initial_stock=dict(zip(products,[18.6,0,19.6,33]))
    min_batch=dict(zip(products,[2,2,2,3]))
    max_batch=dict(zip(products,[50,50,50,30]))
    batch_multiples=dict(zip(products,[1,1,1,3]))

    # Target Stock
    target_0=[6.2,6.2,9.3,9.3,12.4,12.4,15.5,21.7,21.7,24.8,21.7,24.8,27.9,21.7,24.8,24.8,24.8,27.9,27.9,27.9,31,31,34.1,34.1,27.9,27.9,27.9,27.9,34.1,34.1,31,31,21.7,15.5,6.2,0]
    target_1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2]
    target_2=[0,4.9,9.8,9.8,9.8,9.8,19.6,19.6,14.7,19.6,19.6,19.6,14.7,19.6,19.6,14.7,14.7,19.6,19.6,9.8,19.6,19.6,19.6,19.6,24.5,34.3,24.5,29.4,39.2,39.2,29.4,19.6,19.6,14.7,4.9,0]
    target_3=[22,27.5,27.5,27.5,27.5,33,33,27.5,27.5,27.5,38.5,33,33,33,33,33,27.5,33,33,33,38.5,33,38.5,33,33,33,33,44,33,33,33,33,22,11,11,5.5]
    target_stock=[{0: a,1: b,2: c,3: d} for a,b,c,d in zip(target_0,target_1,target_2,target_3)]

    # Setup Time
    s0=[0,10,16,20]
    s1=[16,0,16,20]
    s2=[16,10,0,20]
    s3=[18,10,18,0]
    setup_key_to_subkey=[{0: a,1: b,2: c,3: d} for a,b,c,d in zip(target_0,target_1,target_2,target_3)]

    def calc_throughput(pop_objectives,pop_products):
        # Creates a vector for batch/kg por the products 
        pop_yield=np.vectorize(Planning.yield_kg_batch.__getitem__)(pop_products) #Equivalentt to         # pop_yield=np.array(list(map(Planning.yield_kg_batch.__getitem__,pop_products)))
        # unique,inv=np.unique(pop_products,return_inverse=True)
        # pop_yield=np.array([Planning.yield_kg_batch[x] for x in unique])[inv].reshape(pop_products.shape)

        # Loop per chromossome
        # print("Objectives",pop_objectives)
        for i in range(0,len(pop_batches)):
            pop_objectives[i,0]=np.dot(pop_batches[i],pop_yield[i])
        # print("Objectives",pop_objectives)
        return pop_objectives

pop.batches_raw,pop.products_raw,pop.objectives_raw,pop.masks
    def calc_objectives(pop_batches,pop_products,pop_objectives):
        """Overall, the goal is to generate a set of schedules: 
                maximise the total production throughput (Column 0)
                minimise the median total inventory deficit subject to the median total backlog being no greater than 0 kg (Column 1)
        Args:
            pop_batches (array): Population of number of batches
            pop_products (array): Population of products
        Returns:
            [type]: [description]
        """
        # Calculating the throughput
        pop_objectives=Planning.calc_throughput(pop_objectives,pop_products)
        return pop_objectives

    def main(num_chromossomes,num_geracoes,n_tour,perc_crossover):
        # 1) Random parent population is initialized with its attributes
        pop=CurrentPop(Planning.num_genes,num_chromossomes,Planning.num_products,Planning.num_objectives,Planning.start_date,Planning.initial_stock)
        # print(pop.batches.shape)
        # print(pop.products.shape)
        # print(pop.masks.shape)

        # 1.1) Fills Start and End Date
        
        
        # # 2) Aggregates neighbours products no need in initialization
        # print(np.sum(pop.masks))
        # pop.agg_product_batch()
        # print(np.sum(pop.masks))

        # 2) Avaliar objetivos
        pop_objectives=Planning.calc_objectives(pop.batches_raw,pop.products_raw,pop.objectives_raw,pop.masks)
        print("hey")
    
    def run_cprofile():
        num_chromossomes=100
        num_geracoes=200
        n_tour=2
        perc_crossover=0.6
        Planning.main(num_chromossomes,num_geracoes,n_tour,perc_crossover)



if __name__=="__main__":
    # Planning.main()
    Planning.run_cprofile()

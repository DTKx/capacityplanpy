import numpy as np
import random
import copy
import timeit
import datetime
from dateutil.relativedelta import *
import collections
import pandas as pd
from dateutil import relativedelta

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
        self.batches_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype=int)
        self.batches_raw[:,0]=int(1)

        # Initializes products with random allocation of products 
        self.products_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype=int)
        self.products_raw[:,0]=np.random.randint(low=0,high=num_products,size=num_chromossomes)

        # Initializes a time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        self.start_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[D]')
        # self.start_raw[:,0]=start_date
        self.end_raw=np.zeros(shape=(num_chromossomes,num_genes),dtype='datetime64[D]')

        # Initialize Mask of active items with only one gene
        self.masks=np.zeros(shape=(num_chromossomes,num_genes),dtype=bool)
        self.masks[:,0]=True

        # Initializes Stock
        # self.stock_raw=np.zeros(shape=(num_chromossomes,num_products),dtype=int)
        self.stock_raw=collections.defaultdict(partial)

        # Initializes the objectives
        self.objectives_raw=np.zeros(shape=(num_chromossomes,num_objectives),dtype=float)

        # Initializes genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo=np.sum(self.masks,axis=1,dtype=int)

        # Initialize list of dictionaries with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        self.dicts_batches_end_dsp=[]

        # Initialize list of dictionaries available batches with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        self.dicts_batches_end_dsp=[]

        # # The real population must be returned with the mask
        # self.batches=self.batches_raw[self.masks]
        # self.products=self.products_raw[self.masks]
        # # self.timeline=self.timeline_raw[self.masks]
    def update_genes_per_chromo(self):
        # Updates genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo=np.sum(self.masks,axis=1,dtype=int)

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
    num_products=int(4)
    # Number of Objectives
    num_objectives=2
    # Number of Months
    num_months=36
    # Start date of manufacturing
    start_date=datetime.date(2016,12,1)#  YYYY-MM-DD.
    # Last day of manufacturing
    end_date=datetime.date(2019,12,1)#  YYYY-MM-DD.
    # List of months
    list_months=pd.date_range(start=start_date, end =end_date, freq='MS')[1:]
    date_stock=list_months[0]

    # Process Data 
    products = [0,1,2,3]
    usp_days=dict(zip(products,[45,36,45,49]))
    dsp_days=dict(zip(products,[7,11,7,7]))
    qc_days=dict(zip(products,[90,90,90,90]))
    yield_kg_batch=dict(zip(products,[3.1,6.2,4.9,5.5]))
    # initial_stock=dict(zip(products,[18.6,0,19.6,33]))
    initial_stock=np.array([18.6,0,19.6,33])
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
    setup_key_to_subkey=[{0: a,1: b,2: c,3: d} for a,b,c,d in zip(s0,s1,s2,s3)]

    def calc_start_end(self,pop):
        """Calculates start and end dates of batch manufacturing, as well as generates (dicts_batches_end_dsp) a list of dictionaries (List index = Chromossome, key=Number of products and date values os release from QC) per chromossome with release dates of each batch per product. 

        Args:
            pop (Class object): Class Object of the population to be analized
        """
        # Extracts the population informations
        dsp_raw=np.vectorize(self.dsp_days.__getitem__)(pop.products_raw)
        usp_plus_dsp_raw=np.vectorize(self.usp_days.__getitem__)(pop.products_raw)+copy.deepcopy(dsp_raw)

        # Initialize by addying the first date
        pop.start_raw[:,0]=self.start_date

        # Loop per chromossome i
        for i in range(0,len(pop.start_raw)):
            batches_end_date_i=collections.defaultdict(list)
            # pop.batches_end_date_dsp=[]
            # Loop per gene j starting from second gene
            for j in range(0,pop.genes_per_chromo[i]):
                if j==0:
                    # List of batches end date End date=start+(USP+DSP)*1+DSP*num_batches
                    end_dates=[pop.start_raw[i][j]+np.timedelta64(usp_plus_dsp_raw[i][j],'D')+np.timedelta64(dsp_raw[i][j]*k,'D') for k in range(0,pop.batches_raw[i][j])]
                    # Verifies if End day<Last Day ok else delete
                    num_batch_exceed_end_dates=np.sum(np.array(end_dates)>self.end_date)
                    if num_batch_exceed_end_dates>0:
                        # Removes exceeding batches
                        pop.batches_raw[i][j]=pop.batches_raw[i][j]-num_batch_exceed_end_dates
                        del end_dates[-num_batch_exceed_end_dates:]
                        if pop.batches_raw[i][j]==0:
                            pop.masks[i][j]=False
                            continue
                    # Add first campaign end date0=start0+(USP+DSP)*1+DSP*num_batches
                    pop.end_raw[i][j]=end_dates[-1]
                    # Addying the quality control time
                    end_dates=end_dates+np.timedelta64(self.qc_days[pop.products_raw[i][j]],'D')
                    # Appends to the dictionary 
                    for date in end_dates:
                        batches_end_date_i[pop.products_raw[i][j]].append(date)
                else:
                    # Add a Start Date=Previous End Date+Change Over Time
                    pop.start_raw[i,j]=pop.end_raw[i,j-1]+np.timedelta64(self.setup_key_to_subkey[pop.products_raw[i,j]][pop.products_raw[i,j-1]],'D')

                    # List of batches end date End date=start+(USP+DSP)*1+DSP*num_batches
                    end_dates=[pop.start_raw[i][j]+np.timedelta64(usp_plus_dsp_raw[i][j],'D')+np.timedelta64(dsp_raw[i][j]*k,'D') for k in range(0,pop.batches_raw[i][j])]

                    # Verifies if End day<Last Day ok else delete
                    num_batch_exceed_end_dates=np.sum(np.array(end_dates)>self.end_date)
                    if num_batch_exceed_end_dates>0:
                        # Removes exceeding batches
                        pop.batches_raw[i][j]=pop.batches_raw[i][j]-num_batch_exceed_end_dates
                        del end_dates[-num_batch_exceed_end_dates:]
                        if pop.batches_raw[i][j]==0:
                            pop.masks[i][j]=False
                            continue

                    # Add first campaign end date0=start0+(USP+DSP)*1+DSP*num_batches
                    pop.end_raw[i][j]=end_dates[-1]
                    # Addying the quality control time
                    end_dates=end_dates+np.timedelta64(self.qc_days[pop.products_raw[i][j]],'D')
                    # Appends to the dictionary
                    for date in end_dates:
                        batches_end_date_i[pop.products_raw[i][j]].append(date)           
            # Appends dictionary of individual to the list of dictionaries
            pop.dicts_batches_end_dsp.append(batches_end_date_i)
        # # Loops per solution
        # for i in range(0,len(pop.dicts_batches_end_dsp)):
        #         a_i=pd.DataFrame.from_dict(pop.dicts_batches_end_dsp[i])
        #         b_i=pd.DataFrame(pop.dicts_batches_end_dsp[i],index=self.list_months)
        #         c_i=pd.Series(pd.to_datetime(pop.dicts_batches_end_dsp[i]),index=self.list_months).resample("M", convention='start').sum()

        #         c_i=pd.Series(pd.to_datetime(pop.dicts_batches_end_dsp[i]),index=self.list_months).resample("M", convention='start')

        #         c_i=pd.Series(pop.dicts_batches_end_dsp[i][2],index=self.list_months)

        #         print("h")
        
        # # Loops per solution
        # for i in range(0,len(pop.dicts_batches_end_dsp)):
        #     # Loops per product
        #     for key in pop.dicts_batches_end_dsp[i].keys():
        #         print(pop.dicts_batches_end_dsp[i][key])
        #         # Aggregated count per month 
        #         pop.dicts_batches_end_dsp[i][key]=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='start').sum()
        #         print(pop.dicts_batches_end_dsp[i][key])

        # a=pd.Series(1,index=pd.to_datetime(batches_end_date_i[1])).resample("M", convention='start').sum()
        # print('hey')
        # type(pop.dicts_batches_end_dsp[0])
        # len(pop.dicts_batches_end_dsp[0].keys())
        # pop.dicts_batches_end_dsp[0][0]

        # Updates Genes per Chromo
        pop.update_genes_per_chromo()
    def calc_inventory(self,pop):
        """Calculates Inventory levels (e), Backlog (back), batches Sold(s), Available (a), Demand(d))

        Args:
            pop (class object): Population object to calculate Inventory levels
        """
        # print(pop.dicts_batches_end_dsp)
        # print("h")
        # Loop per Chromossome
        for i in range(0,len(pop.products_raw)):
            available_i=np.zeros(shape=(self.num_months,self.num_products))
            stock_i=np.zeros(shape=(self.num_months,self.num_products))

            # Produced Month 0 is the first month of inventory
            produced_i=np.zeros(shape=(self.num_months,self.num_products),dtype=int)
            for key in pop.dicts_batches_end_dsp[i].keys():
                # Aggregated count per month 
                aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='start').sum()
                for k in range(0,len(aggregated)):
                    m = relativedelta.relativedelta(aggregated.index[k],self.date_stock).months
                    # Updates the month with the number of batches produced
                    produced_i[m,key]=aggregated[k]

            # Loop per Months
            for j in range(0,self.num_months):
                if j==0:
                    # Available=Previous Stock+Produced this month
                    available_i[0,:]=self.initial_stock+produced_i[0,:]
                else:
                    # Available=Previous Stock+Produced this month
                    available_i[j,:]=self.initial_stock+produced_i[j,:]

            # self.stock_raw=collections.defaultdict(partial)
                    print("hey")
            # batches_end_date_i=collections.defaultdict(list)



    def calc_throughput(self,pop_objectives,pop_products):
        # Creates a vector for batch/kg por the products 
        pop_yield=np.vectorize(self.yield_kg_batch.__getitem__)(pop_products) #Equivalentt to         # pop_yield=np.array(list(map(self.yield_kg_batch.__getitem__,pop_products)))
        # unique,inv=np.unique(pop_products,return_inverse=True)
        # pop_yield=np.array([self.yield_kg_batch[x] for x in unique])[inv].reshape(pop_products.shape)

        # Loop per chromossome
        # print("Objectives",pop_objectives)
        for i in range(0,len(pop_batches)):
            pop_objectives[i,0]=np.dot(pop_batches[i],pop_yield[i])
        # print("Objectives",pop_objectives)
        return pop_objectives

    def calc_objectives(self,pop_batches,pop_products,pop_objectives):
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
        pop_objectives=self.calc_throughput(pop_objectives,pop_products)
        return pop_objectives

    def main(self,num_chromossomes,num_geracoes,n_tour,perc_crossover):
        # 1) Random parent population is initialized with its attributes
        pop=CurrentPop(self.num_genes,num_chromossomes,self.num_products,self.num_objectives,self.start_date,self.initial_stock)

        # 1.1) Creates start and end date from schedule assures only batches with End date<Last day of manufacturing
        # 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        self.calc_start_end(pop)       

        # 3)Calculate inventory levels
        self.calc_inventory(pop)

        # print(np.sum(pop.masks))
        # pop.agg_product_batch()
        # print(np.sum(pop.masks))
        # 2) Avaliar objetivos
        pop_objectives=self.calc_objectives(pop.batches_raw,pop.products_raw,pop.objectives_raw,pop.masks)
        print("hey")
    
    def run_cprofile():
        num_chromossomes=100
        num_geracoes=200
        n_tour=2
        perc_crossover=0.6
        Planning().main(num_chromossomes,num_geracoes,n_tour,perc_crossover)



if __name__=="__main__":
    # Planning.main()
    Planning.run_cprofile()

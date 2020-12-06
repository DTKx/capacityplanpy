import numpy as np
import random
import copy
import timeit
import datetime
from dateutil.relativedelta import *
import pandas as pd
from dateutil import relativedelta
from numba import jit
from pygmo import *
from collections import defaultdict

# Local Modules
# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1,'C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\')
# import genetico_permutacao as genetico
from genetic import AlgNsga2,Crossovers,Mutations
# AlgNsga2._crossover_uniform,AlgNsga2._fronts,_crowding_distance


class Population():
    """Stores population attributes and methods
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

        # Initializes Stock backlog_i
        self.backlogs=np.zeros(shape=(num_chromossomes,1),dtype=int)

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

    def update_genes_per_chromo(self):
        """ Updates genes per chromossome (Number of active campaigns per solution)
        """
        self.genes_per_chromo=np.sum(self.masks,axis=1,dtype=int)

    def create_new_population(self,new_products,new_batches,new_mask):
        """Updates the values of the new offspring population in the class object.

        Args:
            new_products (Array of ints): Population of product labels
            new_batches (Array of ints): Population of number of batches
            new_mask (Array of booleans): Population of active genes
        """
        # Updates new Batches values
        self.batches_raw=copy.deepcopy(new_batches)

        # Updates new Products
        self.products_raw=copy.deepcopy(new_products)

        # Updates Mask of active items with only one gene
        self.masks=copy.deepcopy(new_mask)



class Planning():
    # Class Variables

    # General Genetic Algorithms parameters

    # Number of genes
    num_genes=int(37)

    # Mutation
    pmutp=0.5
    pposb=0.5
    pnegb=0.5
    pswap=0.5

    # Problem variables

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
    # First day of stock calculation
    date_stock=list_months[0]

    # Number of Monte Carlo executions Article ==1000
    num_monte=500

    # Process Data 
    products = [0,1,2,3]
    usp_days=dict(zip(products,[45,36,45,49]))
    dsp_days=dict(zip(products,[7,11,7,7]))
    qc_days=dict(zip(products,[90,90,90,90]))
    yield_kg_batch=dict(zip(products,[3.1,6.2,4.9,5.5]))
    yield_kg_batch_ar=np.array([3.1,6.2,4.9,5.5])
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
    # target_stock=[{0: a,1: b,2: c,3: d} for a,b,c,d in zip(target_0,target_1,target_2,target_3)]
    target_stock=np.column_stack([target_0,target_1,target_2,target_3])

    # Setup Time
    s0=[0,10,16,20]
    s1=[16,0,16,20]
    s2=[16,10,0,20]
    s3=[18,10,18,0]
    setup_key_to_subkey=[{0: a,1: b,2: c,3: d} for a,b,c,d in zip(s0,s1,s2,s3)]

    # Inversion val to convert maximization of throughput to minimization, using a value a little bit higher than the article max 630.4
    inversion_val_throughput=650

    # NSGA Variables

    # Number of fronts created
    num_fronts=3

    # Big Dummy for crowding distance computation
    big_dummy=10**5

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
            batches_end_date_i=defaultdict(list)
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
        # Updates Genes per Chromo
        pop.update_genes_per_chromo()

    @staticmethod
    @jit(nopython=True,nogil=True)
    def calc_triangular_dist(tr_min,tr_mode,tr_max,num_monte):
        return np.median(np.random.triangular(tr_min,tr_mode,tr_max,size=num_monte))

    @staticmethod
    # @jit(nopython=True,nogil=True)
    def calc_demand_montecarlo(num_monte,line,col):
        """Performs a Montecarlo Simulation to define the Demand of products, uses a demand_distribution for containing either 0 as expected or a triangular distribution (minimum, mode (most likely),maximum) values in kg

        Args:
            num_monte ([type]): Number of Monte Carlo Executions
        """
        demand_distribution=np.array([[0.0 ,0.0 ,0.0 ,0.0],
        [0.0,0.0,0.0 ,(4.5, 5.5, 8.25) ],
        [(2.1, 3.1, 4.65) ,0.0,0.0 ,(4.5, 5.5, 8.25)],
        [0.0,0.0 ,0.0 ,0.0],
        [0.0,0.0 ,0.0 ,(4.5, 5.5, 8.25) ],
        [(2.1, 3.1, 4.65) ,0.0 ,0.0,(4.5, 5.5, 8.25) ],
        [0.0,0.0 ,(3.9, 4.9, 7.35) ,(4.5, 5.5, 8.25) ],
        [(2.1, 3.1, 4.65) ,0.0 ,(3.9, 4.9, 7.35),(4.5, 5.5, 8.25) ],
        [(2.1, 3.1, 4.65) ,0.0 ,0.0 ,(4.5, 5.5, 8.25)],
        [(2.1, 3.1, 4.65),0.0 ,0.0 ,0.0],
        [0.0,0.0 ,0.0,(10, 11, 16.5) ],
        [(5.2, 6.2, 9.3) ,0.0 ,(8.8, 9.8, 14.7) ,(4.5, 5.5, 8.25)],
        [(5.2, 6.2, 9.3) ,0.0 ,(3.9, 4.9, 7.35),0.0],
        [(2.1, 3.1, 4.65) ,0.0 ,0.0,(4.5, 5.5, 8.25) ],
        [(5.2, 6.2, 9.3),0.0 ,(3.9, 4.9, 7.35) ,(4.5, 5.5, 8.25) ],
        [0.0,0.0 ,0.0,(10, 11, 16.5) ],
        [(2.1, 3.1, 4.65) ,0.0 ,0.0,(4.5, 5.5, 8.25) ],
        [(8.3, 9.3, 13.95),0.0 ,(3.9, 4.9, 7.35) ,(4.5, 5.5, 8.25)],
        [0.0,0.0 ,(8.8, 9.8, 14.7),0.0],
        [(5.2, 6.2, 9.3) ,0.0 ,0.0 ,(4.5, 5.5, 8.25) ],
        [(5.2, 6.2, 9.3),0.0 ,0.0 ,(4.5, 5.5, 8.25) ],
        [0.0,0.0,0.0,(4.5, 5.5, 8.25) ],
        [(5.2, 6.2, 9.3) ,(5.2, 6.2, 9.3) ,(3.9, 4.9, 7.35) ,(10, 11, 16.5) ],
        [(8.3, 9.3, 13.95),0.0,(3.9, 4.9, 7.35),(4.5, 5.5, 8.25)],
        [0.0,0.0 ,0.0,0.0],
        [(8.3, 9.3, 13.95) ,0.0 ,(8.8, 9.8, 14.7) ,(10, 11, 16.5) ],
        [(5.2, 6.2, 9.3) ,0.0 ,0.0,0.0],
        [(2.1, 3.1, 4.65) ,0.0,0.0,(10, 11, 16.5) ],
        [(5.2, 6.2, 9.3) ,(5.2, 6.2, 9.3) ,(3.9, 4.9, 7.35) ,(4.5, 5.5, 8.25) ],
        [(2.1, 3.1, 4.65),0.0,(8.8, 9.8, 14.7) ,(4.5, 5.5, 8.25)],
        [0.0,0.0 ,(8.8, 9.8, 14.7),0.0],
        [(8.3, 9.3, 13.95) ,0.0 ,0.0,(10, 11, 16.5) ],
        [(5.2, 6.2, 9.3) ,0.0 ,(3.9, 4.9, 7.35) ,(10, 11, 16.5)],
        [(8.3, 9.3, 13.95) ,0.0 ,(8.8, 9.8, 14.7) ,0.0],
        [(5.2, 6.2, 9.3),0.0,(3.9, 4.9, 7.35),(4.5, 5.5, 8.25) ],
        [0.0,(5.2, 6.2, 9.3) ,0.0,(4.5, 5.5, 8.25)]])

        # line,col=demand_distribution.shape
        demand_i=np.zeros(shape=(line,col))
        # Loop in line
        for i in range(0,line):
            for j in range(0,col):
                if demand_distribution[i,j]==0:
                    continue
                else:
                    demand_i[i,j]=Planning.calc_triangular_dist(demand_distribution[i,j][0],demand_distribution[i,j][1],demand_distribution[i,j][2],num_monte)
        return demand_i

    @staticmethod
    # @jit(nopython=True,nogil=True)
    def calc_objective_deficit_strat(target_stock_i,stock_i):
        deficit_strat_i=target_stock_i-stock_i
        # Corrects negative values
        # np.float64
        ix_neg=np.where(deficit_strat_i<np.float64(0.0))
        # print(type(len(ix_neg)))
        if len(ix_neg)>int(0):
            deficit_strat_i[ix_neg]=np.float64(0.0)
        # Stores sum of all deficit to the objectives
        return np.sum(deficit_strat_i)

    def calc_inventory_objectives(self,pop):
        """Calculates Inventory levels returning the backlog and calculates the objectives the total deficit and total throughput addying to the pop attribute

        Args:
            pop (class object): Population object to calculate Inventory levels
        """
        # print(pop.dicts_batches_end_dsp)
        # print("h")

        # Creates a vector for batch/kg por the products 
        pop_yield=np.vectorize(self.yield_kg_batch.__getitem__)(pop.products_raw) #Equivalentt to         # pop_yield=np.array(list(map(self.yield_kg_batch.__getitem__,pop_products)))

        # Loop per Chromossome
        for i in range(0,len(pop.products_raw)):
            available_i=np.zeros(shape=(self.num_months,self.num_products))
            stock_i=np.zeros(shape=(self.num_months,self.num_products))
            backlog_i=np.zeros(shape=(self.num_months,self.num_products))

            # Produced Month 0 is the first month of inventory batches
            produced_i=np.zeros(shape=(self.num_months,self.num_products),dtype=int)
            for key in pop.dicts_batches_end_dsp[i].keys():
                # Aggregated count per month 
                aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='start').sum()
                for k in range(0,len(aggregated)):
                    m = relativedelta.relativedelta(aggregated.index[k],self.date_stock).months
                    # Updates the month with the number of batches produced
                    produced_i[m,key]=aggregated[k]
            # Conversion batches to kg
            produced_i=produced_i*self.yield_kg_batch_ar

            # Calling Monte Carlo to define demand
            demand_i=self.calc_demand_montecarlo(self.num_monte,self.num_months,self.num_products)

            # Loop per Months
            for j in range(0,self.num_months):
                if j==0:
                    # Available=Previous Stock+Produced this month
                    available_i[j,:]=self.initial_stock+produced_i[j,:]
                    # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
                    stock_i[j,:]=available_i[j,:]-demand_i[j,:]
                    # Corrects negative values
                    ix_neg=np.where(stock_i[j,:]<0)
                    if len(ix_neg)>0:
                        # Adds negative values to backlog
                        backlog_i[j,:][ix_neg]=(stock_i[j,:][ix_neg])*(-1)
                        # print(f"backlog {backlog_i[j,:][ix_neg]}")
                        # Corrects if Stock is negative
                        stock_i[j,:][ix_neg]=0
                        # print(f"backlog {backlog_i[j,:][ix_neg]} check if mutated after assignement of stock")
                else:
                    # Available=Previous Stock+Produced this month
                    available_i[j,:]=stock_i[j-1,:]+produced_i[j,:]

                    # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
                    stock_i[j,:]=available_i[j,:]-demand_i[j,:]
                    # Corrects negative values
                    ix_neg=np.where(stock_i[j,:]<0)
                    if len(ix_neg)>0:
                        # Adds negative values to backlog
                        backlog_i[j,:][ix_neg]=(stock_i[j,:][ix_neg])*(-1)
                        # print(f"backlog {backlog_i[j,:][ix_neg]}")
                        # Corrects if Stock is negative
                        stock_i[j,:][ix_neg]=0
                        # print(f"backlog {backlog_i[j,:][ix_neg]} check if mutated after assignement of stock")
            # Stores sum of all backlogs to the stock
            pop.backlogs[i,0]=np.sum(backlog_i)

            # Calculates the objective Strategic Deficit 
            pop.objectives_raw[i,1]=self.calc_objective_deficit_strat(self.target_stock,stock_i)

            # Calculates the objective Throughput
            pop.objectives_raw[i,0]=np.dot(pop.batches_raw[i][pop.masks[i]],pop_yield[i][pop.masks[i]])
            # Inversion of the Throughput by a fixed value to generate a minimization problem
            pop.objectives_raw[i,0]=self.inversion_val_throughput-pop.objectives_raw[i,0]
        # Check if inversion value is well selected
        if any(pop.objectives_raw[:,0]<0):
            raise Exception('Inversion Value is too low, generating negative values. Consider:',np.min(pop.objectives_raw[:,0]))


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

    @staticmethod
    def tournament_restrictions_binary(pop,n_parents,n_tour):
        """Tournament with replacement for selection to crossover, considering those criteria:
        1)Lowest backlog value, if draw then: 
        2)Best pareto front, if draw then:
        3)Highest Crowding Distance

        Args:
            pop (object): Population object that contains the individuals to select
            n_parents (int): Number of chromossomes to select
            n_tour (int): Number of individuals to compete during each tournament

        Returns:
            array: Array with indexes of selected individuals
        """
        # Arrays representing the indexes
        idx_population=np.arange(0,pop.num_chromossomes)    
        # Indexes of winners
        idx_winners=np.empty(shape=(n_parents,),dtype=int)

        # Selection all participants
        idx_for_tournament = np.random.choice(idx_population,size=n_tour*n_parents,replace=True)
        j=0
        for i in range(0,n_tour*n_parents-1,2):
            i_1,i_2=idx_for_tournament[i],idx_for_tournament[i+1]
            # Criteria
            # 1) Lowest backlog
            if pop.backlogs[i_1]!=pop.backlogs[i_2]:
                # To change for different number of tours c=np.where(a==np.min(a))
                if pop.backlogs[i_1]-pop.backlogs[i_2]>0:
                    idx_winners[j]=i_1
                else:
                    idx_winners[j]=i_2
            # 2)Best Pareto Front
            elif pop.fronts[i_1]!=pop.fronts[i_2]:
                if pop.fronts[i_1]-pop.fronts[i_2]>0:
                    idx_winners[j]=idx_for_tournament[i]
                else:
                    idx_winners[j]=idx_for_tournament[i+1]
            # 3)Highest Crowding Distance
            else:
                if pop.crowding_dist[i_1]-pop.crowding_dist[i_2]>0:
                    idx_winners[j]=idx_for_tournament[i_1]
                else:
                    idx_winners[j]=idx_for_tournament[i_2]
            j+=1
        return idx_winners

    def mutation_processes(self,new_product,new_batches,new_mask,pmut):
        """Mutation Processes:
            1. To mutate a product label with a rate of pMutP. 
            2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            3. To add a new random gene to the end of the chromosome (un- conditionally).
            4. To swap two genes within the same chromosome once with a rate of pSwap .

        Args:
            new_product (array of int): Number of Products
            new_batches (array of int): Number of Batches
            new_mask (array of bools): Mask of active genes
            pmut (tuple): Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)

        Returns:
            [type]: [description]
        """
        if (new_product>=self.num_products).any():
            raise Exception("Error in labels of products, labels superior than maximum defined.")
        # Active genes per chromossome
        genes_per_chromo=np.sum(new_mask,axis=1,dtype=int)
        # Loop per chromossome
        for i in range(0,len(new_product)):
            # 1. To mutate a product label with a rate of pMutP. 
            # print(new_product[i])
            new_product[i,0:genes_per_chromo[i]]=Mutations._label_mutation(new_product[i,0:genes_per_chromo[i]],self.num_products,pmut[0])
            # print(new_product[i])
            # 2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            # print(new_batches[i])
            new_batches[i,0:genes_per_chromo[i]]=Mutations._add_subtract_mutation(new_batches[i,0:genes_per_chromo[i]],pmut[1],pmut[2])
            # print(new_batches[i])
            # 3. To add a new random gene to the end of the chromosome (un- conditionally).
            # print(new_product[i])
            # print(new_batches[i])
            # print(new_mask[i])
            new_product[i,genes_per_chromo[i]]=random.randint(0,self.num_products-1)
            new_batches[i,genes_per_chromo[i]]=1
            new_mask[i,genes_per_chromo[i]]=True
            # print(new_product[i])
            # print(new_batches[i])
            # print(new_mask[i])
            # 4. To swap two genes within the same chromosome once with a rate of pSwap .
            # print(new_product[i])
            new_product[i,0:genes_per_chromo[i]],new_batches[i,0:genes_per_chromo[i]]=Mutations._swap_mutation(new_product[i,0:genes_per_chromo[i]],new_batches[i,0:genes_per_chromo[i]],pmut[3])
            # print(new_product[i])
        if (new_product>=self.num_products).any():
            raise Exception("Error in labels of products, labels superior than maximum defined.")

        return new_product,new_batches,new_mask

    def agg_product_batch(self,products,batches,masks):
        """Aggregates product batches in case of neighbours products.

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks
        """
        # Active genes per chromossome
        genes_per_chromo=np.sum(masks,axis=1,dtype=int)
        if any(genes_per_chromo)>1:
            # Loop per chromossome in population
            for j in range(0,len(genes_per_chromo)):
                if genes_per_chromo[j]>1:
                    # Loop per gene i in chromossome
                    # for i in range(0,genes_per_chromo[j]-1)
                    i=0
                    while i<genes_per_chromo[j]-1:
                        if products[j][i]==products[j][i+1]:
                            # Sum next
                            print(batches[j])
                            batches[j][i]=batches[j][i]+batches[j][i+1]
                            print(batches[j])
                            # Deletes [i+a] and insert a value in the last
                            print(batches[j])
                            batches[j]=np.insert(np.delete(batches[j],i+1),-1,0)
                            print(batches[j])
                            print(products[j])
                            products[j]=np.insert(np.delete(products[j],i+1),-1,0)
                            print(products[j])
                            print(masks[j])
                            masks[j]=np.insert(np.delete(masks[j],i),-1,False)
                            print(masks[j])
                        else:
                            i+=1
        return products,batches,masks

    def merge_pop_with_offspring(self,pop,pop_new):
        """Appends the offspring population to the Current population.

        Args:
            pop (class object): Current Population object
            pop_new (class object): Offspring population object
        """
        pop.num_chromossomes=pop.num_chromossomes+pop_new.num_chromossomes
        pop.num_genes=pop.num_genes+pop_new.num_genes

        # Batches
        pop.batches_raw=np.vstack((pop.batches_raw,pop_new.batches_raw))

        # Products
        pop.products_raw=np.vstack((pop.products_raw,pop_new.products_raw))

        # Masks
        pop.masks=np.vstack((pop.masks,pop_new.masks))

        # Time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        pop.start_raw=np.vstack((pop.start_raw,pop_new.start_raw))
        pop.end_raw=np.vstack((pop.end_raw,pop_new.end_raw))

        # Stock backlog_i
        pop.backlogs=np.vstack((pop.backlogs,pop_new.backlogs))

        # Objectives throughput_i,deficit_strat_i
        pop.objectives_raw=np.vstack((pop.objectives_raw,pop_new.objectives_raw))

        # Genes per chromossome (Number of active campaigns per solution)
        pop.genes_per_chromo=np.sum(pop.masks,axis=1,dtype=int)

        # List of dictionaries with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        for i in range(0,len(pop_new.dicts_batches_end_dsp)):
            pop.dicts_batches_end_dsp.append(pop_new.dicts_batches_end_dsp[i])

        # NSGA2
        # Creates an array of fronts and crowding distance
        pop.fronts=np.empty(shape=(pop.num_chromossomes,1),dtype=int)
        pop.crowding_dist=np.empty(shape=(pop.num_chromossomes,1),dtype=int)


    def main(self,num_chromossomes,num_geracoes,n_tour,perc_crossover,pmut):
        # 1) Random parent population is initialized with its attributes
        pop=Population(self.num_genes,num_chromossomes,self.num_products,self.num_objectives,self.start_date,self.initial_stock)
        # 1.1) Initializes class object for Offspring Population
        # Number of chromossomes for crossover, guarantees an even number
        n_parents = int(num_chromossomes * perc_crossover)
        if n_parents % 2 == 1:
            n_parents = n_parents + 1
        pop_offspring=Population(self.num_genes,n_parents,self.num_products,self.num_objectives,self.start_date,self.initial_stock)
        # 1.2) Creates start and end date from schedule assures only batches with End date<Last day of manufacturing

        # 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        self.calc_start_end(pop)       

        # 3)Calculate inventory levels and objectives
        self.calc_inventory_objectives(pop)

        # 4)Front Classification
        a0=np.sum(pop.objectives_raw)
        pop.fronts=AlgNsga2._fronts(pop.objectives_raw,self.num_fronts)
        a1=np.sum(pop.objectives_raw)
        if (a1-a0)!=0:
            raise Exception('Mutation is affecting values, consider making a deepcopy.')

        # 5) Crowding Distance
        # print(f"before after objectives {np.sum(pop.objectives_raw)}, fronts {np.sum(pop.fronts)}, check mutation")
        a0,b0=np.sum(pop.objectives_raw),np.sum(pop.fronts)
        pop.crowding_dist=AlgNsga2._crowding_distance(pop.objectives_raw,pop.fronts,self.big_dummy)
        a1,b1=np.sum(pop.objectives_raw),np.sum(pop.fronts)
        if ((a1-a0)!=0)|((b1-b0)!=0):
            raise Exception('Mutation is affecting values, consider making a deepcopy.')

        # 6)Selection for Crossover Tournament

        ix_to_crossover=self.tournament_restrictions_binary(pop,n_parents,n_tour)

        # 7)Crossover
        # 7.1 Sorts Selected by number of genes
        ix_to_crossover=ix_to_crossover[np.argsort(pop.genes_per_chromo[ix_to_crossover])]
        # 7.2 Creates a new population for offspring population crossover and calls uniform crossover 
        new_products,new_batches,new_mask=Crossovers._crossover_uniform(copy.deepcopy(pop.products_raw[ix_to_crossover]),copy.deepcopy(pop.batches_raw[ix_to_crossover]),copy.deepcopy(pop.masks[ix_to_crossover]),copy.deepcopy(pop.genes_per_chromo),perc_crossover)
        # pop_produto,pop_batches,pop_mask=AlgNsga2._crossover_uniform(pop_produto,pop_batches,pop_mask,genes_per_chromo)

        # 8)Mutation
        new_products,new_batches,new_mask=self.mutation_processes(new_products,new_batches,new_mask,pmut)

        # 9)Aggregate batches with same product neighbours
        new_products,new_batches,new_mask=self.agg_product_batch(new_products,new_batches,new_mask)

        # 10) Merge populations Current and Offspring
        # pop.append_offspring(new_products,new_batches,new_mask)
        pop_offspring.create_new_population(new_products,new_batches,new_mask)

        # 11) 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        self.calc_start_end(pop_offspring)       

        # 12) 3)Calculate inventory levels and objectives
        self.calc_inventory_objectives(pop_offspring)

        # 13) Merge Current Pop with Offspring
        pop_offspring_copy=copy.deepcopy(pop_offspring)
        self.merge_pop_with_offspring(pop,pop_offspring_copy)

        # 14) 4)Front Classification
        a0=np.sum(pop.objectives_raw)
        pop.fronts=AlgNsga2._fronts(pop.objectives_raw,self.num_fronts)
        a1=np.sum(pop.objectives_raw)
        if (a1-a0)!=0:
            raise Exception('Mutation is affecting values, consider making a deepcopy.')

        # 15) 5) Crowding Distance
        # print(f"before after objectives {np.sum(pop.objectives_raw)}, fronts {np.sum(pop.fronts)}, check mutation")
        a0,b0=np.sum(pop.objectives_raw),np.sum(pop.fronts)
        pop.crowding_dist=AlgNsga2._crowding_distance(pop.objectives_raw,pop.fronts,self.big_dummy)
        a1,b1=np.sum(pop.objectives_raw),np.sum(pop.fronts)
        if ((a1-a0)!=0)|((b1-b0)!=0):
            raise Exception('Mutation is affecting values, consider making a deepcopy.')


        print("Cheeers! Arrasooou!")
    
    def run_cprofile():
        num_chromossomes=100
        num_geracoes=200
        n_tour=2
        pcross=0.6
        # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
        pmut=(0.04,0.61,0.77,0.47)

        Planning().main(num_chromossomes,num_geracoes,n_tour,pcross,pmut)



if __name__=="__main__":
    # Planning.main()
    Planning.run_cprofile()

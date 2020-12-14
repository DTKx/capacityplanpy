import numpy as np
import random
import copy
# import timeit
import datetime
from dateutil.relativedelta import *
import pandas as pd
from dateutil import relativedelta
from numba import jit
from pygmo import *
from collections import defaultdict
from scipy import stats
import pickle
import time
from itertools import product
import concurrent.futures
import multiprocessing
import cProfile, pstats, io
from pstats import SortKey
import csv


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
        self.backlogs=np.zeros(shape=(num_chromossomes,num_months),dtype=float)

        # Initializes Inventory deficit per month (Objective 1, but with breakdown per month) [kg]
        self.deficit=np.zeros(shape=(num_chromossomes,num_months),dtype=float)

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
        self.update_genes_per_chromo()

    def extract_metrics(self,ix,num_fronts,num_exec,id_solution,name_var,ix_pareto):
        """Extract Metrics

        Args:
            ix (int): Index of the solution to verify metrics

        Returns:
            list: List with the metrics Total throughput [kg] Max total backlog [kg] Mean total backlog [kg] Median total backlog [kg] a Min total backlog [kg] P(total backlog ≤ 0 kg) 
                Max total inventory deficit [kg] Mean total inventory deficit [kg] a Median total inventory deficit [kg] Min total inventory deficit [kg]
        """
        metrics=[name_var,num_exec,id_solution]
        # Total throughput [kg] 
        metrics.append(self.objectives_raw[:,0][ix_pareto][ix])
        # Max total backlog [kg]
        metrics.append(np.max(self.backlogs[ix_pareto][ix]))
        # Mean total backlog [kg] +1stdev 
        metrics.append(np.mean(self.backlogs[ix_pareto][ix]))
        # Standard Dev
        metrics.append(np.std(self.backlogs[ix_pareto][ix]))
        # Median total backlog [kg]
        metrics.append(np.median(self.backlogs[ix_pareto][ix]))
        # Min total backlog [kg] 
        metrics.append(np.min(self.backlogs[ix_pareto][ix]))
        # P(total backlog ≤ 0 kg) 
        metrics.append(np.sum(self.backlogs[ix_pareto][ix]))
        # DeltaXY (total backlog) [kg]

        # Max total inventory deficit [kg]
        metrics.append(np.max(self.deficit[ix_pareto][ix]))
        # Mean total inventory deficit [kg] +1stdev 
        metrics.append(np.mean(self.deficit[ix_pareto][ix]))
        # Standard Dev
        metrics.append(np.std(self.deficit[ix_pareto][ix]))
        # Median total inventory deficit [kg] 
        metrics.append(np.median(self.deficit[ix_pareto][ix]))
        # Min total inventory deficit [kg] 
        metrics.append(np.min(self.deficit[ix_pareto][ix]))
        # DeltaXY (total inventory deficit) [kg]

        # Extra Metrics for plotting
        # Batches
        metrics.append(self.batches_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # Products
        metrics.append(self.products_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # Start of USP
        metrics.append(self.start_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])
        # End of DSP
        metrics.append(self.end_raw[ix_pareto][ix][self.masks[ix_pareto][ix]])

        return metrics

    def metrics_inversion_minimization(self,ref_point,volume_max,inversion_val_throughput,num_fronts,num_exec,name_var):
        """Extract the metrics only from the pareto front, inverts the inversion made to convert form maximization to minimization, organizes metrics and data for visualization.

        Returns:
            list: Array with metrics:
                "Hypervolume"
                Solution X "X Total throughput [kg]", "X Max total backlog [kg]", "X Mean total backlog [kg]", "X Median total backlog [kg]","X Min total backlog [kg]", "X P(total backlog ≤ 0 kg)","X Max total inventory deficit [kg]", "X Mean total inventory deficit [kg]", "X Median total inventory deficit [kg]", "X Min total inventory deficit [kg]" 
                Solution Y "Y Total throughput [kg]", "Y Max total backlog [kg]", "Y Mean total backlog [kg]", "Y Median total backlog [kg]","Y Min total backlog [kg]", "Y P(total backlog ≤ 0 kg)","Y Max total inventory deficit [kg]", "Y Mean total inventory deficit [kg]", "Y Median total inventory deficit [kg]", "Y Min total inventory deficit [kg]" Pareto Front
        """
        # Pareto Fronts
        ix_pareto=np.where(self.fronts==0)

        # Calculates hypervolume
        hv = hypervolume(points = self.objectives_raw[ix_pareto])
        hv_vol_norma=hv.compute(ref_point)/volume_max
        metrics_exec=[name_var,hv_vol_norma]
        # data_plot=[]

        # Reinverts again the throughput, that was modified for minimization by addying a constant
        self.objectives_raw[:,0]=inversion_val_throughput-self.objectives_raw[:,0]
        # Metrics
        ix_best_min=np.argmin(self.objectives_raw[:,0][ix_pareto])
        ix_best_max=np.argmax(self.objectives_raw[:,0][ix_pareto])
        # self.objectives_raw[ix_best_min]
        # self.objectives_raw[ix_best_max]

        metrics_id=[self.extract_metrics(ix_best_min,num_fronts,num_exec,"X",name_var,ix_pareto)]
        metrics_id.append(self.extract_metrics(ix_best_max,num_fronts,num_exec,"Y",name_var,ix_pareto))

        # Plot Data
        metrics_exec.append(self.objectives_raw[ix_pareto])
        return metrics_exec,metrics_id


    def metrics_inversion_violations(self,ref_point,volume_max,inversion_val_throughput,num_fronts,num_exec,name_var,violations):
        """Extract the metrics only from the pareto front, inverts the inversion made to convert form maximization to minimization, organizes metrics and data for visualization.

        Returns:
            list: Array with metrics:
                "Hypervolume"
                Solution X "X Total throughput [kg]", "X Max total backlog [kg]", "X Mean total backlog [kg]", "X Median total backlog [kg]","X Min total backlog [kg]", "X P(total backlog ≤ 0 kg)","X Max total inventory deficit [kg]", "X Mean total inventory deficit [kg]", "X Median total inventory deficit [kg]", "X Min total inventory deficit [kg]" 
                Solution Y "Y Total throughput [kg]", "Y Max total backlog [kg]", "Y Mean total backlog [kg]", "Y Median total backlog [kg]","Y Min total backlog [kg]", "Y P(total backlog ≤ 0 kg)","Y Max total inventory deficit [kg]", "Y Mean total inventory deficit [kg]", "Y Median total inventory deficit [kg]", "Y Min total inventory deficit [kg]" Pareto Front
        """
        # Pareto Fronts
        ix_pareto=np.where(self.fronts==0)

        # Calculates hypervolume
        hv = hypervolume(points = self.objectives_raw[ix_pareto])
        hv_vol_norma=hv.compute(ref_point)/volume_max
        metrics_exec=[name_var,hv_vol_norma]
        # data_plot=[]

        # Reinverts again the throughput, that was modified for minimization by addying a constant
        self.objectives_raw[:,0]=inversion_val_throughput-self.objectives_raw[:,0]
        # Metrics
        ix_best_min=np.argmin(self.objectives_raw[:,0][ix_pareto])
        ix_best_max=np.argmax(self.objectives_raw[:,0][ix_pareto])
        # self.objectives_raw[ix_best_min]
        # self.objectives_raw[ix_best_max]

        metrics_id=[self.extract_metrics(ix_best_min,num_fronts,num_exec,"X",name_var,ix_pareto)]
        metrics_id.append(self.extract_metrics(ix_best_max,num_fronts,num_exec,"Y",name_var,ix_pareto))

        # Plot Data
        metrics_exec.append(self.objectives_raw[ix_pareto])
        return metrics_exec,metrics_id


class Planning():
    # Class Variables

    # General Genetic Algorithms parameters

    # Number of genes
    num_genes=int(37)

    # # Mutation
    # pmutp=0.5
    # pposb=0.5
    # pnegb=0.5
    # pswap=0.5

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
    inversion_val_throughput=2000

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

    # Monte Carlo 

    # Index of values that are not zeros
    ix_not0=np.where(demand_distribution!=0)
    # Length of rows to calculate triangular
    tr_len=len(demand_distribution[ix_not0])
    # # Generates tr_demand 
    # tr_demand=np.zeros(shape=(tr_len,3))
    # for i in range(0,tr_len):
    #     tr_demand[i]=np.array(demand_distribution[ix_not0][i],dtype=np.float64)
    tr_demand=np.array([[ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [ 4.5 ,  5.5 ,  8.25],
        [ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [ 4.5 ,  5.5 ,  8.25],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [10.  , 11.  , 16.5 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 8.8 ,  9.8 , 14.7 ],
        [ 4.5 ,  5.5 ,  8.25],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [ 2.1 ,  3.1 ,  4.65],
        [ 4.5 ,  5.5 ,  8.25],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [10.  , 11.  , 16.5 ],
        [ 2.1 ,  3.1 ,  4.65],
        [ 4.5 ,  5.5 ,  8.25],
        [ 8.3 ,  9.3 , 13.95],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 8.8 ,  9.8 , 14.7 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 4.5 ,  5.5 ,  8.25],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 4.5 ,  5.5 ,  8.25],
        [ 4.5 ,  5.5 ,  8.25],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [10.  , 11.  , 16.5 ],
        [ 8.3 ,  9.3 , 13.95],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 8.3 ,  9.3 , 13.95],
        [ 8.8 ,  9.8 , 14.7 ],
        [10.  , 11.  , 16.5 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 2.1 ,  3.1 ,  4.65],
        [10.  , 11.  , 16.5 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 2.1 ,  3.1 ,  4.65],
        [ 8.8 ,  9.8 , 14.7 ],
        [ 4.5 ,  5.5 ,  8.25],
        [ 8.8 ,  9.8 , 14.7 ],
        [ 8.3 ,  9.3 , 13.95],
        [10.  , 11.  , 16.5 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [10.  , 11.  , 16.5 ],
        [ 8.3 ,  9.3 , 13.95],
        [ 8.8 ,  9.8 , 14.7 ],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 3.9 ,  4.9 ,  7.35],
        [ 4.5 ,  5.5 ,  8.25],
        [ 5.2 ,  6.2 ,  9.3 ],
        [ 4.5 ,  5.5 ,  8.25]])
    # External File Monte Carlo Simulations
    # Open pickle dict Results
    file_name='demand_montecarlo.pkl'
    root_path_data = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
    path=root_path_data+file_name
    infile = open(path,'rb')
    demand_montecarlo= pickle.load(infile)
    infile.close()
    num_demands=len(demand_montecarlo)

    # NSGA Variables

    # Number of fronts created
    num_fronts=3

    # Big Dummy for crowding distance computation
    big_dummy=10**5

    # Hypervolume parameters

    # Reference point
    ref_point=[inversion_val_throughput+500,2500]
    # hv_vol_norma=volume_ger
    volume_max=np.prod(ref_point)


    def calc_start_end(self,pop_obj):
        """Calculates start and end dates of batch manufacturing, as well as generates (dicts_batches_end_dsp) a list of dictionaries (List index = Chromossome, key=Number of products and date values os release from QC) per chromossome with release dates of each batch per product. 

        Args:
            pop_obj (Class object): Class Object of the population to be analized
        """
        # Extracts the population informations
        dsp_raw=np.vectorize(self.dsp_days.__getitem__)(pop_obj.products_raw)
        usp_plus_dsp_raw=np.vectorize(self.usp_days.__getitem__)(pop_obj.products_raw)+(dsp_raw).copy()

        # Initialize by addying the first date
        pop_obj.start_raw[:,0]=self.start_date

        # Loop per chromossome i
        for i in range(0,len(pop_obj.start_raw)):
            # if np.sum(pop_obj.masks[i][pop_obj.genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")
            # if any(pop_obj.batches_raw[i][pop_obj.masks[i]]==0):
            #     raise Exception("Invalid number of batches (0).")

            batches_end_date_i=defaultdict(list)
            # Evaluates gene zero
            j=0
            # List of batches end date End date=start+(USP+DSP)*1+DSP*num_batches
            end_dates=[pop_obj.start_raw[i][j]+np.timedelta64(usp_plus_dsp_raw[i][j],'D')+np.timedelta64(dsp_raw[i][j]*k,'D') for k in range(0,pop_obj.batches_raw[i][j])]
            # Verifies if End day<Last Day ok else delete
            num_batch_exceed_end_dates=np.sum(np.array(end_dates)>self.end_date)
            if num_batch_exceed_end_dates>0:
                # Removes exceeding batches
                pop_obj.batches_raw[i][j]=pop_obj.batches_raw[i][j]-num_batch_exceed_end_dates
                del end_dates[-num_batch_exceed_end_dates:]
                if pop_obj.batches_raw[i][j]==0:
                    # Removes the batch in position j and adds a batch 0 to the last one
                    temp_b=pop_obj.batches_raw[i].copy()
                    temp_b[j:-1]=temp_b[j+1:]
                    temp_b[-1]=0
                    pop_obj.masks[i][pop_obj.genes_per_chromo[i]-1]=False
                    pop_obj.batches_raw[i]=temp_b.copy()
                    continue
            # Add first campaign end date0=start0+(USP+DSP)*1+DSP*num_batches
            pop_obj.end_raw[i][j]=end_dates[-1]
            # Addying the quality control time
            end_dates=end_dates+np.timedelta64(self.qc_days[pop_obj.products_raw[i][j]],'D')
            # Appends to the dictionary 
            for date in end_dates:
                batches_end_date_i[pop_obj.products_raw[i][j]].append(date)
            j+=1
            # Loop per gene j starting from second gene
            while j<pop_obj.genes_per_chromo[i]:
                # Add a Start Date=Previous End Date+Change Over Time
                pop_obj.start_raw[i,j]=pop_obj.end_raw[i,j-1]+np.timedelta64(self.setup_key_to_subkey[pop_obj.products_raw[i,j]][pop_obj.products_raw[i,j-1]],'D')

                # if any(pop_obj.batches_raw[i][pop_obj.masks[i]]==0):
                #     raise Exception("Invalid number of batches (0).")
                # List of batches end date End date=start+(USP+DSP)*1+DSP*num_batches
                end_dates=[pop_obj.start_raw[i][j]+np.timedelta64(usp_plus_dsp_raw[i][j],'D')+np.timedelta64(dsp_raw[i][j]*k,'D') for k in range(0,pop_obj.batches_raw[i][j])]

                # Verifies if End day<Last Day ok else delete
                num_batch_exceed_end_dates=np.sum(np.array(end_dates)>self.end_date)
                if num_batch_exceed_end_dates>0:
                    # Number of batches
                    # print("Number of Batches before removal: ",pop_obj.batches_raw[i][j])
                    # Removes exceeding batches
                    pop_obj.batches_raw[i][j]=pop_obj.batches_raw[i][j]-num_batch_exceed_end_dates
                    del end_dates[-num_batch_exceed_end_dates:]
                    if pop_obj.batches_raw[i][j]==0:
                        # raise Exception("Invalid bool after number of batches.")
                        # Removes the batch in position j and adds a batch 0 to the last one
                        temp_b=pop_obj.batches_raw[i].copy()
                        temp_b[j:-1]=temp_b[j+1:]
                        temp_b[-1]=0
                        # print("Number of Batches before removal: ",pop_obj.batches_raw[i][pop_obj.masks[i]])
                        pop_obj.masks[i][pop_obj.genes_per_chromo[i]-1]=False
                        pop_obj.batches_raw[i]=temp_b.copy()
                        # print("Number of Batches after removal: ",pop_obj.batches_raw[i][pop_obj.masks[i]])
                        pop_obj.genes_per_chromo[i]=pop_obj.genes_per_chromo[i]-1
                        continue
                        # if np.sum(pop_obj.masks[i][pop_obj.genes_per_chromo[i]:])>0:
                        #     raise Exception("Invalid bool after number of active genes.")
                # Add first campaign end date0=start0+(USP+DSP)*1+DSP*num_batches
                pop_obj.end_raw[i][j]=end_dates[-1]
                # Addying the quality control time
                end_dates=end_dates+np.timedelta64(self.qc_days[pop_obj.products_raw[i][j]],'D')
                # Appends to the dictionary
                for date in end_dates:
                    batches_end_date_i[pop_obj.products_raw[i][j]].append(date)           
                j+=1
            # if np.sum(pop_obj.masks[i][pop_obj.genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")
            # Appends dictionary of individual to the list of dictionaries
            pop_obj.dicts_batches_end_dsp.append(batches_end_date_i)

            # # Produced Month 0 is the first month of inventory batches
            # produced_i=np.zeros(shape=(self.num_months,self.num_products),dtype=int)
            # for key in pop.dicts_batches_end_dsp[i].keys():
            #     # Aggregated count per month 
            #     # aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='end').sum()
            #     aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='start').sum()
            #     for k in range(0,len(aggregated)):
            #         m = relativedelta.relativedelta(aggregated.index[k],self.date_stock).months
            #         # Updates the month with the number of batches produced
            #         produced_i[m,key]=aggregated[k]



        # Updates Genes per Chromo
        pop_obj.update_genes_per_chromo()

        # for i in range(0,len(pop_obj.products_raw)):
        #     if np.sum(pop_obj.masks[i][pop_obj.genes_per_chromo[i]:])>0:
        #         raise Exception("Invalid bool after number of active genes.")

    @staticmethod
    # @jit(nopython=True,nogil=True,fastmath=True,parallel=True)
    @jit(nopython=True,nogil=True,fastmath=True)
    def calc_triangular_dist(demand_distribution,num_monte):
        n=len(demand_distribution)
        demand_i=np.zeros(shape=(n,))
        # demand_i=np.median(np.random.triangular(demand_distribution[:][0],demand_distribution[:][1],demand_distribution[:][2],size=num_monte))
        # Loop in line
        for i in np.arange(0,n):
            demand_i[i]=np.median(np.random.triangular(demand_distribution[i][0],demand_distribution[i][1],demand_distribution[i][2],size=num_monte))
        return demand_i

    def calc_demand_montecarlo_to_external_file(self,n_exec_demand):
        """Performs a Montecarlo Simulation to define the Demand of products, uses a demand_distribution for containing either 0 as expected or a triangular distribution (minimum, mode (most likely),maximum) values in kg

        Args:
            n_exec_demand ([type]): Number of Executions of demand calculations
        """       
        demand_dict={}
        for i in range(0,n_exec_demand):
            demand_dict[(i)]=self.calc_triangular_dist(self.tr_demand,self.num_monte)
            print(i)
        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        # Export Pickle
        file_name = "demand_montecarlo.pkl"
        path = root_path + file_name
        file_pkl = open(path, "wb")
        pickle.dump(demand_dict, file_pkl)
        file_pkl.close()

    def load_demand_montecarlo(self,line,col):
        """Loads random demand profile generated by Monte Carlo Simulation.
        Args:
        """       
        i=random.randint(0,self.num_demands-1)
        demand_i=np.zeros(shape=(line,col))
        demand_i[self.ix_not0]=self.demand_montecarlo[i]
        return demand_i

    def calc_demand_montecarlo(self,line,col):
        """Performs a Montecarlo Simulation to define the Demand of products, uses a demand_distribution for containing either 0 as expected or a triangular distribution (minimum, mode (most likely),maximum) values in kg

        Args:
        """       
        demand_i=np.zeros(shape=(line,col))
        demand_i[self.ix_not0]=self.calc_triangular_dist(self.tr_demand,self.num_monte)
        return demand_i

    @staticmethod
    # @jit(nopython=True,nogil=True,fastmath=True)
    def calc_objective_deficit_strat(target_stock_i,stock_i):
        deficit_strat_i=target_stock_i-stock_i
        # Corrects negative values
        ix_neg=np.where(deficit_strat_i<0.0)
        # ix_neg=np.where(deficit_strat_i<np.float64(0.0))
        # ix_neg=np.where(deficit_strat_i<np.float64(0.0))[0]
        if len(ix_neg)>int(0):
            # print(deficit_strat_i)
            deficit_strat_i[ix_neg]=0.0
            # print(deficit_strat_i)
        # Sum of all product deficit per month
        # return np.sum(deficit_strat_i,axis=1)
        # return deficit_strat_i
        return np.median(deficit_strat_i)

    @staticmethod
    @jit(nopython=True,nogil=True,fastmath=True)
    def calc_stock(available_i,stock_i,produced_i,demand_i,backlog_i,num_months):
        """Calculates Stock per month along (over num_months) Stock=Available-Demand if any<0 Stock=0 & Back<0 = else.

        Args:
            available_i (array of int): Number of available batches per month (each column represents a month). Available=Previous Stock+Produced this month
            stock_i (array of int): Stock Available. Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            produced_i (array of int): Produced per month
            demand_i (array of int): Demand on month
            backlog_i (array of int): Backlog in the month
            num_months (int): Number of months to evaluate each column represents a month
        """
        # Loop per Months starting through 1
        for j in np.arange(1,num_months):
            # Available=Previous Stock+Produced this month
            available_i[j]=stock_i[j-1]+produced_i[j]

            # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            stock_i[j]=available_i[j]-demand_i[j]
            # Corrects negative values
            ix_neg=np.where(stock_i[j]<0)[0]
            if len(ix_neg)>0:
                # Adds negative values to backlog
                backlog_i[j][ix_neg]=(stock_i[j][ix_neg])*(int(-1))
                # print(f"backlog {backlog_i[j][ix_neg]}")
                # Corrects if Stock is negative
                stock_i[j][ix_neg]=int(0)
                # print(f"backlog {backlog_i[j][ix_neg]} check if mutated after assignement of stock")
        return stock_i,backlog_i

    def calc_inventory_objectives(self,pop):
        """Calculates Inventory levels returning the backlog and calculates the objectives the total deficit and total throughput addying to the pop attribute

        Args:
            pop (class object): Population object to calculate Inventory levels
        """
        # Creates a vector for batch/kg por the products 
        pop_yield=np.vectorize(self.yield_kg_batch.__getitem__)(pop.products_raw) #Equivalentt to         # pop_yield=np.array(list(map(self.yield_kg_batch.__getitem__,pop_products)))

        # Loop per Chromossome
        for i in range(0,len(pop.products_raw)):
            # if any(pop.batches_raw[i][pop.masks[i]]==0):
            #     raise Exception("Invalid number of batches (0).")

            available_i=np.zeros(shape=(self.num_months,self.num_products))
            stock_i=np.zeros(shape=(self.num_months,self.num_products))
            backlog_i=np.zeros(shape=(self.num_months,self.num_products))

            # Produced Month 0 is the first month of inventory batches
            produced_i=np.zeros(shape=(self.num_months,self.num_products),dtype=int)
            for key in pop.dicts_batches_end_dsp[i].keys():
                # Aggregated count per month 
                # aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='end').sum()
                aggregated=pd.Series(1,index=pd.to_datetime(pop.dicts_batches_end_dsp[i][key])).resample("M", convention='start').sum()
                for k in range(0,len(aggregated)):
                    m = relativedelta.relativedelta(aggregated.index[k],self.date_stock).months
                    # Updates the month with the number of batches produced
                    produced_i[m,key]=aggregated[k]
            # Conversion batches to kg
            produced_i=produced_i*self.yield_kg_batch_ar
            # print("Produced",produced_i)

            # Calling Monte Carlo to define demand
            # demand_i=self.calc_demand_montecarlo(self.num_monte,self.num_months,self.num_products)
            demand_i=self.load_demand_montecarlo(self.num_months,self.num_products)

            # Loop per Months (Values already in kg)

            # Evaluates stock for Initial Month (0)
            # Available=Previous Stock+Produced this month
            available_i[0,:]=self.initial_stock+produced_i[0,:]
            # Stock=Available-Demand if any<0 Stock=0 & Back<0 = else
            stock_i[0,:]=available_i[0,:]-demand_i[0,:]
            # Corrects negative values
            ix_neg=np.where(stock_i[0,:]<0)
            if len(ix_neg)>0:
                # Adds negative values to backlog
                backlog_i[0,:][ix_neg]=(stock_i[0,:][ix_neg]).copy()*(-1)
                # Corrects if Stock is negative
                stock_i[0,:][ix_neg]=0
            # Evaluates Stock over all months
            # print("inStock")
            # print(stock_i)
            # print("inBack")
            # print(backlog_i)
            # print("Produced",produced_i)
            stock_i,backlog_i=self.calc_stock(available_i,stock_i,produced_i,demand_i,backlog_i,self.num_months)
            # print("Produced",produced_i)
            # print("ouStock")
            # print(stock_i)
            # print("ouBack")
            # print(backlog_i)

            # Stores sum of all products backlogs per month
            pop.backlogs[i]=np.median(backlog_i,axis=1).T
            # pop.backlogs[i]=np.sum(backlog_i,axis=1).T

            # Calculates the objective Strategic Deficit 
            deficit=self.calc_objective_deficit_strat(self.target_stock,stock_i)
            pop.deficit[i]=deficit
            pop.objectives_raw[i,1]=deficit

            # pop.deficit[i]=np.median(deficit,axis=1)
            # pop.objectives_raw[i,1]=np.median(pop.deficit[i])
            # pop.deficit[i]=self.calc_objective_deficit_strat(self.target_stock,stock_i)
            # pop.objectives_raw[i,1]=np.sum(pop.deficit[i])

            # Calculates the objective Throughput
            # a=np.sum(produced_i)
            pop.objectives_raw[i,0]=np.dot(pop.batches_raw[i][pop.masks[i]],pop_yield[i][pop.masks[i]])
            # if pop.objectives_raw[i,0]-a>1:
            #     raise Exception("Error in Objective 1")
            # pop.objectives_raw[i,0]=np.dot(pop.batches_raw[i][pop.masks[i]],pop_yield[i][pop.masks[i]])
            # Inversion of the Throughput by a fixed value to generate a minimization problem
            pop.objectives_raw[i,0]=self.inversion_val_throughput-pop.objectives_raw[i,0]

    def calc_violations(self,pop):
        """Calculates number of violations of constraints, each type of violation is type any, ie if any campaign violates it counts as one.
        Considers 1)Median Backlog>0, 2)Minimum number of batches, 3)Maximum number of batches, 4)Multiples of number of batches

        Args:
            pop (class object): Class object to evaluate 

        Returns:
            array: Array with number of violations per individual
        """
        # 1)Median Backlog>0, 
        num_violations=np.median(pop.backlogs,axis=1)
        num_violations[num_violations>0]=1
        num_violations[num_violations<=0]=0

        min_batch_raw=np.vectorize(self.min_batch.__getitem__)(pop.products_raw)
        max_batch_raw=np.vectorize(self.max_batch.__getitem__)(pop.products_raw)
        batch_multiples_raw=np.vectorize(self.batch_multiples.__getitem__)(pop.products_raw)
        # Loop per chromossome
        for i in range(0,pop.num_chromossomes):
            # Counter for num of violations
            # 2)Minimum number of batches, 
            v_min=(pop.batches_raw[i,:pop.genes_per_chromo[i]]<min_batch_raw[i,:pop.genes_per_chromo[i]]).any()
            # 3)Maximum number of batches, 
            v_max=(pop.batches_raw[i,:pop.genes_per_chromo[i]]>max_batch_raw[i,:pop.genes_per_chromo[i]]).any()
            # 4)Multiples of number of batches
            v_mult=(np.remainder(pop.batches_raw[i,:pop.genes_per_chromo[i]],batch_multiples_raw[i,:pop.genes_per_chromo[i]])!=0).any()
            num_violations[i]+=v_min+v_max+v_mult
        return num_violations

    def calc_violations_unit(self,pop):
        """Calculates number of violations of constraints, each violation counts one.
        Considers 1)Median Backlog>0, 2)Minimum number of batches, 3)Maximum number of batches, 4)Multiples of number of batches

        Args:
            pop (class object): Class object to evaluate 

        Returns:
            array: Array with number of violations per individual
        """
        # 1)Median Backlog>0, 
        num_violations=np.sum(pop.backlogs>0,axis=1)

        min_batch_raw=np.vectorize(self.min_batch.__getitem__)(pop.products_raw)
        max_batch_raw=np.vectorize(self.max_batch.__getitem__)(pop.products_raw)
        batch_multiples_raw=np.vectorize(self.batch_multiples.__getitem__)(pop.products_raw)
        # Loop per chromossome
        for i in range(0,pop.num_chromossomes):
            # Counter for num of violations
            # # 2)Minimum number of batches, 
            v_min=np.sum(pop.batches_raw[i,:pop.genes_per_chromo[i]]>=min_batch_raw[i,:pop.genes_per_chromo[i]])
            # # 3)Maximum number of batches, 
            v_max=np.sum(pop.batches_raw[i,:pop.genes_per_chromo[i]]<=max_batch_raw[i,:pop.genes_per_chromo[i]])
            # # 4)Multiples of number of batches
            v_mult=np.sum(np.remainder(pop.batches_raw[i,:pop.genes_per_chromo[i]],batch_multiples_raw[i,:pop.genes_per_chromo[i]])!=0)
            num_violations[i]+=v_min+v_max+v_mult
        return num_violations

    # @staticmethod
    def tournament_restrictions_binary(self,pop,n_parents,n_tour,num_violations):
        """Tournament with replacement for selection to crossover, considering those criteria:
        1)Lowest number of constraints: 1)Median Backlog>0, 2)Minimum number of batches, 3)Maximum number of batches, 4)Multiples of number of batches. If draw then: 
        2)Best pareto front, if draw then:
        3)Highest Crowding Distance

        Args:
            pop (object): Population object that contains the individuals to select
            n_parents (int): Number of chromossomes to select
            n_tour (int): Number of individuals to compete during each tournament

        Returns:
            array: Array with indexes of selected individuals
        """
        # # Calculates number of violated constraints
        # num_violations=self.calc_violations(pop)

        # Backlogs contains values per month
        aggregated_backlogs=np.sum(pop.backlogs,axis=1)
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
            # 1) Lowest Restrictions
            if num_violations[i_1]!=num_violations[i_2]:
                # To change for different number of tours c=np.where(a==np.min(a))
                if num_violations[i_1]-num_violations[i_2]>0:
                    idx_winners[j]=i_1
                else:
                    idx_winners[j]=i_2
            # 2)Best Pareto Front
            elif pop.fronts[i_1]!=pop.fronts[i_2]:
                if pop.fronts[i_1]-pop.fronts[i_2]>0:
                    idx_winners[j]=i_1
                else:
                    idx_winners[j]=i_2
            # 3)Highest Crowding Distance
            else:
                if pop.crowding_dist[i_1]-pop.crowding_dist[i_2]>0:
                    idx_winners[j]=i_1
                else:
                    idx_winners[j]=i_2
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
       
        # if (new_product>=self.num_products).any():
        #     raise Exception("Error in labels of products, labels superior than maximum defined.")
        # Active genes per chromossome
        genes_per_chromo=np.sum(new_mask,axis=1,dtype=int)
        # Loop per chromossome
        for i in range(0,len(new_product)):
            # if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")
            # # print(new_batches[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # 1. To mutate a product label with a rate of pMutP. 
            # print("In label",new_product[i])
            new_product[i,0:genes_per_chromo[i]]=Mutations._label_mutation(new_product[i,0:genes_per_chromo[i]],self.num_products,pmut[0])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # print(new_product[i])
            # 2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
            # print("In add_subtract",new_batches[i])
            new_batches[i,0:genes_per_chromo[i]]=Mutations._add_subtract_mutation(new_batches[i,0:genes_per_chromo[i]],pmut[1],pmut[2])
            # print(new_batches[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # 3. To add a new random gene to the end of the chromosome (un- conditionally).
            # print(new_product[i])
            # print("In new gene",new_batches[i])
            # print(new_mask[i])
            new_product[i,genes_per_chromo[i]]=random.randint(0,self.num_products-1)
            new_batches[i,genes_per_chromo[i]]=1
            new_mask[i,genes_per_chromo[i]]=True
            genes_per_chromo[i]=genes_per_chromo[i]+1
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # print(new_product[i])
            # print(new_batches[i])
            # print(new_mask[i])
            # 4. To swap two genes within the same chromosome once with a rate of pSwap .
            # print("In Swap",new_product[i])
            new_product[i,0:genes_per_chromo[i]],new_batches[i,0:genes_per_chromo[i]]=Mutations._swap_mutation(new_product[i,0:genes_per_chromo[i]],new_batches[i,0:genes_per_chromo[i]],pmut[3])
            # print(new_product[i])
            # if any(new_batches[i][new_mask[i]]==0):
            #     raise Exception("Invalid number of batches (0).")
            # if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
            #     raise Exception("Invalid bool after number of active genes.")

        # if (new_product>=self.num_products).any():
        #     raise Exception("Error in labels of products, labels superior than maximum defined.")

        return new_product,new_batches,new_mask

    # @jit(nopython=True,nogil=True,fastmath=True,parallel=True)
    @staticmethod
    @jit(nopython=True,nogil=True,fastmath=True)
    def agg_product_batch(products,batches,masks,genes_per_chromo):
        """Aggregates product batches in case of neighbours products.

        Args:
            products (array): Array of products
            batches (array): Array of batches
            masks (array): Array of masks
        """
        # # Active genes per chromossome
        # genes_per_chromo=np.sum(masks,axis=1,dtype=int)
        if (genes_per_chromo>1).any():
            # Loop per chromossome in population
            for j in np.arange(0,len(genes_per_chromo)):
                # if np.sum(masks[j,genes_per_chromo[j]:])>0:
                #     raise Exception("Invalid bool after number of active genes.")
                if genes_per_chromo[j]>1:
                    # Loop per gene i in chromossome
                    # for i in range(0,genes_per_chromo[j]-1)
                    i=0
                    while i<genes_per_chromo[j]-1:
                        if products[j,i]==products[j,i+1]:
                            # Option 2
                            # Sum next
                            # print(batches[j])
                            batches[j,i]=batches[j,i]+batches[j,i+1]
                            # print(batches[j])
                            # Deletes [i+a] and insert a value in the last
                            # print(batches[j])
                            # Brings the sequence forward and sets the last value as 0
                            temp_ar=batches[j,i+2:].copy()
                            batches[j,i+1:-1]=temp_ar
                            batches[j,-1]=0
                            # print(batches[j])
                            # print(products[j])
                            # Brings the sequence forward and sets the last value as 0
                            temp_ar=products[j,i+2:].copy()
                            products[j,i+1:-1]=temp_ar
                            products[j,-1]=0
                            # print(products[j])
                            # print(masks[j])
                            masks[j,genes_per_chromo[j]-1]=False
                            genes_per_chromo[j]=genes_per_chromo[j]-1
                            # print(masks[j])
                        else:
                            i+=1
                # if np.sum(masks[j,genes_per_chromo[j]:])>0:
                #     raise Exception("Invalid bool after number of active genes.")
        return products,batches,masks

    def merge_pop_with_offspring(self,pop,pop_new):
        """Appends the offspring population to the Current population.

        Args:
            pop (class object): Current Population object
            pop_new (class object): Offspring population object
        """
        # Batches
        pop.batches_raw=np.vstack((pop.batches_raw,pop_new.batches_raw))
        pop.num_chromossomes=len(pop.batches_raw)

        # Products
        pop.products_raw=np.vstack((pop.products_raw,pop_new.products_raw))

        # Masks
        pop.masks=np.vstack((pop.masks,pop_new.masks))

        # Time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        pop.start_raw=np.vstack((pop.start_raw,pop_new.start_raw))
        pop.end_raw=np.vstack((pop.end_raw,pop_new.end_raw))

        # Stock backlog_i
        pop.backlogs=np.vstack((pop.backlogs,pop_new.backlogs))

        # Stock Deficit_i
        pop.deficit=np.vstack((pop.deficit,pop_new.deficit))

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

    def select_pop_by_index(self,pop,ix_reinsert):
        """Selects chromossomes to maintain in pop class object, updating the class atributes given the index.

        Args:
            pop (class object): Population Class Object to reduce based on selected index ix_reinsert
            ix_reinsert (array): Indexes selected to maintain in the population class object.
        """
        pop.num_chromossomes=len(ix_reinsert)

        # Batches
        pop.batches_raw=pop.batches_raw[ix_reinsert]

        # Products
        pop.products_raw=pop.products_raw[ix_reinsert]

        # Masks
        pop.masks=pop.masks[ix_reinsert]

        # Time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        pop.start_raw=pop.start_raw[ix_reinsert]
        pop.end_raw=pop.end_raw[ix_reinsert]

        # Stock backlog_i
        pop.backlogs=pop.backlogs[ix_reinsert]

        # Stock Deficit_i
        pop.deficit=pop.deficit[ix_reinsert]

        # Objectives throughput_i,deficit_strat_i
        pop.objectives_raw=pop.objectives_raw[ix_reinsert]

        # Genes per chromossome (Number of active campaigns per solution)
        pop.genes_per_chromo=np.sum(pop.masks,axis=1,dtype=int)

        # List of dictionaries with the index of list equal to the chromossome, keys of dictionry with the number of the product and the value as the number of batches produced
        pop.dicts_batches_end_dsp=list(map(pop.dicts_batches_end_dsp.__getitem__,list(ix_reinsert)))

        # NSGA2
        # Creates an array of fronts and crowding distance
        pop.fronts=pop.fronts[ix_reinsert]
        pop.crowding_dist=pop.crowding_dist[ix_reinsert]

    def main(self,num_exec,num_chromossomes,num_geracoes,n_tour,perc_crossover,pmut):
        var="front_nsga,tour_vio,rein_vio,metrics_pareto_vio"
        name_var=f'{var},{num_chromossomes},{num_geracoes},{n_tour},{perc_crossover},{pmut}'
        print("START")
        # 1) Random parent population is initialized with its attributes
        pop=Population(self.num_genes,num_chromossomes,self.num_products,self.num_objectives,self.start_date,self.initial_stock,self.num_months)
        # 1.1) Initializes class object for Offspring Population
        # Number of chromossomes for crossover, guarantees an even number
        n_parents = int(num_chromossomes * perc_crossover)
        if n_parents % 2 == 1:
            n_parents = n_parents + 1
        pop_offspring=Population(self.num_genes,n_parents,self.num_products,self.num_objectives,self.start_date,self.initial_stock,self.num_months)
        # 1.2) Creates start and end date from schedule assures only batches with End date<Last day of manufacturing

        # 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
        self.calc_start_end(pop)       

        # 3)Calculate inventory levels and objectives
        self.calc_inventory_objectives(pop)
        # if (pop.objectives_raw<0).any():
        #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")

        # 4)Front Classification
        # a0=np.sum(copy.deepcopy(pop.objectives_raw))
        pop.fronts=AlgNsga2._fronts(pop.objectives_raw,self.num_fronts)
        # violations=self.calc_violations(pop)
        # pop.fronts=AlgNsga2._fronts_violations(pop.objectives_raw,self.num_fronts,violations)
   
        # a1=np.sum(pop.objectives_raw)
        # if (a1-a0)!=0:
        #     raise Exception('Mutation is affecting values, consider making a deepcopy.')
        # if (pop.objectives_raw<0).any():
        #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")

        # 5) Crowding Distance
        # print(f"before after objectives {np.sum(pop.objectives_raw)}, fronts {np.sum(pop.fronts)}, check mutation")
        # a0,b0=np.sum(copy.deepcopy(pop.objectives_raw)),np.sum(copy.deepcopy(pop.fronts))
        pop.crowding_dist=AlgNsga2._crowding_distance(pop.objectives_raw,pop.fronts,self.big_dummy)
        # a1,b1=np.sum(pop.objectives_raw),np.sum(pop.fronts)
        # if ((a1-a0)!=0)|((b1-b0)!=0):
        #     raise Exception('Mutation is affecting values, consider making a deepcopy.')
        # if (pop.objectives_raw<0).any():
        #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")

        for i_gen in range(0,num_geracoes):
            print("Generation ",i_gen)

        #     for i in range(0,len(pop.products_raw)):
        #         if any(pop.batches_raw[i][pop.masks[i]]==0):
        #             raise Exception("Invalid number of batches (0).")
        #         if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
        #             raise Exception("Invalid bool after number of active genes.")

            # 6)Selection for Crossover Tournament

            # ix_to_crossover=self.tournament_restrictions_binary(pop,n_parents,n_tour,violations)
            violations=self.calc_violations(pop)
            ix_to_crossover=self.tournament_restrictions_binary(pop,n_parents,n_tour,violations)
            # selected_num_genes=copy.deepcopy(pop.genes_per_chromo)[ix_to_crossover]
            # sorted_ix=np.argsort(selected_num_genes)
            # ix_to_crossover=ix_to_crossover[sorted_ix]
            # 7)Crossover
            # 7.1 Sorts Selected by number of genes
            ix_to_crossover=ix_to_crossover[np.argsort(copy.deepcopy(pop.genes_per_chromo)[ix_to_crossover])]
            # 7.2 Creates a new population for offspring population crossover and calls uniform crossover 
            # new_products,new_batches,new_mask=Crossovers._crossover_uniform(copy.deepcopy(pop.products_raw[ix_to_crossover]),copy.deepcopy(pop.batches_raw[ix_to_crossover]),copy.deepcopy(pop.masks[ix_to_crossover]),copy.deepcopy(pop.genes_per_chromo),perc_crossover)
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            new_products,new_batches,new_mask=Crossovers._crossover_uniform(copy.deepcopy(pop.products_raw[ix_to_crossover]),copy.deepcopy(pop.batches_raw[ix_to_crossover]),copy.deepcopy(pop.masks[ix_to_crossover]),perc_crossover)

            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # pop_produto,pop_batches,pop_mask=AlgNsga2._crossover_uniform(pop_produto,pop_batches,pop_mask,genes_per_chromo)
            # 8)Mutation
            new_products,new_batches,new_mask=self.mutation_processes(new_products,new_batches,new_mask,pmut)

            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 9)Aggregate batches with same product neighbours
            # if i_gen>10:
            #     print("Hey")
            # Active genes per chromossome
            genes_per_chromo=np.sum(new_mask,axis=1,dtype=int)
            new_products,new_batches,new_mask=self.agg_product_batch(new_products,new_batches,new_mask,genes_per_chromo)
            genes_per_chromo=np.sum(new_mask,axis=1,dtype=int)

            # for i in range(0,len(new_products)):
            #     if any(new_batches[i][new_mask[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(new_mask[i][genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 10) Merge populations Current and Offspring
            # pop.append_offspring(new_products,new_batches,new_mask)
            pop_offspring.create_new_population(new_products,new_batches,new_mask)
            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 11) 2) Is calculated along Step 1, Note that USP end dates are calculated, but not stored.
            self.calc_start_end(pop_offspring)       
            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 12) 3)Calculate inventory levels and objectives
            self.calc_inventory_objectives(pop_offspring)
            # for i in range(0,len(pop_offspring.products_raw)):
            #     if any(pop_offspring.batches_raw[i][pop_offspring.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop_offspring.masks[i][pop_offspring.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # if (pop_offspring.objectives_raw<0).any():
            #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")
            # 13) Merge Current Pop with Offspring
            # pop_offspring_copy=copy.deepcopy(pop_offspring)
            self.merge_pop_with_offspring(pop,pop_offspring)
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")
            # if (pop.objectives_raw<0).any():
            #     raise Exception ("Negative value of objectives, consider modifying the inversion value.")
  
            # 14) 4)Front Classification
            # a0=np.sum(copy.deepcopy(pop.objectives_raw))
            pop.fronts=AlgNsga2._fronts(pop.objectives_raw,self.num_fronts)

            # violations=self.calc_violations(pop)
            # pop.fronts=AlgNsga2._fronts_violations(pop.objectives_raw,self.num_fronts,violations)

            # a1=np.sum(pop.objectives_raw)
            # if (a1-a0)!=0:
            #     raise Exception('Mutation is affecting values, consider making a deepcopy.')

            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 15) 5) Crowding Distance
            # print(f"before after objectives {np.sum(pop.objectives_raw)}, fronts {np.sum(pop.fronts)}, check mutation")
            # a0,b0=np.sum(copy.deepcopy(pop.objectives_raw)),np.sum(copy.deepcopy(pop.fronts))
            pop.crowding_dist=AlgNsga2._crowding_distance(pop.objectives_raw,pop.fronts,self.big_dummy)
            # a1,b1=np.sum(pop.objectives_raw),np.sum(pop.fronts)
            # if ((a1-a0)!=0)|((b1-b0)!=0):
            #     raise Exception('Mutation is affecting values, consider making a deepcopy.')

            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

            # 16) Linear Reinsertion

            # 16.1) Selects indexes to maintain
            # Calculates number of violated constraints
            violations=self.calc_violations(pop)
            ix_reinsert=AlgNsga2._index_linear_reinsertion_nsga_constraints(violations,pop.crowding_dist,pop.fronts,num_chromossomes)
            # 16.2) Remove non reinserted chromossomes from pop
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")
            ix_reinsert_copy=ix_reinsert.copy()
            violations=violations[ix_reinsert_copy]
            self.select_pop_by_index(pop,ix_reinsert_copy)

            # try:
            #     ix_vio=np.where(violations==0)[0]
            #     print("Number of violations ",len(ix_vio))
            #     ix_par=np.where(pop.fronts==0)[0]
            #     ix_pareto=np.intersect(ix_vio,ix_par)
            #     # ix_pareto=np.where(pop.fronts==0)[0]
            #     print("Objectives",pop.objectives_raw[ix_pareto])
            #     # print("Products",pop.products_raw[ix_pareto])
            #     # print("Batches",pop.batches_raw[ix_pareto])
            #     # print("Masks",pop.masks[ix_pareto])
            #     ix_best_min=np.argmin(pop.objectives_raw[:,0][ix_pareto])
            #     ix_best_max=np.argmax(pop.objectives_raw[:,0][ix_pareto])

            #     print("X batches",pop.batches_raw[ix_pareto][ix_best_min][pop.masks[ix_pareto][ix_best_min]])
            #     print("X Products",pop.products_raw[ix_pareto][ix_best_min][pop.masks[ix_pareto][ix_best_min]])
            #     print("Y batches",pop.batches_raw[ix_pareto][ix_best_max][pop.masks[ix_pareto][ix_best_max]])
            #     print("Y Products",pop.products_raw[ix_pareto][ix_best_max][pop.masks[ix_pareto][ix_best_max]])
            # except:
            #     pass
            # for i in range(0,len(pop.products_raw)):
            #     if any(pop.batches_raw[i][pop.masks[i]]==0):
            #         raise Exception("Invalid number of batches (0).")
            #     if np.sum(pop.masks[i][pop.genes_per_chromo[i]:])>0:
            #         raise Exception("Invalid bool after number of active genes.")

        # metrics=pop.metrics_inversion_minimization(self.ref_point,self.volume_max,self.inversion_val_throughput)

        # root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        # name_var="v_0_pop"
        # # Export Pickle
        # file_name = name_var+"_results.pkl"
        # path = root_path + file_name
        # file_pkl = open(path, "wb")
        # pickle.dump(pop, file_pkl,pickle.HIGHEST_PROTOCOL)
        # file_pkl.close()
        # r_exec,r_ind=pop.metrics_inversion_minimization(self.ref_point,self.volume_max,self.inversion_val_throughput,self.num_fronts,num_exec,name_var)
        r_exec,r_ind=pop.metrics_inversion_violations(self.ref_point,self.volume_max,self.inversion_val_throughput,self.num_fronts,num_exec,name_var,violations)
        return r_exec,r_ind,num_exec

    def run_parallel():
        """Runs with Multiprocessing.
        """
        # Parameters

        # Number of executions
        n_exec=10
        n_exec_ite=range(0,n_exec)

        # Variation 1
        # Number of Chromossomes
        nc=[100]
        # Number of Generations
        ng=[1000]
        # Number of tour
        nt=[2]
        # Crossover Probability
        pcross=[0.9,0.5,0.1]
        # pcross=[0.5]
        # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
        pmut=[(0.04,0.61,0.77,0.47),(0.04,0.80,0.80,0.47),(0.04,0.90,0.80,0.47)]

        # List of variants
        list_vars = list(product(*[nc, ng, nt, pcross,pmut]))

        # Dictionary store results
        results={}
        times=[]
        var=0
        for v_i in list_vars:
            t0=time.perf_counter()
            result_execs={}
            result_ids={}
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # for result in (executor.map(Planning().main,n_exec_ite,[v_i[0]]*n_exec,[v_i[1]]*n_exec,[v_i[2]]*n_exec,[v_i[3]]*n_exec,[v_i[4]]*n_exec)):

                for result_exec,result_id,n_exec in (executor.map(Planning().main,n_exec_ite,[v_i[0]]*n_exec,[v_i[1]]*n_exec,[v_i[2]]*n_exec,[v_i[3]]*n_exec,[v_i[4]]*n_exec)):
                    result_execs[(var,n_exec)]=result_exec
                    result_ids[(var,n_exec)]=result_id
                    print(n_exec)

            tf=time.perf_counter()
            delta_t=tf-t0
            print("Total time ",delta_t)
            times.append([v_i,delta_t])
            var+=1

        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        name_var="v_0"
        # name_var=f"exec{n_exec}_chr{nc}_ger{ng}_tour{nt}_cross{pcross}_mut{pmut}"
        file_name = name_var+"_results.csv"
        path = root_path + file_name
        # print(f"{tempo} tempo/exec{tempo/n_exec}")
        # Export times
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            try:
                writer.writerows(times)
            except:
                writer.writerow(times)

        # Export Pickle
        file_name = name_var+"_exec.pkl"
        path = root_path + file_name
        file_pkl = open(path, "wb")
        pickle.dump(result_execs, file_pkl)
        file_pkl.close()

        file_name = name_var+"_id.pkl"
        path = root_path + file_name
        file_pkl = open(path, "wb")
        pickle.dump(result_ids, file_pkl)
        file_pkl.close()


    def run_cprofile():
        """Runs without multiprocessing.
        """
        num_exec=1
        num_chromossomes=100
        num_geracoes=2
        n_tour=2
        pcross=0.50
        # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
        pmut=(0.04,0.61,0.77,0.47)
        t0=time.perf_counter()

        # results,results_ind,n_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)
        # cProfile.runctx("results,num_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)", globals(), locals())

        pr = cProfile.Profile()
        pr.enable()
        pr.runctx("results,results_ind,n_exec=Planning().main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)", globals(), locals())
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\05_trabalho3\\01_dados\\01_raw\\"
        file_name = "cprofile.txt"
        path = root_path + file_name
        ps.print_stats()
        with open(path, 'w+') as f:
            f.write(s.getvalue())
        tf=time.perf_counter()
        delta_t=tf-t0
        print("Total time ",delta_t)


if __name__=="__main__":
    Planning.run_cprofile()
    # Planning.run_parallel()
    # Saves Monte Carlo Simulations
    # Planning().calc_demand_montecarlo_to_external_file(5000)

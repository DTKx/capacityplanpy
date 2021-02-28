import copy
import numpy as np
from dateutil.relativedelta import *
from pygmo import hypervolume


class Population:
    """Stores population attributes and methods
    """

    # Metrics per backlog deficit
    # 0)Max total months and products, 1)Mean total months and products,
    # 2)Std Dev total months and products, 3)Median total months and products,
    # 4)Min total months and products 5)Sum total months and products
    # 6)Backlog violations
    num_metrics = 7

    def __init__(
        self,
        num_genes,
        num_chromossomes,
        num_products,
        num_objectives,
        # start_date,
        qc_max_months,
        num_months,
    ):
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
            qc_max_months (array): Additional number of months to finish quality control.
            num_months (int): Number of months of planning
        """
        self.name_variation = "-"
        self.num_chromossomes = num_chromossomes
        self.num_genes = num_genes

        # Initializes Batch with 1 batch
        self.batches_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
        self.batches_raw[:, 0] = int(1)

        # Initializes products with random allocation of products
        self.products_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype=int)
        self.products_raw[:, 0] = np.random.randint(low=0, high=num_products, size=num_chromossomes)

        # Initialize Mask of active items with only one gene
        self.masks = np.zeros(shape=(num_chromossomes, num_genes), dtype=bool)
        self.masks[:, 0] = True

        # Initializes a time vector Start (Start of USP) and end (end of DSP) of manufacturing campaign Starting with the first date
        self.start_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype="datetime64[D]")
        # self.start_raw[:,0]=start_date
        self.end_raw = np.zeros(shape=(num_chromossomes, num_genes), dtype="datetime64[D]")

        # Initializes Stock backlog_i [kg]

        # 0)Max total backlog months and products, 1)Mean total backlog months and products,
        # 2)Std Dev total backlog months and products, 3)Median total backlog months and products,
        # 4)Min total backlog months and products 5)Sum total backlog months and products
        # 6)Backlog violations
        self.backlogs = np.zeros(shape=(num_chromossomes, self.num_metrics), dtype=float)

        # Initializes Inventory deficit per month (Objective 1, but with breakdown per month) [kg]
        # 0)Max total months and products, 1)Mean total months and products,
        # 2)Std Dev total months and products, 3)Median total months and products,
        # 4)Min total months and products 5)Sum total months and products
        self.deficit = np.zeros(shape=(num_chromossomes, self.num_metrics - 1), dtype=float)

        # Initializes the objectives throughput_i,deficit_strat_i
        self.objectives_raw = np.zeros(shape=(num_chromossomes, num_objectives), dtype=float)

        # Initializes genes per chromossome (Number of active campaigns per solution)
        self.genes_per_chromo = np.sum(self.masks, axis=1, dtype=int)

        # Initialize 3d array with produced (month,product,individual)
        self.produced_month_product_individual = np.zeros(
            shape=(num_months + qc_max_months, num_products, num_chromossomes)
        )

        # NSGA2
        # Creates an array of fronts and crowding distance
        self.fronts = np.empty(shape=(num_chromossomes, 1), dtype=int)
        self.crowding_dist = np.empty(shape=(num_chromossomes, 1), dtype=int)

    def update_genes_per_chromo(self):
        """ Updates genes per chromossome (Number of active campaigns per solution)
        """
        self.genes_per_chromo = np.sum(self.masks, axis=1, dtype=int)

    def update_new_population(self, new_products, new_batches, new_mask):
        """Updates the values of the new offspring population in the class object.

        Args:
            new_products (Array of ints): Population of product labels
            new_batches (Array of ints): Population of number of batches
            new_mask (Array of booleans): Population of active genes
        """
        # Updates new Batches values
        self.batches_raw = copy.deepcopy(new_batches)

        # Updates new Products
        if isinstance(self.products_raw[0][0], np.int32) == False:
            raise ValueError("Not int")

        self.products_raw = copy.deepcopy(new_products)

        if isinstance(self.products_raw[0][0], np.int32) == False:
            raise ValueError("Not int")

        # Updates Mask of active items with only one gene
        self.masks = copy.deepcopy(new_mask)
        self.update_genes_per_chromo()

    def extract_metrics(self, ix, num_fronts, num_exec, id_solution, name_var, ix_pareto):
        """Extract Metrics

        Args:
            ix (int): Index of the solution to verify metrics

        Returns:
            list: List with the metrics Total throughput [kg] Max total backlog [kg] Mean total backlog [kg] Median total backlog [kg] a Min total backlog [kg] P(total backlog ≤ 0 kg) 
                Max total inventory deficit [kg] Mean total inventory deficit [kg] a Median total inventory deficit [kg] Min total inventory deficit [kg]
        """
        metrics = [num_exec, name_var, id_solution]
        # Total throughput [kg]
        metrics.append(self.objectives_raw[:, 0][ix_pareto][ix])
        # Max total backlog [kg]
        metrics.append(self.backlogs[:, 0][ix_pareto][ix])
        # Mean total backlog [kg] +1stdev
        metrics.append(self.backlogs[:, 1][ix_pareto][ix])
        # Standard Dev
        metrics.append(self.backlogs[:, 2][ix_pareto][ix])
        # Median total backlog [kg]
        metrics.append(self.backlogs[:, 3][ix_pareto][ix])
        # Min total backlog [kg]
        metrics.append(self.backlogs[:, 4][ix_pareto][ix])
        # P(total backlog ≤ 0 kg)
        metrics.append(self.backlogs[:, 5][ix_pareto][ix])
        # DeltaXY (total backlog) [kg]

        # Max total inventory deficit [kg]
        metrics.append(self.deficit[:, 0][ix_pareto][ix])
        # Mean total inventory deficit [kg] +1stdev
        metrics.append(self.deficit[:, 1][ix_pareto][ix])
        # Standard Dev
        metrics.append(self.deficit[:, 2][ix_pareto][ix])
        # Median total inventory deficit [kg]
        metrics.append(self.deficit[:, 3][ix_pareto][ix])
        # Min total inventory deficit [kg]
        metrics.append(self.deficit[:, 4][ix_pareto][ix])
        # Total Deficit
        metrics.append(self.objectives_raw[:, 1][ix_pareto][ix])

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

    def metrics_inversion_violations(
        self,
        ref_point,
        volume_max,
        num_fronts,
        num_exec,
        name_var,
        violations,
    ):
        """Extract the metrics only from the pareto front, inverts the inversion made to convert form maximization to minimization, organizes metrics and data for visualization.

        Returns:
            list: Array with metrics:
                "Hypervolume"
                Solution X "X Total throughput [kg]", "X Max total backlog [kg]", "X Mean total backlog [kg]", "X Median total backlog [kg]","X Min total backlog [kg]", "X P(total backlog ≤ 0 kg)","X Max total inventory deficit [kg]", "X Mean total inventory deficit [kg]", "X Median total inventory deficit [kg]", "X Min total inventory deficit [kg]" 
                Solution Y "Y Total throughput [kg]", "Y Max total backlog [kg]", "Y Mean total backlog [kg]", "Y Median total backlog [kg]","Y Min total backlog [kg]", "Y P(total backlog ≤ 0 kg)","Y Max total inventory deficit [kg]", "Y Mean total inventory deficit [kg]", "Y Median total inventory deficit [kg]", "Y Min total inventory deficit [kg]" Pareto Front
        """
        # Indexes
        try:
            ix_vio = np.where(violations == 0)[0]
            ix_par = np.where(self.fronts == 0)[0]
            ix_pareto = np.intersect(ix_vio, ix_par)
        except:
            ix_pareto = np.where(self.fronts == 0)[0]

        # Calculates hypervolume
        try:
            hv = hypervolume(points=self.objectives_raw[ix_pareto])
            hv_vol_norma = hv.compute(ref_point) / volume_max
        except ValueError:
            hv_vol_norma = 0
        metrics_exec = [num_exec, name_var, hv_vol_norma]
        # data_plot=[]

        # Reinverts again the throughput, that was modified for minimization
        self.objectives_raw[:, 0] =self.objectives_raw[:, 0]*(-1.0)
        # Metrics
        ix_best_min = np.argmin(self.objectives_raw[:, 0][ix_pareto])
        ix_best_max = np.argmax(self.objectives_raw[:, 0][ix_pareto])

        metrics_id = [
            self.extract_metrics(ix_best_min, num_fronts, num_exec, "X", name_var, ix_pareto)
        ]
        metrics_id.append(
            self.extract_metrics(ix_best_max, num_fronts, num_exec, "Y", name_var, ix_pareto)
        )

        # Plot Data
        metrics_exec.append(self.objectives_raw[ix_pareto])
        return metrics_exec, metrics_id

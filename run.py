import concurrent.futures
import cProfile
import csv
import datetime
import io
import pickle
from time import perf_counter
from itertools import product
from pstats import SortKey,Stats
import numpy as np
# from scipy import stats
# import tracemalloc
import os
from capacityplanpy import planning,population
from capacityplanpy.genetic import AlgNsga2

# from pygmo.util import hypervolume

def export_list_to_csv(mylist,filepath,methodexport="a"):
    """Export a list to csv, receives an argument for mode of export

    Args:
        mylist (list): List object to be exported
        filepath (str): File path to export
        methodexport (str, optional): Method to export, "a"= appends, "w", please refer to python open docs for further information. Defaults to "a".
    """
    with open(filepath,methodexport, newline="") as f:
        writer = csv.writer(f)
        try:
            writer.writerows(mylist)
        except:
            writer.writerow(mylist)

def export_to_txt(filetogetvalues,path):
    """Export gettable file to export_to_txt

    Args:
        filetogetvalues ([type]): File to export
        path (str): Path to file
    """
    with open(path, "w+") as f:
        f.write(filetogetvalues.getvalue())

def run_parallel(numExec, numGenerations, maxWorkers):
    """Run GA using Multiprocessing.

    Args:
        numExec (int): Number of executions
        numGenerations (int): Number of generations
        maxWorkers (int): Number of workers
    """

    def export_obj(obj, path):
        """Export object to pickle file.

        Args:
            obj (object): Object to be exported.
            path (String): Path for the object to be exported.
        """
        with open(path, "wb") as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_obj(path):
        """Load pickle file to object.

        Args:
            path (String): Path to pickle object.

        Returns:
            object: Object from pickle file.
        """
        with open(path, "rb") as input:
            obj = pickle.load(input)
        return obj

    # Parameters
    n_exec_ite = range(0, numExec)

    # Number of genes
    num_genes = int(25)
    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    qc_max_months = 4  # Max number of months
    # Number of Months
    num_months = 36
    num_fronts = 3  # Number of fronts created

    # Inversion val to convert maximization of throughput to minimization, using a value a little bit higher than the article max 630.4
    ref_point = [2500, 2500]
    volume_max = np.prod(ref_point)  # Maximum Volume

    # Variables
    # Variant
    var = "front_nsga,tour_vio,rein_vio,vio_back,calc_montecarlo"

    # Number of Chromossomes
    nc = [100]
    ng = [numGenerations]  # Number of Generations

    # Number of tour
    nt = [2]
    # Crossover Probability
    # pcross = [0.11]
    pcross = [0.5]
    # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
    pmut = [(0.04, 0.61, 0.77, 0.47)]

    output_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/output_raw/"))
    if os.path.exists(output_data_path)==False:
        raise Exception(f"Could not find the path {output_data_path}, please modify the path.")
    
    # List of variants
    list_vars = list(product(*[nc, ng, nt, pcross, pmut]))

    # Lists store results
    result_execs = []
    result_ids = []
    times = []
    for v_i in list_vars:
        name_var = f"{var},{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}"
        # Creates a dummy pop with one chromossome to concatenate results
        pop_main = population.Population(
            num_genes, 1, num_products, num_objectives, start_date, qc_max_months, num_months,
        )
        pop_main.name_variation = name_var
        file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
        export_obj(pop_main, output_data_path + file_name)
        del pop_main  # 1)Is it a best practice delete an object after exporting and then load, export and del again?

        t0 = perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            for pop_exec in executor.map(
                planning.Planning(
                    num_genes,
                    num_products,
                    num_objectives,
                    start_date,
                    qc_max_months,
                    num_months,
                    num_fronts,
                ).main,
                n_exec_ite,
                [v_i[0]] * numExec,
                [v_i[1]] * numExec,
                [v_i[2]] * numExec,
                [v_i[3]] * numExec,
                [v_i[4]] * numExec,
            ):
                # print("In merge pop exec", pop_exec.fronts)
                # print("Backlog In merge", pop_exec.backlogs[:, 6])

                pop_main = load_obj(output_data_path + file_name)
                # print("In merge pop main", pop_main.fronts)
                print("In merge num_chromossomes", pop_main.num_chromossomes)
                pop_main = planning.Planning.merge_pop_with_offspring(pop_main, pop_exec)
                # print("Out merge pop main", pop_main.fronts)
                # print("Backlog Out merge", pop_main.backlogs[:, 6])
                print("Out merge num_chromossomes", pop_main.num_chromossomes)

                export_obj(pop_main, output_data_path + file_name)
                del pop_main
                del pop_exec

        pop_main = load_obj(output_data_path + file_name)
        # Removes the first dummy one chromossome
        print("Final Num Chromossomes", pop_main.fronts.shape)
        planning.Planning.select_pop_by_index(pop_main, np.arange(1, pop_main.num_chromossomes))
        # Front Classification
        pop_main.fronts = AlgNsga2._fronts(pop_main.objectives_raw, num_fronts)
        print("fronts out", pop_main.fronts)
        # Select only front 0 with no violations or front 0
        ix_vio = np.where(pop_main.backlogs[:, 6] == 0)[0]
        ix_par = np.where(pop_main.fronts == 0)[0]
        ix_pareto_novio = np.intersect1d(ix_vio, ix_par)
        if len(ix_pareto_novio) > 0:
            var = var + "metrics_front0_wo_vio"
            print(
                "Found Solutions without violations and in pareto front",
                len(ix_pareto_novio),
                ix_pareto_novio,
            )
        else:
            print(
                "No solution without violations and in front 0, passing all in front 0.",
                ix_pareto_novio,
            )
            var = var + "metrics_front0_w_vio"
            ix_pareto_novio = ix_par
        print("Fronts In select by index", pop_main.fronts)
        print("Backlog In select by index", pop_main.backlogs[:, 6])
        planning.Planning.select_pop_by_index(pop_main, ix_pareto_novio)
        print("Fronts out select by index", pop_main.fronts)
        print("Backlog out select by index", pop_main.backlogs[:, 6])
        print("Objectives before metrics_inversion_violations", pop_main.objectives_raw)

        # Extract Metrics

        r_exec, r_ind = pop_main.metrics_inversion_violations(
            ref_point, volume_max, num_fronts, 0, name_var, pop_main.backlogs[:, 6],
        )

        result_execs.append(r_exec)
        result_ids.append(r_ind[0])  # X
        result_ids.append(r_ind[1])  # Y
        print("Objectives after metrics_inversion_violations", pop_main.objectives_raw)
        print("Backlog Out after metrics_inversion_violations", pop_main.backlogs[:, 6])

        file_name = f"pop_{v_i[0]},{v_i[1]},{v_i[2]},{v_i[3]},{v_i[4]}.pkl"
        export_obj(pop_main, output_data_path + file_name)

        tf = perf_counter()
        delta_t = tf - t0
        print("Total time ", delta_t, "Per execution", delta_t / numExec)
        times.append([v_i, delta_t, delta_t / numExec])
    name_var = "v_0"
    # name_var=f"exec{numExec}_chr{nc}_ger{ng}_tour{nt}_cross{pcross}_mut{pmut}"
    # print(f"{tempo} tempo/exec{tempo/numExec}")
    # Export times
    export_list_to_csv(times,os.path.join(output_data_path,name_var + "_results.csv"),methodexport="a")

    # Export Pickle
    export_obj(result_execs, os.path.join(output_data_path,name_var + "_exec.pkl"))
    export_obj(result_ids, os.path.join(output_data_path,name_var + "_id.pkl"))

    print("Finish")


def run_cprofile(numExec, numGenerations, maxWorkers):
    """Runs file making a profile of the file.
    """
    # tracemalloc.start()

    # Parameters
    n_exec_ite = range(0, numExec)

    # Number of chromossomes
    num_chromossomes = 100
    # Number of genes
    num_genes = int(25)
    # Number of products
    num_products = int(4)
    # Number of Objectives
    num_objectives = 2
    # Start date of manufacturing
    start_date = datetime.date(2016, 12, 1)  # YYYY-MM-DD.
    qc_max_months = 4  # Max number of months
    # Number of Months
    num_months = 36
    num_fronts = 3  # Number of fronts created

    # Variables

    # Number of tour
    n_tour = 2
    # Crossover Probability
    # pcross = 0.11
    pcross = 0.5
    # Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
    pmut = (0.04, 0.61, 0.77, 0.47)

    t0 = perf_counter()

    pr = cProfile.Profile()
    pr.enable()

    # pr.runctx(
    #     "run_parallel(numExec,numGenerations,maxWorkers)", globals(), locals(),
    # )

    # pr.runctx(
    #     "pop_exec=self.main(num_exec,num_chromossomes,num_geracoes,n_tour,pcross,pmut)",
    #     globals(),
    #     locals(),
    # )

    pr.runctx(
        "for pop_exec in map(planning.Planning(num_genes,num_products,num_objectives,start_date,qc_max_months,num_months,num_fronts).main,n_exec_ite,[num_chromossomes] * numExec,[numGenerations] * numExec,[n_tour] * numExec,[pcross] * numExec,[pmut] * numExec):print()",
        globals(),
        locals(),
    )

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = Stats(pr, stream=s).sort_stats("tottime")
    output_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/output_raw/"))
    if os.path.exists(output_data_path)==False:
        raise Exception(f"Could not find the path {output_data_path}, please modify the path.")

    ps.print_stats()
    export_to_txt(s,os.path.join(output_data_path,"cprofile.txt"))
    tf = perf_counter()
    delta_t = tf - t0
    print("Total time ", delta_t)
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

if __name__ == "__main__":
    # planning.Planning().run_cprofile()
    numExec = 1  # Number of executions
    numGenerations = 1  # Number of executions
    maxWorkers = 1  # Local parallelization Maximum number of threads
    run_parallel(numExec, numGenerations, maxWorkers)
    # run_cprofile(numExec,numGenerations,maxWorkers)

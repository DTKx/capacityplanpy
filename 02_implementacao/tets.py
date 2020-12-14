import concurrent.futures
import multiprocessing
from itertools import product

class Test:
    def print_var(self,n_exec,var):
        return n_exec,var,0

    def run():
        # Parameters

        # Number of executions
        n_exec=1
        n_exec_ite=range(0,n_exec)

        # Variation 1
        # Number of Chromossomes
        nc=[1,2]
        # Number of Generations
        ng=[2,1]
        # Number of tour
        nt=[2,5,7,1]

        # List of variants
        list_vars = list(product(*[nc, ng, nt]))
        result_execs={}
        for v_i in list_vars:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Executes the function n_exec times
                for n_exec,var,a in (executor.map(Test().print_var,[n_exec]*n_exec,[v_i]*n_exec)):
                    result_execs[(n_exec,var)]=a
        print(len(result_execs))

if __name__=="__main__":
    Test.run()

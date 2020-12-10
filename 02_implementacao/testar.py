from itertools import product
import concurrent.futures
import multiprocessing

def printar(n):
    a={'key':1}
    b="Hey"
    return a,b

def main():
    n_exec_ite=range(0,10)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for a,b in executor.map(printar,n_exec_ite):
            print(a)
            print(b)

if __name__=="__main__":
    main()

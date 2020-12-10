from itertools import product

# Number of executions
n_exec=200
n_exec_ite=range(0,n_exec)

# Number of Chromossomes
nc=[100]
# Number of Generations
ng=[200]
# Number of tour
nt=[2]
# Crossover Probability
pcross=[0.6]
# Parameters for the mutation operator (pmutp,pposb,pnegb,pswap)
pmut=[(0.04,0.61,0.77,0.47)]

# List of variants
list_vars = list(product(*[nc, ng, nt, pcross,pmut]))
for v_i in list_vars:
    for i in range(0,len(v_i)):
        v_i[i]=[v_i[i]]*n_exec
    print(v_i)

import numpy as np
import random
import copy
import numba as nb
from numba import jit,prange
from scipy.stats import rankdata
import logging
import os
from errors import CountError, InvalidValuesError

LOG_FILENAME = 'genetic.log'
filepath= os.path.join(os.path.dirname(os.path.realpath(__file__)),LOG_FILENAME)
logging.basicConfig(filename=filepath,filemode="w", level=logging.DEBUG)#Defines the path and level of log file

class Helpers:
    @jit(nopython=True, nogil=True)
    def _is_in(array_2d_to_search, array_1d_search):
        """Equivalent to np.isin

        Args:
            array_2d_to_search ([type]): [description]
            array_1d_search ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.array([x in set(array_1d_search) for x in array_2d_to_search])

    @jit(nopython=True, nogil=True)
    def _find_idx_duplicates(array_duplicates):
        """Returns indexes of first duplicate found 

        Args:
            list_duplicates ([type]): [description]

        Returns:
            [type]: [description]
        """
        oc_set = set()
        res = []
        for val in array_duplicates:
            if val not in oc_set:
                oc_set.add(val)
            else:
                res.append(val)
                break
        return np.where(array_duplicates == val)[0], val


class Crossovers:
    """Methods applied for Crossover of genes between selected parents.
    """

    def _crossover_uniform(pop_product, pop_batches, pop_mask, perc_crossover):
        """Performs the uniform crossover with 2 populations, in case one length of one parent is larger, then the last cromossome may be added with the perc_crossover probability to the shorter offspring.

        Args:
            pop_product (array of ints): Product Population
            pop_batches (Array of ints): Batches Population
            pop_mask (Array of booleans): Masks with active cromossomes
            perc_crossover (float): Probability of crossover, ranging from 0 to 1.

        Returns:
            Arrays: Returns the offspring of the product, batches and mask.
        """
        new_product=copy.deepcopy(pop_product)
        new_batches=copy.deepcopy(pop_batches)
        new_mask=copy.deepcopy(pop_mask)
        genes_per_chromo = np.sum(pop_mask, axis=1, dtype=int)
        if any(genes_per_chromo >= 3):  # Check if any fullfills crossover
            for i in range(0, len(new_product), 2):
                if genes_per_chromo[i] >= 3:  # Condition for crossover
                    # Masks
                    mask = np.random.randint(100, size=(1, genes_per_chromo[i]))
                    mask[mask <= perc_crossover * 100] = 1#1==crossover activated
                    mask[mask > perc_crossover * 100] = 0
                    mask_invert = mask ^ 1
                    # Offspring1=
                    new_product[i, 0 : genes_per_chromo[i]] = (
                        pop_product[i, 0 : genes_per_chromo[i]] * mask_invert
                        + pop_product[i + 1, 0 : genes_per_chromo[i]] * mask
                    )
                    new_product[i + 1, 0 : genes_per_chromo[i]] = (
                        pop_product[i + 1, 0 : genes_per_chromo[i]] * mask_invert
                        + pop_product[i, 0 : genes_per_chromo[i]] * mask
                    )

                    new_batches[i, 0 : genes_per_chromo[i]] = (pop_batches[i, 0 : genes_per_chromo[i]] * mask_invert+ pop_batches[i + 1, 0 : genes_per_chromo[i]] * mask)
                    new_batches[i + 1, 0 : genes_per_chromo[i]] = (pop_batches[i + 1, 0 : genes_per_chromo[i]] * mask_invert+ pop_batches[i, 0 : genes_per_chromo[i]] * mask)

                    len_dif = genes_per_chromo[i + 1] - genes_per_chromo[i]
                    if (
                        len_dif > 0
                    ):  # length difference, if true uses the probability to decide whether to add

                        k = genes_per_chromo[i]
                        proba = np.random.rand(1, len_dif)[0]  # [0, 1) 0<=x<1
                        for j in range(0, len_dif):
                            if proba[j] <= perc_crossover:
                                new_product[i, genes_per_chromo[i]] = pop_product[i + 1, k + j]
                                new_batches[i, genes_per_chromo[i]] = pop_batches[i + 1, k + j]
                                new_mask[i, genes_per_chromo[i]] = True
                                genes_per_chromo[i] = genes_per_chromo[i] + 1

        return new_product, new_batches, new_mask


class Mutations:
    """Methods applied for mutation of individuals.
    """

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True)
    def _label_mutation(chromossome, range_max, pmutp):
        """Label Mutation considering a probability of label mutation of pmutp, considering a range of label from 0 to range_max.

        Args:
            chromossome (array of int): Chromossome with labels within range_max
            range_max (array of int): Range maximum of variation for the mutation
            pmutp (float): Probability of mutation per gene ranging from 0 to 1

        Returns:
            [array of int]: Mutated chromossome
        """
        mask = np.random.randint(0, 100, size=chromossome.shape)
        ix_mut = np.where(mask <= pmutp * 100)
        mutations = len(ix_mut)
        if mutations > 0:
            chromossome[ix_mut] = np.random.randint(0, range_max, size=mutations)
        return chromossome

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True)
    def _add_subtract_mutation(batches,products,masks,num_genes, pposb, pnegb):
        """2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
           3. To add a new random gene to the end of the chromosome (un- conditionally).
        Args:
            batches (array of int): batches with labels within range_max
            products (array of int): products with labels within range_max
            masks (array of bool): masks with labels within range_max
            num_genes (int): Number of genes in the chromossome
            pposb (float): Probability of mutation to add per gene ranging from 0 to 1
            pnegb (float): Probability of mutation to subtract per gene ranging from 0 to 1

        Returns:
            [array of int]: Mutated batches
        """
        # Add
        mask_add = np.random.randint(0, 100, size=num_genes)
        ix_mut_add = np.where(mask_add <= pposb * 100)[0]
        if len(ix_mut_add) > 0:
            batches[0 : num_genes][ix_mut_add] = batches[0 : num_genes][ix_mut_add] + 1
        mask_sub = np.random.randint(0, 100, size=num_genes)
        ix_mut_sub = np.where(mask_sub <= pnegb * 100)[0]
        if len(ix_mut_sub) > 0:
            for i in ix_mut_sub:
                if batches[i] > 1:# Subtract only if > 1
                    batches[i] = batches[i] - 1
                elif (batches[i] == 1)&(num_genes>1):#Removes batch if there is more than 1 gene
                    # print(batches)
                    temp_ar = batches[ i + 1 :].copy()
                    batches[i : -1] = temp_ar# Brings the sequence forward and sets the 
                    batches[-1] = 0#last value as 0
                    # print(batches)
                    # print(products)
                    temp_ar = products[i +1 :].copy()#Adjust Products
                    products[i: -1] = temp_ar
                    products[-1] = 0
                    # print(products)
                    # print(masks)
                    masks[num_genes - 1] = False
                    num_genes= num_genes- 1
                    # print(masks)
                    
        return batches,products,masks,num_genes

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True)
    def _swap_mutation(chromossome_atrib0, chromossome_atrib1, pswap):
        """Swaps in mutation two genes position, the cromossome received have two attribute, that means that if swapping occurs (given a pswap probability) both attributes are swapped.

        Args:
            chromossome_atrib0 (array): Atribute 1
            chromossome_atrib1 (array): Atribute 2
            pswap (float): Probability of mutation to subtract per gene ranging from 0 to 1

        Returns:
            [arrays]: Returns both attributes if changed
        """
        genes = len(chromossome_atrib1)
        if genes > 1:
            mutate = random.randint(0, 100)
            if mutate <= pswap * 100:
                ix_change = np.random.choice(np.arange(0, genes), size=2, replace=False)
                # print(chromossome_atrib0)
                chromossome_atrib0[ix_change[0]], chromossome_atrib0[ix_change[1]] = (
                    chromossome_atrib0[ix_change[1]],
                    chromossome_atrib0[ix_change[0]],
                )
                # print(chromossome_atrib0)
                # print(chromossome_atrib1)
                chromossome_atrib1[ix_change[0]], chromossome_atrib1[ix_change[1]] = (
                    chromossome_atrib1[ix_change[1]],
                    chromossome_atrib1[ix_change[0]],
                )
                # print(chromossome_atrib1)
        return chromossome_atrib0, chromossome_atrib1


class AlgNsga2:
    """ Methods for Algorithm NSGA 2 (1. Deb, K. et al.: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. (2002).)
    """

    @jit(nopython=True, nogil=True, fastmath=True)
    def _fronts_violations(objectives_fn, num_fronts, violations):
        """Avalia as fronteiras de pareto para alocação de cada valor do individuo da população. Modified criteria evaluates first the number of violations and then in case draw evaluates the objective functions.

        Args:
            resultado_fn (Array floats): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronts (int): Número de fronteiras
            violations (Array int): Array with the number of violations
        Returns:
            int: Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        """
        row, col = objectives_fn.shape
        # Definição de fronteiras
        ix_falta_classificar = np.arange(0, row)
        fronts = np.zeros(shape=(row, 1), dtype=np.int64)
        # fronts=np.empty(dtype=int,shape=(row,0))
        # Loop por fronts exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j = 0
        existe_n_dominados = True
        while (j < num_fronts - 1) & (existe_n_dominados):
            dominado = np.ones(shape=(row, 1))
            # dominado[ix_falta_classificar]=_ponto_dominado_minimizacao(objectives_fn[ix_falta_classificar])
            resultado_fn = objectives_fn[ix_falta_classificar].copy()

            # Loop por função
            p = resultado_fn.shape[0]
            dominado_fn = np.ones(shape=(p, 1))
            # Loop por ponto a verificar se é dominado
            for i in prange(0, p):
                # # Loop por ponto da população para comparar até verificar se há algum ponto que domina ou varrer todos
                k = 0
                dominado_sum = int(0)
                while (k < p) & (dominado_sum == int(0)):
                    if i == k:
                        k += int(1)
                        continue
                    # 1) Violation Criteria
                    if violations[i] > violations[k]:
                        dominado_count = 1
                    else:
                        # 2)Domiation Criteria
                        # Problema de minimização, verifico se o ponto é dominado (1) ou não dominado (0)
                        ar_distintos = np.where(resultado_fn[k] != resultado_fn[i])

                        # Se são exatamente iguais são não dominados, porque se não posso perder todos os valores duplicados vou ter que filtrar depois
                        if len(ar_distintos) == 0:
                            dominado_count = 0
                        else:
                            dominado_count = int(
                                np.all(
                                    resultado_fn[k][ar_distintos] < resultado_fn[i][ar_distintos]
                                )
                            )
                    dominado_sum += dominado_count
                    k += int(1)
                if dominado_sum == 0:
                    dominado_fn[i] = 0
                else:
                    dominado_fn[i] = 1

            dominado[ix_falta_classificar] = dominado_fn.copy()

            # dominado[ix_falta_classificar]=_ponto_dominado_minimizacao(objectives_fn[ix_falta_classificar])
            ix_nao_dominados = np.where(dominado == 0)[0]
            if len(ix_nao_dominados) == 0:
                existe_n_dominados = False
                continue
            fronts[ix_nao_dominados] = j
            # Creates an array for ix deletion
            ix_to_delete_list = []
            for ix in ix_nao_dominados:
                ix_to_add = np.where(ix_falta_classificar == ix)[0]
                for ix in ix_to_add:
                    ix_to_delete_list.append(ix)
            num_delete = len(ix_to_delete_list)
            ix_to_delete = np.array(num_delete)
            # Removes classified points
            if num_delete > int(0):
                ix_falta_classificar = np.delete(ix_falta_classificar, ix_to_delete)
            j += 1
        # Adiciona todos os outros pontos na última fronteira
        fronts[ix_falta_classificar] = j

        return fronts

    # @jit(nopython=True, nogil=True)
    def _fronts(objectives_fn, num_fronts):
        """ Calculates pareto fronts for each individual (each row) in population.
        Considers a minimization problem.
        Deprecated.

        Args:
            resultado_fn (float): Array of floats shape(m,n) with the n fitnesses   Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronts (int): Number of Fronts to separate 
        Returns:
            int: Array shape (n,) with the pareto fronts classification for each individual.
        """

        @jit(nopython=True, nogil=True)  # Fast_math must be disabled for Unit Testing
        # @jit(nopython=True, nogil=True,fast_math=True)
        def _ponto_dominado_minimizacao(resultado_fn):
            """Defines dominated points, from n objectives (columns)

            Args:
                resultado_fn (array): Each column represents an objective.

            Returns:
                array: Returns dominated points.
            """
            # Loop por função
            row = resultado_fn.shape[0]
            dominado_fn = np.ones(shape=(row,))
            # Loop por ponto a verificar se é dominado
            for i in prange(0, row):
                # # Loop por ponto da população para comparar até verificar se há algum ponto que domina ou varrer todos
                j = 0
                dominado_sum = int(0)
                while (j < row) & (dominado_sum == int(0)):
                    if i == j:
                        j += int(1)
                        continue
                    # Problema de minimização, verifico se o ponto é dominado (1) ou não dominado (0)
                    ar_distintos = np.where(resultado_fn[j] != resultado_fn[i])

                    # Se são exatamente iguais são não dominados, porque se não posso perder todos os valores duplicados vou ter que filtrar depois
                    if len(ar_distintos) == 0:
                        dominado = 0
                    else:
                        dominado = int(
                            np.all(resultado_fn[j][ar_distintos] < resultado_fn[i][ar_distintos])
                        )
                    dominado_sum += dominado
                    j += int(1)
                if dominado_sum == 0:
                    dominado_fn[i] = 0
                else:
                    dominado_fn[i] = 1
            return dominado_fn

        row, col = objectives_fn.shape
        # Definição de fronteiras
        ix_falta_classificar = np.arange(0, row)
        fronts = np.empty(dtype=int, shape=(row,))
        list_classified_non_dominated = []
        # Loop por fronts exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j = 0
        existe_dominados = True
        while (j < num_fronts - 1) & (existe_dominados):
            dominado = np.ones(shape=(row,))
            dominado[ix_falta_classificar] = _ponto_dominado_minimizacao(
                objectives_fn[ix_falta_classificar]
            )
            ix_nao_dominados = np.where(dominado == 0)[0]
            list_classified_non_dominated.append(ix_nao_dominados)
            if len(ix_nao_dominados) == 0:
                existe_dominados = False
                continue
            fronts[ix_nao_dominados] = j
            ix_falta_classificar = np.setdiff1d(ix_falta_classificar, ix_nao_dominados)
            j += 1
        fronts[ix_falta_classificar] = j  # Adiciona todos os outros pontos na última fronteira

        if (any(fronts>=num_fronts))|(any(fronts<0)):
            expression=f"(any({fronts>num_fronts}))|(any({fronts<0}))"
            e = "Invalid values of fronts."
            logging.error(
                InvalidValuesError(expression, e), exc_info=True
            )  # Adds Exception to log file
            raise InvalidValuesError(expression, e)  # Raise


        return fronts

    def _crowding_distance(objectives_fn, fronts, big_dummy):
        """Calculates share crowding distance for each individual.

        Args:
            objectives_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            fronts (int): Array shape (n,1) com a classificação das fronts de pareto para cada individuo
            sigmashare (float): Coeficiente de agrupamento Sigma share
        Returns:
            float: Array com shared fitness shape (n,1)
        """
        num_ind, num_obj = objectives_fn.shape
        ranks=np.zeros(shape=(num_ind,num_obj),dtype=int)
        crowd_dist = np.zeros(shape=(num_ind,), dtype=float)# Last Column will remain dummy (0)
        num_fronts = np.unique(fronts)
        fit_obj_max_delta = np.zeros(num_obj,dtype=float)#stores max and min for normalization
        #Populates ranks per objectives and add high values to min and max of fronts and objectives
        for j in range(num_obj):#Objectives
            fit_obj_max_delta[j] = np.max(objectives_fn[:,j]) - np.min(objectives_fn[:,j])# Stores Max and minimum for each objective
            if fit_obj_max_delta[j] == 0.0:#Fix in case only one value
                fit_obj_max_delta[j] = 1.0
            for m in num_fronts:#Fronts
                ix_i_front = np.where(fronts == m)[0]
                ranks[ix_i_front,j]=rankdata(objectives_fn[ix_i_front,j],method="dense")
                mask_borders=(ranks[ix_i_front,j]==1)|(ranks[ix_i_front,j]==np.max(ranks[ix_i_front,j]))
                crowd_dist[ix_i_front[mask_borders]]=big_dummy #Adds a large value to the borders
        for k in range(num_ind):#loop per individual
            crowd_val=0
            for j in range(0, num_obj):#Loop objectives
                if crowd_dist[k]==big_dummy:#If contains any border maintains the high value
                    crowd_val=big_dummy
                    break
                else:
                    ix_rank_before=np.where((ranks[:,j]==ranks[k,j]-1)&(fronts==fronts[k]))[0][0]#Select the first one
                    ix_rank_after=np.where((ranks[:,j]==ranks[k,j]+1)&(fronts==fronts[k]))[0][0]#Select the first one
                    crowd_val+=(objectives_fn[ix_rank_after,j]-objectives_fn[ix_rank_before,j])/fit_obj_max_delta[j]
            crowd_dist[k]=crowd_val
        return crowd_dist

    def _index_linear_reinsertion_nsga(crowd_dist, fronts, n_ind):
        """Returns the indexes of the chromossomes for the reinsertion in the new population, evaluating:
        1)Best front, in case of draw evaluates the next item.
        2)Highest Crowding distance

        Args:
            crowd_dist ([type]): Crowding Distance criteria
            fronts ([type]): Ascendent pareto fronts
            n_ind ([type]): Number of chromossome to select

        Returns:
            [array]: Array com os index selecionados para prosseguir na próxima geração
        """
        ix = np.arange(0, len(fronts))
        front_crowd = np.column_stack((fronts, crowd_dist, ix))
        # Verifica qual a fronteira em que o ultimo individuo selecionado está
        ix_asc_fronts = np.argsort(front_crowd[:, 0], axis=0)
        front_for_crowd = int(front_crowd[:, 0][ix_asc_fronts][n_ind])
        indice_nova_pop = np.ones(shape=(n_ind,), dtype=int) * -1
        # Ix already added
        k = 0
        for i in range(0, front_for_crowd + 1):
            # Considera a ultima fronteira sendo verificada por crowding distance
            if i == front_for_crowd:
                val_front_i = copy.deepcopy(front_crowd)[np.where(front_crowd[:, 0] == i)[0]]
                len_val_front_i = len(val_front_i)
                ix_asc_crowd = np.argsort(val_front_i[:, 1], axis=0)
                sorted_ix_crowd_front = val_front_i[:, 2][ix_asc_crowd]
                indice_nova_pop[k : k + len_val_front_i] = (
                    sorted_ix_crowd_front[len_val_front_i - (n_ind - k) :]
                ).astype(int)
            else:
                val_front_i = copy.deepcopy(front_crowd)[np.where(front_crowd[:, 0] == i)[0]]
                len_val_front_i = len(val_front_i)
                indice_nova_pop[k : k + len_val_front_i] = (val_front_i[:, 2]).astype(int)
                k += len_val_front_i

        if (len(indice_nova_pop)<n_ind)|(len(indice_nova_pop[indice_nova_pop<0])>0):
            expression=f"({len(indice_nova_pop)}<{n_ind})|({len(indice_nova_pop[indice_nova_pop<0])}>0)"
            e=f"Error in index selection, number of selected index vs. expected."
            logging.error(CountError(expression,e),exc_info=True)#Adds Exception to log file
            raise CountError(expression,e)#Raise
        return indice_nova_pop

    def _index_linear_reinsertion_nsga_constraints(violations, crowd_dist, fronts, n_ind):
        """Returns the indexes of the chromossomes for the reinsertion in the new population, evaluating:
        1)Lowest number of violations.
        2)Best front, in case of draw evaluates the next item.
        3)Highest Crowding distance

        Args:
            violations (array): Number of violations criteria
            crowd_dist (array): Crowding Distance criteria
            fronts (array): Ascendent pareto fronts
            n_ind (array): Number of chromossome to select

        Returns:
            [array]: Array with selected indexes for next generation.
        """
        ix_selected=np.zeros(shape=(n_ind,),dtype=np.int32)
        ix = np.arange(0, len(fronts),dtype=np.int32)
        remain=n_ind
        criteria_array = np.column_stack(
            [
                violations,
                fronts,
                crowd_dist,
                ix
            ]
        )
        # 1) Evaluate Violations
        sort_vio = np.argsort(criteria_array[:, 0])#Sort Violations Low->High
        criteria_array = criteria_array[sort_vio]

        last_vio_number = criteria_array[:, 0][n_ind - 1]
        count=np.sum(criteria_array[:, 0] == last_vio_number)
        if  count== 1:  # Verify if there is a draw in the highest number of violations
            ix_selected = criteria_array[0:n_ind, 3].astype(np.int32)
            remain=remain-n_ind
        else:  # 2) Draw, evaluates fronts
            ix_selected_vio=np.where(criteria_array[:,0]<last_vio_number)[0]
            if len(ix_selected_vio)>0:#If num violations<last_vio_number add to selected and remove from criteria
                ix_selected[0:len(ix_selected_vio)] = criteria_array[ix_selected_vio,3]
                remain=remain-len(ix_selected_vio)
                criteria_array=criteria_array[np.setdiff1d(np.arange(0,len(criteria_array)),ix_selected_vio),:]
            #If already selected or num violations>last_vio_number+1 delete values
            ix_keep_vio=np.where(criteria_array[:,0]<last_vio_number+1)[0]
            criteria_array=criteria_array[ix_keep_vio,:]
            
            sort_fronts = np.argsort(criteria_array[:, 1])#Sort per fronts Low->High
            criteria_array = criteria_array[sort_fronts]

            last_front_number=criteria_array[:,1][remain-1]
            count=np.sum(criteria_array[:, 1] == last_front_number)
            if  count== 1:  # Verify if there is a draw in the highest number of fronts
                ix_selected[n_ind-remain:] = criteria_array[:remain, 3]
                remain=remain-remain
            else:  # 2) Draw, evaluates Crowding distance
                ix_selected_vio=np.where(criteria_array[:,1]<last_front_number)[0]
                if len(ix_selected_vio)>0:#If num fronts<last_vio_number add to selected
                    ix_selected[n_ind-remain:n_ind-remain+len(ix_selected_vio)] = criteria_array[ix_selected_vio,3]
                    remain=remain-len(ix_selected_vio)
                    criteria_array=criteria_array[np.setdiff1d(np.arange(0,len(criteria_array)),ix_selected_vio),:]
                #If already selected or num violations>last_vio_number+1 delete values
                ix_keep_vio=np.where(criteria_array[:,1]<last_front_number+1)[0]
                criteria_array=criteria_array[ix_keep_vio,:]
                sort_crowd = np.argsort(criteria_array[:, 2])#Sort per Crowding Distance Low->High
                criteria_array = criteria_array[sort_crowd]
                ix_selected[n_ind-remain:] = criteria_array[(len(criteria_array)-remain):, 3]
                remain=remain-remain
        if  remain!=0:
            expression=f"{remain}!=0"
            e=f"Error in index selection, number of selected index:{remain}, Expected: 0"
            logging.error(CountError(expression,e),exc_info=True)#Adds Exception to log file
            raise CountError(expression,e)#Raise
        return ix_selected
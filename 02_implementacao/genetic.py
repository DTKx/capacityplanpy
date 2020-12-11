import numpy as np
import random
import copy
# import numba as nb
from numba import jit
import time
from sklearn.cluster import KMeans
import math
from itertools import combinations 


class Helpers():
    @jit(nopython=True,nogil=True)
    def _is_in(array_2d_to_search,array_1d_search):
        """Equivalent to np.isin

        Args:
            array_2d_to_search ([type]): [description]
            array_1d_search ([type]): [description]

        Returns:
            [type]: [description]
        """
        return np.array([x in set(array_1d_search) for x in array_2d_to_search])

    # @jit(nopython=True,nogil=True)
    # def _hblock(array_1,array_2,array_3):
    #     """Equivalent to np.block

    #     Args:
    #         array_2d_to_search ([type]): [description]
    #         array_1d_search ([type]): [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     ar_1_2=np.column_stack((array_1,array_2))

    #     return np.column_stack((ar_1_2,array_3))
    @jit(nopython=True,nogil=True)
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
        return np.where(array_duplicates==val)[0],val

class InitialPopulation():
    """Methods to generate initial Population for permutation populations.

    Returns:
        [type]: [description]
    """
    # @jit(nopython=True)
    def _gerar_pop_inicial_permutacao(num_pop_total,num_genes,cromossomos_repetidos=True):
        """Gera população inicial para o algoritimo genético, considerando o cromossomo com permutação.

        Args:
            num_pop_total (int): Número de cromossomos na população
            num_genes (int): Número de genes por cromossomo
            cromossomos_repetidos (bool): Aceita populacao com individuos repetidos (True) ou não (False),

        Returns:
            np.array(int): Array com população inicial de cromossomos vide especificações acima
        """
        ind = np.arange(0,num_genes,dtype=int)

        i = 0
        pop_gerada = np.array([],dtype=int)
        while i < num_pop_total:
            # ind = random.sample(space_n, k=10)
            # ind = np.random.permutation(num_genes)
            # ind = np.arange(1,num_genes,dtype=int)
            # np.random.default_rng().shuffle(ind)
            np.random.shuffle(ind)
            # ind = np.delete(ind,np.where(ind==0))
            # pop_gerada.append(ind)
            # ind = np.random.permutation(num_genes).astype(int)
            # ind =np.random.shuffle(np.arange(1,num_genes,dtype=int))
            # ind=np.random.default_rng().permutation(num_genes-1)
            # print(ind)

            pop_gerada=np.append(pop_gerada,ind)
            # print(pop_gerada)
            i = i + 1
        # print(pop_gerada)
        pop_gerada=pop_gerada.reshape(-1,num_genes)
        # print(pop_gerada)
        if cromossomos_repetidos == False:
            # Sorts the unique arrays
            pop_gerada_test = np.unique(pop_gerada, axis=0)

            # pop_gerada = np.unique(pop_gerada, axis=0)
            # print(pop_gerada)
            i = pop_gerada.shape[0]
            if i != num_pop_total:
                # Individuos a gerar para completar individuos únicos
                n_novos_ind=0
                while n_novos_ind<(num_pop_total-i):
                    # ind = np.arange(1,num_genes,dtype=int)
                    np.random.shuffle(ind)
                    np.random.shuffle(ind)
                    # individuo = np.random.permutation(num_genes)
                    if (pop_gerada == ind).all(1).any()==False:            
                        pop_gerada=(np.append(pop_gerada,(ind))).reshape(-1,num_genes)
                        # print(individuo)
                        n_novos_ind+=1
                        # print(pop_gerada)
        return pop_gerada

    # @jit(nopython=True)
    def _gerar_pop_tarefa_proc_permutacao(num_pop_total,num_genes,cromossomos_repetidos=True):
        """Gera população inicial para o algoritimo genético, considerando o cromossomo com permutação.

        Args:
            num_pop_total (int): Número de cromossomos na população
            num_genes (int): Número de genes por cromossomo
            cromossomos_repetidos (bool): Aceita populacao com individuos repetidos (True) ou não (False),

        Returns:
            np.array(int): Array com população inicial de cromossomos vide especificações acima
        """
        # Gerando a populacao de processadores ajustada tal que a primeira tarefa está sempre no processador 0
        pop_processador=(np.random.randint(0,2,size=num_pop_total*(num_genes-1))).reshape(-1,num_genes-1)

        # # Ranking array with index from lowest to highest value
        ind_low_high = np.argsort(pop_processador)

        # Inicia por 1 pois no final eu ajusto
        ind = np.arange(1,num_genes,dtype=int)

        pop_tarefa = np.empty(shape=(num_pop_total,num_genes-1),dtype=int)
        ix_inicio_p1 = np.empty(shape=(num_pop_total,num_genes),dtype=int)
        for i in range(0,num_pop_total):
            np.random.shuffle(ind)
            pop_tarefa[i]=np.copy(ind)
            # ix_inicio_p1 é o Vetor de tarefas + o último valor é o index da posicao em que comeca o processador 1
            ix_inicio_p1[i,:-1]=np.copy(ind)[ind_low_high[i]]
            try:
                ix_inicio_p1[i,-1]=np.where(pop_processador[i][ind_low_high[i]]==0)[0][-1]
            except IndexError:
                ix_inicio_p1[i,-1]=0
        # Sorts the unique arrays
        pop_tare_proc_uniq,ix_tare_proc= np.unique(ix_inicio_p1,return_index=True,axis=0)

        while len(ix_tare_proc)!=num_pop_total:
            # Find index of duplicates
            ix_duplicados=set(range(0,num_pop_total))-set(ix_tare_proc)
            for i in ix_duplicados:
                np.random.shuffle(ind)
                pop_tarefa[i]=np.copy(ind)
                # Vetor de tarefas e a posicao em que comeca o processador 1
                ix_inicio_p1[i,:-1]=np.copy(ind)[ind_low_high[i]]
                try:
                    ix_inicio_p1[i,-1]=np.where(pop_processador[i][ind_low_high[i]]==0)[0][-1]
                except IndexError:
                    ix_inicio_p1[i,-1]=0

            # Sorts the unique arrays
            pop_tare_proc_uniq,ix_tare_proc= np.unique(ix_inicio_p1,return_index=True,axis=0)

        # Corrige a populacao de tarefas tal que a primeira tarefa é sempre a 0 
        # pop_tarefa=np.append(np.zeros(shape=(num_pop_total,1),dtype=int),pop_tarefa[:,:-1],axis=1)

        pop_tarefa=np.append(np.zeros(shape=(num_pop_total,1),dtype=int),pop_tarefa,axis=1)
        pop_processador=np.append(np.zeros(shape=(num_pop_total,1),dtype=int),pop_processador,axis=1)
        return pop_tarefa,pop_processador

class SelectionCrossover():
    """Methods applied to select individuals for Crossover.
    """
    def _roleta_simples_normalizada(pesos, n_ind_selecionar):
            """Seleciona individuos utilizando o método da roleta a partir dos pesos que serão normalizados 

            Args:
                pesos (float): Pesos de cada individuo
                n_ind_selecionar (int): Número de individuos a selecionar

            Returns:
                int: Array com indices dos selecionados
            """
            import bisect

            # Criando array com peso acumulados normalizados
            pesos_acumulado = list(np.cumsum(pesos/sum(pesos)))

            # Selecionando um número randomico
            vals_sorteados = [random.random() for x in range(0,n_ind_selecionar)]

            index_sorteados = []
            # Verificando index sorteado, primeiro valor maior que o corte
            for valor_sorteado in vals_sorteados:
                i_sorteado = bisect.bisect(pesos_acumulado,valor_sorteado)
                # print(pesos[i_sorteado-1],pesos_acumulado[i_sorteado-1])
                # print(valor_sorteado)
                # print(pesos[i_sorteado],pesos_acumulado[i_sorteado])
                index_sorteados.append(i_sorteado)

            return np.array(index_sorteados)

    def _torneio_simples(populacao,pesos, n_ind_selecionar,n_tour,tipo_maximizacao_minimizacao,permite_repeticao_pais):
        """Seleciona n_ind_selecionar individuos da população utilizando o torneio simples:
        1)Realiza se a seleção aleatória sem pesos de n_tour individuos
        2)Seleciona se o individuo com a maior aptidão dos 3 individuos
        3)Se permite_repeticao_pais sem_repeticao_pais evita se possivel a repetição de pais, com_repeticao_pais permite a repeticção
        4)Retorna se todos os individuos mesmo o selecionado à população


        Args:
            pesos (int): Valores de aptidão
            n_ind_selecionar (int): número de vetores a selecionar da população
            n_tour (int): Número de individuos a selecionar para a comparação de aptidão

        Returns:
            array: array com a população escolhida
        """
        # Indexes de individuos da população e pesos de aptidao
        indices_apt=np.arange(0,pesos.shape[0])    
        
        # # Mascara para armazenar index de valores selecionados
        # mask_idx_nao_selecionados=np.ones(populacao.shape,dtype=bool)
        idx_ganhadores=[]

        if tipo_maximizacao_minimizacao=="maximizacao":
            # Seleção do primeiro index
            # Seleção de indices de individuos para participação do torneio.
            idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
            # Retorna a linha com a maior aptidao
            idx_ganhadores.append(idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])])

            # Loop de torneios de seleção de individuos iniciando a partir do primeiro número
            i=1
            while i<n_ind_selecionar:
                # Seleção de indices de individuos para participação do torneio.
                idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
                # Retorna a linha com a maior aptidao
                # (populacao_tarefas[mask_idx_selecionados[i]]==populacao_tarefas[mask_idx_selecionados[i+1]]).all()
                idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                if (permite_repeticao_pais=="sem_repeticao_pais"):
                    if ((populacao[idx_ganhador]==populacao[idx_ganhadores[i-1]]).all()) & (i%2==0):
                        try:
                            distinto_ix=np.where((populacao!=populacao[idx_ganhador]).any(axis=1))[0]
                            # Seleção de indices de individuos para participação do torneio.
                            idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour,replace=True)
                            # Retorna a linha com a maior aptidao
                            idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                        except Exception as e:
                            print(e)

                idx_ganhadores.append(idx_ganhador)
                i=i+1
        else:
            # Seleção do primeiro index
            # Seleção de indices de individuos para participação do torneio.
            idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
            # Retorna a linha com a maior aptidao
            idx_ganhadores.append(idx_sorteados_torneio[np.argmin(pesos[idx_sorteados_torneio])])

            # Loop de torneios de seleção de individuos iniciando a partir do primeiro número
            i=1
            while i<n_ind_selecionar:
                # Seleção de indices de individuos para participação do torneio.
                idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
                # Retorna a linha com a maior aptidao
                idx_ganhador = idx_sorteados_torneio[np.argmin(pesos[idx_sorteados_torneio])]
                if (permite_repeticao_pais=="sem_repeticao_pais"):
                    if ((populacao[idx_ganhador]==populacao[idx_ganhadores[i-1]]).all()) & (i%2==0):
                        try:
                            # Index valores distintos
                            distinto_ix=np.where((populacao!=populacao[idx_ganhador]).any(axis=1))[0]
                            # Seleção de indices de individuos para participação do torneio.
                            idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour,replace=True)
                            idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                        except Exception as e:
                            print(e)

                # Retorna a linha com a maior aptidao
                # (populacao_tarefas[mask_idx_selecionados[i]]==populacao_tarefas[mask_idx_selecionados[i+1]]).all()
                idx_ganhador = idx_sorteados_torneio[np.argmin(pesos[idx_sorteados_torneio])]
                if (permite_repeticao_pais=="sem_repeticao_pais"):
                    if (i%2==0):
                        if (populacao[idx_ganhador]==populacao[idx_ganhadores[i-1]]).all():
                            try:
                                # Index valores distintos
                                distinto_ix=np.where((populacao!=populacao[idx_ganhador]).any(axis=1))[0]

                                # Seleção de indices de individuos para participação do torneio.
                                idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour,replace=True)
                                idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                            except Exception as e:
                                print(e)

                idx_ganhadores.append(idx_ganhador)
                i=i+1
        return idx_ganhadores

    def _torneio_simples_2_pops(pop_tarefa,pop_processador,pesos, n_ind_selecionar,n_tour,tipo_maximizacao_minimizacao):
        """Seleciona n_ind_selecionar individuos da população utilizando o torneio simples:
        1)Realiza se a seleção aleatória sem pesos de n_tour individuos
        2)Seleciona se o individuo com a maior aptidão dos 3 individuos
        3)Evita se possivel a repetição de pais
        4)Retorna se todos os individuos mesmo o selecionado à população


        Args:
            pesos (int): Valores de aptidão
            n_ind_selecionar (int): número de vetores a selecionar da população
            n_tour (int): Número de individuos a selecionar para a comparação de aptidão

        Returns:
            array: array com a população escolhida
        """
        # Indexes de individuos da população e pesos de aptidao
        indices_apt=np.arange(0,pesos.shape[0])    

        # # Ranking array with index from lowest to highest value
        ind_low_high = np.argsort(pop_processador)
        
        # # Mascara para armazenar index de valores selecionados
        # mask_idx_nao_selecionados=np.ones(pop_tarefa.shape,dtype=bool)
        idx_ganhadores=[]

        if tipo_maximizacao_minimizacao=="maximizacao":
            # Seleção do primeiro index
            # Seleção de indices de individuos para participação do torneio.
            idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
            # Retorna a linha com a maior aptidao
            idx_ganhadores.append(idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])])

            # Loop de torneios de seleção de individuos iniciando a partir do primeiro número
            for i in range(1,n_ind_selecionar):
                # Seleção de indices de individuos para participação do torneio.
                idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
                # Retorna a linha com a maior/menor aptidao de acordo com o tipo de maximização
                idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                paridade=(i%2==0)
                if paridade:
                    tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                    proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])

                    # while tarefa_duplicada & proc_duplicada:
                    if tarefa_duplicada & proc_duplicada:
                        try:
                            distinto_ix=np.where((pop_tarefa!=pop_tarefa[idx_ganhador]).any(axis=1))[0]
                            # Seleção de indices de individuos para participação do torneio.
                            idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour)
                            # Retorna a linha com a maior aptidao
                            idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                            tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                            proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                        except Exception as e:
                            # print(e," Breaking.")
                            break
                idx_ganhadores.append(idx_ganhador)
                # i=i+1
        else:
            # Seleção do primeiro index
            # Seleção de indices de individuos para participação do torneio.
            idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
            # Retorna a linha com a menor aptidao
            idx_ganhadores.append(idx_sorteados_torneio[np.argmin(pesos[idx_sorteados_torneio])])

            # Loop de torneios de seleção de individuos iniciando a partir do primeiro número
            for i in range(1,n_ind_selecionar):
            # while i<n_ind_selecionar:
                # Seleção de indices de individuos para participação do torneio.
                idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
                # Retorna a linha com a maior/menor aptidao de acordo com o tipo de maximização
                idx_ganhador = idx_sorteados_torneio[np.argmin(pesos[idx_sorteados_torneio])]
                paridade=(i%2==0)
                if paridade:
                    tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                    proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                    if tarefa_duplicada & proc_duplicada:
                        try:
                            distinto_ix=np.where((pop_tarefa!=pop_tarefa[idx_ganhador]).any(axis=1))[0]
                            # Seleção de indices de individuos para participação do torneio.
                            idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour)
                            # Retorna a linha com a maior aptidao
                            idx_ganhador = idx_sorteados_torneio[np.argmax(pesos[idx_sorteados_torneio])]
                            tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                            proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                        except Exception as e:
                            # print(e," Breaking.")
                            break
                idx_ganhadores.append(idx_ganhador)

        return idx_ganhadores

class Crossovers():
    """Methods applied for Crossover of genes between selected parents.
    """

    def _crossover_ciclico(pop_tarefa,pop_processador):
        @jit(nopython=True,nogil=True)
        def _trocar_valores_ciclo(pais,pop_processador,valores_ciclo):
            # Encontrando indexes a trocar de posição
            index_ciclo = np.where(np.array([x in set(valores_ciclo) for x in pais[0]]))[0]
            k=0
            while k<len(valores_ciclo):

                # Troca os valores na tarefa

                pais[0,index_ciclo[k]], pais[1,index_ciclo[k]] = (
                    pais[1,index_ciclo[k]],pais[0,index_ciclo[k]])

                # Troca os valores no processador
                pop_processador[0,index_ciclo[k]], pop_processador[1,index_ciclo[k]] = (
                    pop_processador[1,index_ciclo[k]],pop_processador[0,index_ciclo[k]])

                k=k+1
            return pais,pop_processador

        @jit(nopython=True,nogil=True)
        def _encontrar_val_ciclo(pais, indice_cross_0):
            # Primeiro par a ser trocado
            par = np.copy(pais[:, indice_cross_0])

            # Adicionando o primeiro par ao ciclo
            # ciclo = np.array(par, dtype=np.int32)
            # ciclo=np.empty(shape=(par.shape),dtype=np.int32)
            ciclo=np.copy(par)

            id_prox_par=np.empty(shape=(2,),dtype=np.int32)

            # Próximo par
            # Encontrando as duas localizações do próximo par a trocar
            id_prox_par[0],id_prox_par[1]=(np.where(pais[0] == par[1])[0][0]),(np.where(pais[1] == par[0])[0][0])

            # Encontrando as valores do próximo par a trocar
            par[0],par[1]=(pais[0][id_prox_par[1]]),(pais[1][id_prox_par[0]])
            
            # Verificando se o primeiro par está contido no ciclo
            # Numero de valores do par já contidos no ciclo

            n_in_v0_v1 = np.cumsum(np.array([x in set(pais[:,id_prox_par[0]]) for x in ciclo]))[-1]
            n_in_v1_v0 = np.cumsum(np.array([x in set(pais[:,id_prox_par[1]]) for x in ciclo]))[-1]

            # Verificando se eu fechei o ciclo 3 condições: par0 ou par1 já está no ciclo ou par0=par1
            fechou_ciclo = ((n_in_v0_v1 >= 2)|(n_in_v1_v0 >= 2)|(par[0]==par[1]))

            # Troca dos valores no ciclo
            pais[0][indice_cross_0], pais[1][indice_cross_0] = (pais[1][indice_cross_0],pais[0][indice_cross_0])

            ciclo = np.append(ciclo, par)

            # Conto número de loops para realizar break se ocorrer algum problema no ciclo
            n_break=0

            while fechou_ciclo==False:
                if n_break>=40:
                    print("Erro em encontrar_val_ciclo não fecha o ciclo.")
                else:
                    # id_prox_par,pais,par,ciclo,fechou_ciclo=add_val_ciclo(id_prox_par,pais,par,ciclo,fechou_ciclo)

                    loc_troca_par=np.copy(id_prox_par)

                    # Próximo par
                    # Encontrando as duas localizações do próximo par a trocar
                    id_prox_par[0],id_prox_par[1]=np.where(pais[0] == par[1])[0][0],np.where(pais[1] == par[0])[0][0]

                    # Encontrando as valores do próximo par a trocar
                    par[0],par[1]=pais[0][id_prox_par[1]],pais[1][id_prox_par[0]]
                    
                    # Verificando se o primeiro par está contido no ciclo
                    # Numero de valores do par já contidos no ciclo
                    n_in_v0_v1 = np.cumsum(np.array([x in set(pais[:,id_prox_par[0]]) for x in ciclo]))[-1]
                    n_in_v1_v0 = np.cumsum(np.array([x in set(pais[:,id_prox_par[1]]) for x in ciclo]))[-1]
                    # Verificando se eu fechei o ciclo 3 condições: par0 ou par1 já está no ciclo ou par0=par1
                    fechou_ciclo = ((n_in_v0_v1 >= 2)|(n_in_v1_v0 >= 2)|(par[0]==par[1]))

                    # Troca dos valores no ciclo
                    pais[0][loc_troca_par], pais[1][loc_troca_par] = (pais[1][loc_troca_par],pais[0][loc_troca_par])

                    ciclo = np.append(ciclo, par)

                    n_break+=1

            return np.unique(ciclo)

        num_genes=pop_tarefa.shape[1]
        i = 0
        while i < len(pop_tarefa):
            # Garanto que não fecho o ciclo de primeira, no caso de pares iguais eu saio do loop
            bool_pares_cross_iguais = pop_tarefa[i] == pop_tarefa[i + int(1)]
            # Condição de passar no caso de igualdade se todos os valores forem iguais move se ao próximo loop
            if np.all(bool_pares_cross_iguais):
                i=i+2
                continue
            else:
                # Extraindo a localização dos pares iguais se eu tivesse um posicionamento diferente, logo selecionando apenas os ix distintos
                p1_cross = random.choice(np.arange(0,num_genes)[np.invert(bool_pares_cross_iguais)])

            # Encontrando valores do ciclo a partir do primeiro par a ser trocado
            valores_ciclo = _encontrar_val_ciclo(copy.deepcopy(pop_tarefa[i:i+2]), p1_cross)

            # Trocando os valores das tarefas e da populacao
            pop_tarefa[i:i+2],pop_processador[i:i+2]=_trocar_valores_ciclo(pop_tarefa[i:i+2],pop_processador[i:i+2],valores_ciclo)
            i = i + 2        

        return pop_tarefa,pop_processador

    def _crossover_ponto(pais,index_ponto_crossover):
        """Crossover de um ponto, altera se a sequencia de dois pais a partir do ponto

        Args:
            pais (array): Array com shape=(n_pais,numero_genes)
            index_ponto_crossover (int): Ponto de crossover

        Returns:
            Array: Pais novos
        """
        
        pais_novos=np.concatenate(np.append(pais[0,:][:index_ponto_crossover],pais[1,:][:index_ponto_crossover]),
        np.append(pais[0,:][:index_ponto_crossover],pais[1,:][:index_ponto_crossover]),axis=0)
        return pais_novos

    def _crossover_multiplos_ponto(pais,lista_index_ponto_crossover):
        """Crossover de um ponto, altera se a sequencia de dois pais a partir do ponto

        Args:
            pais (array): Array com shape=(n_pais,numero_genes)
            lista_index_ponto_crossover (int): Lista de pontos de crossover

        Returns:
            Array: Pais novos
        """
        for ponto in lista_index_ponto_crossover:
            pais=_crossover_ponto(pais,ponto)
        return pais

    def _crossover_uniform(pop_product,pop_batches,pop_mask,perc_crossover):
        """Performs the uniform crossover with 2 populations, in case one length of one parent is larger, then the last cromossome may be added with the perc_crossover probability to the shorter offspring.

        Args:
            pop_product (array of ints): Product Population
            pop_batches (Array of ints): Batches Population
            pop_mask (Array of booleans): Masks with active cromossomes
            perc_crossover (float): Probability of crossover, ranging from 0 to 1.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Returns:
            Arrays: Returns the offspring of the product, batches and mask.
        """
        new_product,new_batches,new_mask=copy.deepcopy(pop_product),copy.deepcopy(pop_batches),copy.deepcopy(pop_mask)
        genes_per_chromo=np.sum(new_mask,axis=1,dtype=int)
        if any(genes_per_chromo>=3):
            for i in range(0,len(new_product),2):
                # print("In ",new_batches[i],new_batches[i+1])
                if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
                    raise Exception("Invalid bool after number of active genes.")
                if np.sum(new_mask[i+1,genes_per_chromo[i+1]:])>0:
                    raise Exception("Invalid bool after number of active genes.")
                if any(new_batches[i][new_mask[i]]==0)|any(new_batches[i+1][new_mask[i+1]]==0):
                    raise Exception("Invalid number of batches (0).")

                if genes_per_chromo[i]>=3:
                    # Masks
                    mask=np.random.randint(100,size=(1,genes_per_chromo[i]))
                    mask[mask<=perc_crossover*100]=1
                    mask[mask>perc_crossover*100]=0
                    mask_invert=mask^1
                    # Offspring1=
                    new_product[i,0:genes_per_chromo[i]]=pop_product[i,0:genes_per_chromo[i]]*mask+pop_product[i+1,0:genes_per_chromo[i]]*mask_invert
                    new_product[i+1,0:genes_per_chromo[i]]=pop_product[i+1,0:genes_per_chromo[i]]*mask_invert+pop_product[i,0:genes_per_chromo[i]]*mask

                    # print("Parents",new_batches[i],new_batches[i+1])
                    new_batches[i,0:genes_per_chromo[i]]=new_batches[i,0:genes_per_chromo[i]]*mask+new_batches[i+1,0:genes_per_chromo[i]]*mask_invert
                    new_batches[i+1,0:genes_per_chromo[i]]=new_batches[i+1,0:genes_per_chromo[i]]*mask_invert+new_batches[i,0:genes_per_chromo[i]]*mask
                    # print("Offspring",new_batches[i],new_batches[i+1])

                    # Check length difference of both parents, if true uses the probability to decide whether to add
                    if genes_per_chromo[i]<genes_per_chromo[i+1]:
                        # random integer N such that a <= N <= b high inclusive
                        last_gene=random.randint(0,100)
                        if last_gene<=perc_crossover*100:
                            # print("Parents to add",new_batches[i][new_mask[i]],new_batches[i+1][new_mask[i+1]])
                            # genes_per_chromo[i] and [i-1] because of the difference Index (Starts with 0) and the genes per chromo
                            new_product[i,genes_per_chromo[i]]=new_product[i+1,genes_per_chromo[i+1]-1]
                            new_batches[i,genes_per_chromo[i]]=new_batches[i+1,genes_per_chromo[i+1]-1]
                            new_mask[i,genes_per_chromo[i]]=True
                            genes_per_chromo[i]=genes_per_chromo[i]+1
                            # print("Offspring added",new_batches[i][new_mask[i]],new_batches[i+1][new_mask[i+1]])
                #             if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
                #                 raise Exception("Invalid bool after number of active genes.")
                #             if any(new_batches[i][new_mask[i]]==0)|any(new_batches[i+1][new_mask[i+1]]==0):
                #                 raise Exception("Invalid number of batches (0).")

                # if np.sum(new_mask[i,genes_per_chromo[i]:])>0:
                #     raise Exception("Invalid bool after number of active genes.")

                # if any(new_batches[i][new_mask[i]]==0)|any(new_batches[i+1][new_mask[i+1]]==0):
                #     raise Exception("Invalid number of batches (0).")

        return new_product,new_batches,new_mask

class Mutations():
    """Methods applied for mutation of individuals.
    """
    @staticmethod
    @jit(nopython=True,nogil=True,fastmath=True)
    def _label_mutation(chromossome,range_max,pmutp):
        """Label Mutation considering a probability of label mutation of pmutp, considering a range of label from 0 to range_max.

        Args:
            chromossome (array of int): Chromossome with labels within range_max
            range_max (array of int): Range maximum of variation for the mutation
            pmutp (float): Probability of mutation per gene ranging from 0 to 1

        Returns:
            [array of int]: Mutated chromossome
        """
        mask=np.random.randint(0,100,size=chromossome.shape)
        ix_mut=np.where(mask<=pmutp*100)
        mutations=len(ix_mut)
        if mutations>0:
            chromossome[ix_mut]=np.random.randint(0,range_max,size=mutations)
        return chromossome
    @staticmethod
    @jit(nopython=True,nogil=True,fastmath=True)
    def _add_subtract_mutation(chromossome,pposb,pnegb):
        """2. To increase or decrease the number of batches by one with a rate of pPosB and pNegB , respectively.
           3. To add a new random gene to the end of the chromosome (un- conditionally).
        Args:
            chromossome (array of int): Chromossome with labels within range_max
            pposb (float): Probability of mutation to add per gene ranging from 0 to 1
            pnegb (float): Probability of mutation to subtract per gene ranging from 0 to 1

        Returns:
            [array of int]: Mutated chromossome
        """
        # Add
        mask_add=np.random.randint(0,100,size=chromossome.shape)
        ix_mut_add=np.where(mask_add<=pposb*100)[0]
        if len(ix_mut_add)>0:
            chromossome[ix_mut_add]=chromossome[ix_mut_add]+1

        mask_sub=np.random.randint(0,100,size=chromossome.shape)
        ix_mut_sub=np.where(mask_sub<=pnegb*100)[0]
        if len(ix_mut_sub)>0:
            # Subtract only if higher than 2
            for ix in ix_mut_sub:
                if chromossome[ix]>1:
                    chromossome[ix]=chromossome[ix]-1
        return chromossome

    @staticmethod
    @jit(nopython=True,nogil=True,fastmath=True)
    def _swap_mutation(chromossome_atrib0,chromossome_atrib1,pswap):
        """Swaps in mutation two genes position, the cromossome received have two attribute, that means that if swapping occurs (given a pswap probability) both attributes are swapped.

        Args:
            chromossome_atrib0 (array): Atribute 1
            chromossome_atrib1 (array): Atribute 2
            pswap (float): Probability of mutation to subtract per gene ranging from 0 to 1

        Returns:
            [arrays]: Returns both attributes if changed
        """
        genes=len(chromossome_atrib1)
        if genes>1:
            mutate=random.randint(0,100)
            if mutate<=pswap*100:
                ix_change=np.random.choice(np.arange(0,genes),size=2,replace=False)
                # print(chromossome_atrib0)
                chromossome_atrib0[ix_change[0]],chromossome_atrib0[ix_change[1]]=chromossome_atrib0[ix_change[1]],chromossome_atrib0[ix_change[0]]
                # print(chromossome_atrib0)
                # print(chromossome_atrib1)
                chromossome_atrib1[ix_change[0]],chromossome_atrib1[ix_change[1]]=chromossome_atrib1[ix_change[1]],chromossome_atrib1[ix_change[0]]
                # print(chromossome_atrib1)
        return chromossome_atrib0,chromossome_atrib1


    def _mutacao_2_genes(pop_tarefa,pop_processador, probabilidade_mutacao):
            # Sorteio de  um array com o tamanho da população com um valor para ditar se ocorre ou nao mutacao (Se valor=0 ou 1 ocorre mutacao)
            sorteio_proba_mutacao_pop = np.random.randint(0, high=100, size=len(pop_tarefa), dtype=np.int64)

            # Binarization of values, se 1 então ocorre mutação
            sorteio_proba_mutacao_pop = np.where(sorteio_proba_mutacao_pop > probabilidade_mutacao * 100, 0, 1)

            # Encontrando os index onde haverá mutação
            # ind_mutacao=np.where(sorteio_proba_mutacao_pop==1)
            ind_mutacao = np.array([np.where(sorteio_proba_mutacao_pop == 1)])

            # Troca dos valores da mutação

            if ind_mutacao.size != 0:
                i = 0
                while i < ind_mutacao.size:
                    # Seleção de index para mutação
                    # Sorteando um index para o pai 1 até posição 9
                    index_mutacao = np.random.randint(0, high=10, size=2, dtype=np.int64)

                    # Trocando a populacao de tarefas
                    # print(f"Antes mutação: {pop_tarefa[ind_mutacao[0][0][i]]}")
                    (
                        pop_tarefa[ind_mutacao[0][0][i],index_mutacao[0]],
                        pop_tarefa[ind_mutacao[0][0][i],index_mutacao[1]],
                    ) = (
                        pop_tarefa[ind_mutacao[0][0][i],index_mutacao[1]],
                        pop_tarefa[ind_mutacao[0][0][i],index_mutacao[0]],
                    )
                    # print(f"Depois mutação: {pop_tarefa[ind_mutacao[0][0][i]]}")

                    # Trocando a populacao de processador
                    # print(f"Antes mutação: {pop_processador[ind_mutacao[0][0][i]]}")
                    (
                        pop_processador[ind_mutacao[0][0][i],index_mutacao[0]],
                        pop_processador[ind_mutacao[0][0][i],index_mutacao[1]],
                    ) = (
                        pop_processador[ind_mutacao[0][0][i],index_mutacao[1]],
                        pop_processador[ind_mutacao[0][0][i],index_mutacao[0]],
                    )
                    # print(f"Depois mutação: {pop_processador[ind_mutacao[0][0][i]]}")

                    i = i + 1
            return pop_tarefa,pop_processador

class OffspringReinsertion():
    """Methods applied for reinsertion of new population for the next generation.
    """

    def _index_reinsercao_ordenada(aptidao_populacao,n_individuos_selecionar, tipo_maximizacao_minimizacao):
            """Retorna os index de posição dos valores selecionados para reinserção na nova população
            Utilizando a reinserção ordenada

            Args:
                aptidao_populacao (array): [valores do método de pontuação a ser utilizado]
            """
            # Selecionando a nova população Top 100 maiores valores de apt inversa

            # np.random.shuffle(aptidao_populacao)
            # Ranking array with index from lowest to highest value
            ind_low_high = np.argsort(aptidao_populacao,axis=0)
            # aptidao_populacao[ind_low_high]
            # np.amin(aptidao_populacao[0:100])
            # np.amax(aptidao_populacao[100:])

            # Selecionando maximização
            if tipo_maximizacao_minimizacao == "maximizacao":
                # Selecionando valores máximos
                indice_nova_pop = ind_low_high[-n_individuos_selecionar:]

            elif tipo_maximizacao_minimizacao == "minimizacao":
                # Selecionando valores mínimos
                indice_nova_pop = ind_low_high[:n_individuos_selecionar]

            return indice_nova_pop

    def _evitar_repeticao_pais(mask_idx_selecionados,populacao_tarefas,populacao_processadores):
        # Verifico duplicados
        duplicados=[(populacao_tarefas[mask_idx_selecionados[i]]==populacao_tarefas[mask_idx_selecionados[i+1]]).all() for i in range(0,len(mask_idx_selecionados),2)]
        # Verifica se há duplicados
        # Verifico a localização dos indices
        ix_duplicados=[index*2 for index,value in enumerate(duplicados) if value ==True]
        num_duplicados=len(ix_duplicados)
        # print(num_duplicados)

        # Check número de unique arrays
        uniquevalues,count_unique=np.unique(populacao_tarefas[mask_idx_selecionados],axis=0,return_counts=True)
        sum_unique=sum(count_unique)
        if (num_duplicados>0) & (sum_unique>1):
            # (populacao_tarefas!=populacao_tarefas[ix_duplicados[i]]).any()
            for i in range(0,num_duplicados):
                check=False
                j=0
                try:
                    while check==False:
                        # duplicados=[(populacao_tarefas[mask_idx_selecionados[i]]==populacao_tarefas[mask_idx_selecionados[i+1]]).all() for i in range(0,len(mask_idx_selecionados),2)]
                        # Valores distintos
                        condicao_1=(populacao_tarefas[ix_duplicados[i]]!=populacao_tarefas[j]).any()
                        # Distinto do pai adjacente
                        condicao_2=(populacao_tarefas[ix_duplicados[i]+1]!=populacao_tarefas[j]).any()
                        # Distinto do array de troca adjacente
                        condicao_3=(populacao_tarefas[ix_duplicados[i]]!=populacao_tarefas[j+1]).any()
                        check=(condicao_1 & condicao_2 & condicao_3)
                        j+=2
                    # Change index
                    # a=copy.deepcopy(mask_idx_selecionados)
                    mask_idx_selecionados[ix_duplicados[i]],mask_idx_selecionados[j-2]=mask_idx_selecionados[j-2],mask_idx_selecionados[ix_duplicados[i]]
                    # print([x1 - x2 for (x1, x2) in zip(a, mask_idx_selecionados)])
                    # duplicados_test=[(populacao_tarefas[i]==populacao_tarefas[i+1]).all() for i in range(0,len(populacao_tarefas),2)]
                    # ix_duplicados_test=[index*2 for index,value in enumerate(duplicados_test) if value ==True]
                    # num_duplicados=len(ix_duplicados_test)
                except IndexError as e:
                    print(f"Index {j-2} ",e)
                    pass             
        else:
            pass

        return mask_idx_selecionados

class AlgNsga2():
    """ Methods for Algorithm NSGA 2 (1. Deb, K. et al.: A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. (2002).)
    """
    @jit(nopython=True,nogil=True,fastmath=True)
    def _fronts_numba(objectives_fn,num_fronts):
        """Avalia as fronteiras de pareto para alocação de cada valor do individuo da população.

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronts (int): Número de fronteiras
        Returns:
            int: Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        """
        row,col=objectives_fn.shape
        # Definição de fronteiras
        ix_falta_classificar=np.arange(0,row)
        fronts=np.zeros(shape=(row,1),dtype=np.int64)
        # fronts=np.empty(dtype=int,shape=(row,0))
        # Loop por fronts exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronts-1) & (existe_n_dominados):
            dominado=np.ones(shape=(row,1))
            # dominado[ix_falta_classificar]=_ponto_dominado_minimizacao(objectives_fn[ix_falta_classificar])
            resultado_fn=objectives_fn[ix_falta_classificar].copy()


            # Loop por função
            p=resultado_fn.shape[0]
            dominado_fn=np.ones(shape=(p,1))
            # Loop por ponto a verificar se é dominado
            for i in np.arange(0,p):
                # # Loop por ponto da população para comparar até verificar se há algum ponto que domina ou varrer todos
                k=0
                dominado_sum=int(0)
                while (k<p) & (dominado_sum==int(0)):
                    if i==k:
                        k+=int(1)
                        continue
                    # Problema de minimização, verifico se o ponto é dominado (1) ou não dominado (0)
                    ar_distintos=np.where(resultado_fn[k]!=resultado_fn[i])

                    # Se são exatamente iguais são não dominados, porque se não posso perder todos os valores duplicados vou ter que filtrar depois
                    if len(ar_distintos)==0:
                        dominado_count=0               
                    else:
                        dominado_count=int(np.all(resultado_fn[k][ar_distintos]<resultado_fn[i][ar_distintos]))
                    dominado_sum+=dominado_count
                    k+=int(1)
                if dominado_sum==0:
                    dominado_fn[i]=0
                else:
                    dominado_fn[i]=1

            dominado[ix_falta_classificar]=dominado_fn.copy()

            # dominado[ix_falta_classificar]=_ponto_dominado_minimizacao(objectives_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                existe_n_dominados=False
                continue
            fronts[ix_nao_dominados]=j
            # Creates an array for ix deletion
            ix_to_delete_list=[]
            for ix in ix_nao_dominados:
                ix_to_add=np.where(ix_falta_classificar==ix)[0]
                for ix in ix_to_add:
                    ix_to_delete_list.append(ix)
            num_delete=len(ix_to_delete_list)
            ix_to_delete=np.array(num_delete)
            # Removes classified points
            if num_delete>int(0):
                ix_falta_classificar=np.delete(ix_falta_classificar,ix_to_delete)
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronts[ix_falta_classificar]=j

        return fronts

    def _fronts(objectives_fn,num_fronts):
        """Avalia as fronteiras de pareto para alocação de cada valor do individuo da população.

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronts (int): Número de fronteiras
        Returns:
            int: Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        """
        @jit(nopython=True,nogil=True)
        def _ponto_dominado_minimizacao(resultado_fn):
            """Defines dominated points, from n objectives (columns)

            Args:
                resultado_fn (array): Each column represents an objective.

            Returns:
                array: Returns dominated points.
            """
            # Loop por função
            row=resultado_fn.shape[0]
            dominado_fn=np.ones(shape=(row,))
            # Loop por ponto a verificar se é dominado
            for i in np.arange(0,row):
                # # Loop por ponto da população para comparar até verificar se há algum ponto que domina ou varrer todos
                j=0
                dominado_sum=int(0)
                while (j<row) & (dominado_sum==int(0)):
                    if i==j:
                        j+=int(1)
                        continue
                    # Problema de minimização, verifico se o ponto é dominado (1) ou não dominado (0)
                    ar_distintos=np.where(resultado_fn[j]!=resultado_fn[i])

                    # Se são exatamente iguais são não dominados, porque se não posso perder todos os valores duplicados vou ter que filtrar depois
                    if len(ar_distintos)==0:
                        dominado=0               
                    else:
                        dominado=int(np.all(resultado_fn[j][ar_distintos]<resultado_fn[i][ar_distintos]))
                    dominado_sum+=dominado
                    j+=int(1)
                if dominado_sum==0:
                    dominado_fn[i]=0
                else:
                    dominado_fn[i]=1
            return dominado_fn

        row,col=objectives_fn.shape
        # Definição de fronteiras
        ix_falta_classificar=np.arange(0,row)
        fronts=np.empty(dtype=int,shape=(row,))
        # Loop por fronts exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronts-1) & (existe_n_dominados):
            dominado=np.ones(shape=(row,))
            dominado[ix_falta_classificar]=_ponto_dominado_minimizacao(objectives_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                existe_n_dominados=False
                continue
            fronts[ix_nao_dominados]=j
            ix_falta_classificar=np.delete(ix_falta_classificar,np.where(np.isin(ix_falta_classificar,ix_nao_dominados))[0])
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronts[ix_falta_classificar]=j


        return fronts

    def _crowding_distance(objectives_fn,fronts,big_dummy):
        """Calcula o shared fitness para o algoritmo NSGA para cada individuo

        Args:
            objectives_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            fronts (int): Array shape (n,1) com a classificação das fronts de pareto para cada individuo
            sigmashare (float): Coeficiente de agrupamento Sigma share
        Returns:
            float: Array com shared fitness shape (n,1)
        """
        num_ind,num_obj=objectives_fn.shape

        # A primeira coluna é dummy repleta de 0s
        crowd_dist=np.zeros(shape=(num_ind,num_obj+1),dtype=float)
        crowd_dist[:,0]=0
        num_fronts=np.unique(fronts)
       
        # Loop por fronts i
        for i in num_fronts:
            # Mask de index de individuos na fronteira
            ix_ind_front_i=np.where(fronts==i)[0]

            # Loop por objetivos j
            # A primeira coluna é dummy repleta de 0s
            for j in range(1,num_obj+1):
                fit_obj_max_delta=np.max(objectives_fn[ix_ind_front_i,j-1])-np.min(objectives_fn[ix_ind_front_i,j-1])
                if fit_obj_max_delta==0:
                    fit_obj_max_delta=1
                ix_rank_asc=np.argsort(objectives_fn[ix_ind_front_i,j-1])
                ix_max_cd=np.where(crowd_dist[:,j][ix_ind_front_i]==np.max(crowd_dist[:,j][ix_ind_front_i]))
                ix_min_cd=np.where(crowd_dist[:,j][ix_ind_front_i]==np.min(crowd_dist[:,j][ix_ind_front_i]))
                if len(ix_max_cd)>1:
                    raise ValueError("Mais de um máx")

                crowd_dist[:,j][ix_ind_front_i[ix_rank_asc[ix_max_cd]]]=big_dummy
                crowd_dist[:,j][ix_ind_front_i[ix_rank_asc[ix_min_cd]]]=big_dummy

                # Loop por individuo exceto extremos
                for k in np.arange(len(ix_min_cd),len(ix_rank_asc)-len(ix_min_cd)):
                    # index absoluto
                    ix_abs=ix_ind_front_i[ix_rank_asc[k]]
                    crowd_obj_anterior=crowd_dist[:,j-1][ix_abs]
                    fit_next=objectives_fn[:,j-1][ix_ind_front_i[ix_rank_asc[k+1]]]
                    fit_anterior=objectives_fn[:,j-1][ix_ind_front_i[ix_rank_asc[k-1]]]
                    crowd_dist[:,j][ix_abs]=crowd_obj_anterior+(fit_next-fit_anterior)/fit_obj_max_delta
                    if np.isnan(crowd_dist[:,j][ix_abs]):
                        raise ValueError("Nan")
        return crowd_dist[:,-1]

    def _torneio_simples_nsga2(pop_tarefa,pop_processador,crowd_dist,fronts, n_ind_selecionar,n_tour):
        """Seleciona n_ind_selecionar individuos da população utilizando o torneio simples:
        1)Realiza se a seleção aleatória sem pesos de n_tour individuos
        2)Seleciona se o individuo com a maior aptidão dos 3 individuos
        3)Evita se possivel a repetição de pais
        4)Retorna se todos os individuos mesmo o selecionado à população


        Args:
            pesos (int): Valores de aptidão
            n_ind_selecionar (int): número de vetores a selecionar da população
            n_tour (int): Número de individuos a selecionar para a comparação de aptidão

        Returns:
            array: array com a população escolhida
        """
        # Indexes de individuos da população e pesos de aptidao
        indices_apt=np.arange(0,fronts.shape[0])    

        # # Ranking array with index from lowest to highest value
        ind_low_high = np.argsort(pop_processador)
        
        # # Mascara para armazenar index de valores selecionados
        # mask_idx_nao_selecionados=np.ones(pop_tarefa.shape,dtype=bool)
        idx_ganhadores=[]

        # Torneio para Minimizacao

        # Seleção do primeiro index
        # Seleção de indices de individuos para participação do torneio.
        idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
        # Verifica os ganhadores com as melhores fronts em seguida com as maiores crowding distance
        # fronts dos sorteados
        front_competidores=fronts[idx_sorteados_torneio]
        ix_ganhadores_front=idx_sorteados_torneio[front_competidores==np.min(front_competidores)]
        if len(ix_ganhadores_front)>1:
            crowd_competidores=crowd_dist[ix_ganhadores_front]
            idx_ganhadores.append(ix_ganhadores_front[np.argmax(crowd_competidores)])
        else:
            idx_ganhadores.append(ix_ganhadores_front[0])

        # Loop de torneios de seleção de individuos iniciando a partir do primeiro número
        for i in range(1,n_ind_selecionar):
            # Seleção de indices de individuos para participação do torneio.
            idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=False)

            # # Retorna a linha com a maior/menor aptidao de acordo com o tipo de maximização

            # fronts dos sorteados
            front_competidores=fronts[idx_sorteados_torneio]
            ix_ganhadores_front=idx_sorteados_torneio[front_competidores==np.min(front_competidores)]
            if len(ix_ganhadores_front)>1:
                crowd_competidores=crowd_dist[ix_ganhadores_front]
                idx_ganhador=ix_ganhadores_front[np.argmax(crowd_competidores)]
            else:
                idx_ganhador=ix_ganhadores_front[0]
            # Verifica Duplicados seleciona apenas não duplicados
            paridade=(i%2==0)
            if paridade:
                tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                if tarefa_duplicada & proc_duplicada:
                    try:
                        distinto_ix=np.where((pop_tarefa!=pop_tarefa[idx_ganhador]).any(axis=1))[0]
                        # Seleção de indices de individuos para participação do torneio.
                        idx_sorteados_torneio = np.random.choice(distinto_ix,size=n_tour)
                        # Retorna a linha com a maior aptidao
                        idx_ganhador = idx_sorteados_torneio[np.argmax(fronts[idx_sorteados_torneio])]
                        tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                        proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                    except Exception as e:
                        # print(e," Breaking.")
                        break
            idx_ganhadores.append(idx_ganhador)
        return idx_ganhadores

    def _index_linear_reinsertion_nsga(crowd_dist,fronts,n_ind):
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
        ix=np.arange(0,len(fronts))
        front_crowd=np.column_stack((fronts,crowd_dist,ix))
        # Verifica qual a fronteira em que o ultimo individuo selecionado está
        ix_asc_fronts=np.argsort(front_crowd[:,0],axis=0)
        front_for_crowd=int(front_crowd[:,0][ix_asc_fronts][n_ind])
        indice_nova_pop=np.ones(shape=(n_ind,),dtype=int)*-1
        # Ix already added
        k=0
        for i in range(0,front_for_crowd+1):
            # Considera a ultima fronteira sendo verificada por crowding distance
            if i==front_for_crowd:
                val_front_i=copy.deepcopy(front_crowd)[np.where(front_crowd[:,0]==i)[0]]
                len_val_front_i=len(val_front_i)
                ix_asc_crowd=np.argsort(val_front_i[:,1],axis=0)
                sorted_ix_crowd_front = val_front_i[:,2][ix_asc_crowd]
                indice_nova_pop[k:k+len_val_front_i]=(sorted_ix_crowd_front[len_val_front_i-(n_ind-k):]).astype(int)
            else:
                val_front_i=copy.deepcopy(front_crowd)[np.where(front_crowd[:,0]==i)[0]]
                len_val_front_i=len(val_front_i)
                indice_nova_pop[k:k+len_val_front_i]=(val_front_i[:,2]).astype(int)
                k+=len_val_front_i

        # if (len(indice_nova_pop)<n_ind)|(len(indice_nova_pop[indice_nova_pop<0])>0):
        #     raise Exception("ValueError")

        return indice_nova_pop


class AlgSpea():
    """Métodos para o algoritmo Spea. 
    Ex para importação: 
    import genetico as gn
    Ex uso 
    a=gn.AlgSpea._gerar_populacao(100)
    """   
    def _cluster_knn(ext_objectives,num_clusters):
        mask=range(0,len(ext_objectives))
        kmeans=KMeans(n_clusters=num_clusters, random_state=0).fit(ext_objectives)
        ix_duplicates,val=Helpers._find_idx_duplicates(np.array(kmeans.labels_))
        dif=ext_objectives[ix_duplicates]-kmeans.cluster_centers_[val]
        max_dist,max_ix=0,0
        for i in range(0,len(ix_duplicates)):
            dist = np.linalg.norm(dif[i])
            if dist>max_dist:
                max_ix,max_dist=i,dist                    
        mask=np.delete(mask,ix_duplicates)
        mask=np.append(mask,max_ix)
        return ext_objectives[mask],mask

    # @jit(nopython=True,nogil=True)
    def _ponto_dominado_minimizacao(resultado_fn):
        # Loop por função
        row=resultado_fn.shape[0]
        dominado_fn=np.zeros(shape=(row,))
        # Loop por ponto a verificar se é dominado
        for i in np.arange(0,row):
            # Concept of domination for minimization a dominates b if fia>fib and fja>=fjb
            mask_better_or_equal=(resultado_fn[i]<=resultado_fn).all(axis=1)
            mask_contains_any_dominant_value=(resultado_fn[i]<resultado_fn).any(axis=1)
            mask_distinct=(resultado_fn[i]!=resultado_fn).any(axis=1)
            mask_values_dominated_by_i=mask_better_or_equal&mask_contains_any_dominant_value&mask_distinct
            dominado_fn+=mask_values_dominated_by_i
        dominado_fn[dominado_fn>0]=1
        return dominado_fn

    def _fronteiras(resultado_fn,num_fronteiras):
        """Avalia as fronteiras de pareto para alocação de cada valor do individuo da população.

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronteiras (int): Número de fronteiras
        Returns:
            int: Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        """
        row,col=resultado_fn.shape
        # Definição de fronteiras
        ix_falta_classificar=np.arange(0,row)
        fronteiras=np.empty(dtype=int,shape=(row,))
        # Loop por fronteiras exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronteiras-1) & (existe_n_dominados):
        # for j in range(0,num_fronteiras-1):
            dominado=np.ones(shape=(row,))
            dominado[ix_falta_classificar]=AlgSpea._ponto_dominado_minimizacao(resultado_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                existe_n_dominados=False
                continue
            fronteiras[ix_nao_dominados]=j
            ix_falta_classificar=np.delete(ix_falta_classificar,np.where(np.isin(ix_falta_classificar,ix_nao_dominados))[0])
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronteiras[ix_falta_classificar]=j


        return fronteiras

    def _calc_fitness(resultado_fn,ext_resultado_fn):
        """Calculates fitness according to Zitzler1999, note that the goal is to minimize the fitness, the smaller the fitness higher the better the individual

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            ext_resultado_fn (float): Array floats com os objetivos da população externa calculados

        Returns:
            [array]: Fitness of current population
        """
        num_ind,num_obj=resultado_fn.shape
        # ext_population[:,0:num_obj+1]
        num_ind_ext=len(ext_resultado_fn)

        fitness=np.zeros(shape=(num_ind,),dtype=float)
        ext_fitness=np.zeros(shape=(num_ind_ext,),dtype=float)

        # Loop per individual i of the external population
        for i in range(0,num_ind_ext):
            # Concept of domination for minimization a dominates b if fia>fib and fja>=fjb
            mask_better_or_equal=(ext_resultado_fn[i]<=resultado_fn).all(axis=1)
            mask_contains_any_dominant_value=(ext_resultado_fn[i]<resultado_fn).any(axis=1)
            mask_distinct=(ext_resultado_fn[i]!=resultado_fn).any(axis=1)
            mask_values_dominated_by_i=mask_better_or_equal&mask_contains_any_dominant_value&mask_distinct

            n_dominates=float(len(mask_values_dominated_by_i[mask_values_dominated_by_i==True]))/(num_ind+1.0)
            # Updates dominated sum for external Population 
            fitness=fitness+mask_values_dominated_by_i*n_dominates
            # Updates n_dominated0
            ext_fitness[i]=n_dominates

        fitness+=1
        return fitness,ext_fitness
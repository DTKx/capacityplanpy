import numpy as np
import random
import copy
import numba as nb
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


# Função para geração de população
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

# Função para geração de população
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

@jit(nopython=True,nogil=True)
def _is_in(array_2d_to_search,array_1d_search):
    return np.array([x in set(array_1d_search) for x in array_2d_to_search])

# jit funciona
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


# Crossover ciclico
# @jit(nopython=True,nogil=True)
def _crossover_ciclico(pop_tarefa,pop_processador):
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

        # soma_linha=sum([x for x in range(0,pop_tarefa[i:i+2].shape[1])])
        # verifico_val_duplicados=np.where(np.sum(pop_tarefa[i:i+2],axis=1)!=soma_linha)
        # if verifico_val_duplicados[0].size!=0:
        #     print("Individuos perderam a caracteristica de permutação")

        i = i + 2        

    return pop_tarefa,pop_processador

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
    """Métodos para o algoritmo Nsga. 
    Ex para importação: 
    import genetico as gn
    Ex uso 
    a=gn.AlgNsga2._gerar_populacao(100)
    """

    @jit(nopython=True,nogil=True)
    def _ponto_dominado_minimizacao(resultado_fn):
        # Loop por função
        row=resultado_fn.shape[0]
        dominado_fn=np.ones(shape=(row,))
        # Loop por ponto a verificar se é dominado
        for i in np.arange(0,row):
            # # Loop por ponto da população para comparar até verificar se há algum ponto que domina ou varrer todos
            j=0
            dominado_sum=int(0)
            # dominado=np.any(np.all(resultado_fn[j]>resultado_fn[i],axis=1)).astype(int)
            while (j<row) & (dominado_sum==int(0)):
                if i==j:
                    j+=int(1)
                    continue
                # Problema de minimização, verifico se o ponto é dominado (1) ou não dominado (0)
                ar_distintos=np.where(resultado_fn[j]!=resultado_fn[i])
                # ar_dominado=resultado_fn[j]<resultado_fn[i]
                # ar_dominado_igual=resultado_fn[j]<=resultado_fn[i]
                # ar_sum=sum(ar_dominado)
                # ar_sum_igual=sum(ar_dominado_igual)

                # Se são exatamente iguais são não dominados, porque se não posso perder todos os valores duplicados vou ter que filtrar depois
                if len(ar_distintos)==0:
                    dominado=0               
                # Se não há valores iguais utilizase o all()
                else:
                    dominado=int(np.all(resultado_fn[j][ar_distintos]<resultado_fn[i][ar_distintos]))
                # else:
                #     raise ValueError("Novo caso")
                dominado_sum+=dominado
                j+=int(1)
            if dominado_sum==0:
                dominado_fn[i]=0
            else:
                dominado_fn[i]=1
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
        # mask_nao_classificados=np.ones(dtype=bool,shape=(row,))
        ix_falta_classificar=np.arange(0,row)
        fronteiras=np.empty(dtype=int,shape=(row,))
        # Loop por fronteiras exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronteiras-1) & (existe_n_dominados):
        # for j in range(0,num_fronteiras-1):
            dominado=np.ones(shape=(row,))
            # # Verifica se ainda tenho pontos nao classificados
            # if len(ix_falta_classificar)==0:
            #     break 
            # 0=Não dominados 1=Dominados Loop por pontos
            dominado[ix_falta_classificar]=AlgNsga2._ponto_dominado_minimizacao(resultado_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                # print("Não encontrei não dominados")
                existe_n_dominados=False
                continue
            fronteiras[ix_nao_dominados]=j
            # print(sum([1 for x in ix_falta_classificar]))
            ix_falta_classificar=np.delete(ix_falta_classificar,np.where(np.isin(ix_falta_classificar,ix_nao_dominados))[0])
            # ix_falta_classificar_for_numba_working=np.delete(ix_falta_classificar,np.where(np.array([x in set(ix_nao_dominados) for x in ix_falta_classificar]))[0])
            # print(sum([1 for x in ix_falta_classificar]))
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronteiras[ix_falta_classificar]=j


        return fronteiras

    def _crowding_distance(resultado_fn,fronteiras,big_dummy):
        """Calcula o shared fitness para o algoritmo NSGA para cada individuo

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            fronteiras (int): Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
            sigmashare (float): Coeficiente de agrupamento Sigma share
        Returns:
            float: Array com shared fitness shape (n,1)
        """
        num_ind,num_obj=resultado_fn.shape

        # A primeira coluna é dummy repleta de 0s
        crowd_dist=np.zeros(shape=(num_ind,num_obj+1),dtype=float)
        crowd_dist[:,0]=0
        num_fronteiras=np.unique(fronteiras)
       
        # Loop por fronteiras i
        for i in num_fronteiras:
            # Mask de index de individuos na fronteira
            ix_ind_front_i=np.where(fronteiras==i)[0]

            # Loop por objetivos j
            # A primeira coluna é dummy repleta de 0s
            for j in range(1,num_obj+1):
                fit_obj_max_delta=np.max(resultado_fn[ix_ind_front_i,j-1])-np.min(resultado_fn[ix_ind_front_i,j-1])
                if fit_obj_max_delta==0:
                    fit_obj_max_delta=1
                # print(f"obj{j}")
                ix_rank_asc=np.argsort(resultado_fn[ix_ind_front_i,j-1])

                # # Teste 1 (Funciona) Assigning Crowding distance for extremes first and last
                # crowd_dist[:,j][ix_ind_front_i[ix_rank_asc[[0,-1]]]]=big_dummy

                # Teste 2 Assigning CD para todos os valores que estão nos extremos
                # val_max_cd=np.max(crowd_dist[:,j][ix_ind_front_i])
                # val_min_cd=np.min(crowd_dist[:,j][ix_ind_front_i])
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
                    fit_next=resultado_fn[:,j-1][ix_ind_front_i[ix_rank_asc[k+1]]]
                    fit_anterior=resultado_fn[:,j-1][ix_ind_front_i[ix_rank_asc[k-1]]]
                    # print(f"{crowd_obj_anterior}+({fit_next}-{fit_anterior})/{fit_obj_max_delta}")
                    crowd_dist[:,j][ix_abs]=crowd_obj_anterior+(fit_next-fit_anterior)/fit_obj_max_delta
                    if np.isnan(crowd_dist[:,j][ix_abs]):
                        raise ValueError("Nan")
                    # elif (j==3) & (crowd_dist[:,j][ix_abs]==10000.0):
                    #     raise ValueError("Evaluate")
                    # print(crowd_dist[:,j][ix_abs])
        # print("Crowd")
        return crowd_dist[:,-1]

    def _torneio_simples_nsga2(pop_tarefa,pop_processador,crowd_dist,fronteiras, n_ind_selecionar,n_tour):
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
        indices_apt=np.arange(0,fronteiras.shape[0])    

        # # Ranking array with index from lowest to highest value
        ind_low_high = np.argsort(pop_processador)
        
        # # Mascara para armazenar index de valores selecionados
        # mask_idx_nao_selecionados=np.ones(pop_tarefa.shape,dtype=bool)
        idx_ganhadores=[]

        # Torneio para Minimizacao

        # Seleção do primeiro index
        # Seleção de indices de individuos para participação do torneio.
        idx_sorteados_torneio = np.random.choice(indices_apt,size=n_tour,replace=True)
        # Verifica os ganhadores com as melhores fronteiras em seguida com as maiores crowding distance
        # Fronteiras dos sorteados
        front_competidores=fronteiras[idx_sorteados_torneio]
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

            # Fronteiras dos sorteados
            front_competidores=fronteiras[idx_sorteados_torneio]
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
                        idx_ganhador = idx_sorteados_torneio[np.argmax(fronteiras[idx_sorteados_torneio])]
                        tarefa_duplicada=(pop_tarefa[idx_ganhador][ind_low_high[idx_ganhador]]==pop_tarefa[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]).all()
                        proc_duplicada=(np.where(pop_processador[idx_ganhador][ind_low_high[idx_ganhador]]==0)[0][-1]==np.where(pop_processador[idx_ganhadores[i-1]][ind_low_high[idx_ganhadores[i-1]]]==0)[0][-1])
                    except Exception as e:
                        # print(e," Breaking.")
                        break
            idx_ganhadores.append(idx_ganhador)
        return idx_ganhadores

    def _index_reinsercao_ordenada_nsga(crowd_dist,fronteiras,n_ind):
        """Retorna os index de posição dos valores selecionados para reinserção na nova populaçãoUtilizando a reinserção ordenada, para o NSGA inicialmente as melhores fronteiras são selecionadas e no caso de empate utiliza se a crowding distance como critério de desempate

        Args:
            crowd_dist ([type]): Crowding Distance criterio de desempate em fronteiras
            fronteiras ([type]): Alocação da solução na fronteira de pareto, ascendente menores fronteiras melhores
            n_ind ([type]): Número de individuos que serão selecionados para as próximas gerações.

        Returns:
            [array]: Array com os index selecionados para prosseguir na próxima geração
        """  
        ix=np.arange(0,len(fronteiras))
        front_crowd=np.column_stack((fronteiras,crowd_dist,ix))
        # front_crowd=np.block(fronteiras,crowd_dist,ix)
        # Verifica qual a fronteira em que o ultimo individuo selecionado está
        ix_asc_fronteiras=np.argsort(front_crowd[:,0],axis=0)
        front_for_crowd=int(front_crowd[:,0][ix_asc_fronteiras][n_ind])
        indice_nova_pop=np.ones(shape=(n_ind,),dtype=int)*-1
        # indice_nova_pop=np.empty(shape=(n_ind,))
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
                # indice_nova_pop=np.append(indice_nova_pop,(sorted_ix_crowd_front[:n_ind-k]).astype(int))
                # indice_nova_pop.append(list((sorted_ix_crowd_front[:n_ind]).astype(int)))
            else:
                val_front_i=copy.deepcopy(front_crowd)[np.where(front_crowd[:,0]==i)[0]]
                len_val_front_i=len(val_front_i)
                indice_nova_pop[k:k+len_val_front_i]=(val_front_i[:,2]).astype(int)
                k+=len_val_front_i

        if (len(indice_nova_pop)<n_ind)|(len(indice_nova_pop[indice_nova_pop<0])>0):
            raise ValueError

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
        # f1=np.array([10,8,7,7,13])
        # f2=np.array([5,10,11,12,6])
        # resultado_fn=np.column_stack((f1,f2))
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
        # mask_nao_classificados=np.ones(dtype=bool,shape=(row,))
        ix_falta_classificar=np.arange(0,row)
        fronteiras=np.empty(dtype=int,shape=(row,))
        # Loop por fronteiras exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronteiras-1) & (existe_n_dominados):
        # for j in range(0,num_fronteiras-1):
            dominado=np.ones(shape=(row,))
            # # Verifica se ainda tenho pontos nao classificados
            # if len(ix_falta_classificar)==0:
            #     break 
            # 0=Não dominados 1=Dominados Loop por pontos
            dominado[ix_falta_classificar]=AlgSpea._ponto_dominado_minimizacao(resultado_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                # print("Não encontrei não dominados")
                existe_n_dominados=False
                continue
            fronteiras[ix_nao_dominados]=j
            # print(sum([1 for x in ix_falta_classificar]))
            ix_falta_classificar=np.delete(ix_falta_classificar,np.where(np.isin(ix_falta_classificar,ix_nao_dominados))[0])
            # ix_falta_classificar_for_numba_working=np.delete(ix_falta_classificar,np.where(np.array([x in set(ix_nao_dominados) for x in ix_falta_classificar]))[0])
            # print(sum([1 for x in ix_falta_classificar]))
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronteiras[ix_falta_classificar]=j


        return fronteiras

        # """Calculates fitness according to Zitzler1999, note that the goal is to minimize the fitness, the smaller the fitness higher the better the individual

        # Args:
        #     resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
        #     fronteiras (int): Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        #     sigmashare (float): Coeficiente de agrupamento Sigma share
        # Returns:
        #     float: Array com shared fitness shape (n,1)
        # """

    def _calc_fitness(resultado_fn,ext_resultado_fn):
        """Calculates fitness according to Zitzler1999, note that the goal is to minimize the fitness, the smaller the fitness higher the better the individual

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            ext_resultado_fn (float): Array floats com os objetivos da população externa calculados

        Returns:
            [array]: Fitness of current population
        """
        # f1=np.array([0.9,1,2,2.2,2.5,2.8,1.5])
        # f2=np.array([0.9,3,2.2,0.7,1.8,0.8,2])

        # resultado_fn=np.column_stack((f1,f2))
        # ext_resultado_fn=np.array([[1.8,4],[2.7,2.5],[3,1.5]])

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

            # # Concept of domination for maximization a dominates b if fia>fib and fja>=fjb
            # mask_better_or_equal=(ext_resultado_fn[i]>=resultado_fn).all(axis=1)
            # mask_contains_any_dominant_value=(ext_resultado_fn[i]>resultado_fn).any(axis=1)
            # mask_distinct=(ext_resultado_fn[i]!=resultado_fn).any(axis=1)
            # mask_values_dominated_by_i=mask_better_or_equal&mask_contains_any_dominant_value&mask_distinct

            n_dominates=float(len(mask_values_dominated_by_i[mask_values_dominated_by_i==True]))/(num_ind+1.0)
            # Updates dominated sum for external Population 
            fitness=fitness+mask_values_dominated_by_i*n_dominates
            # Updates n_dominated0
            ext_fitness[i]=n_dominates

        fitness+=1
        return fitness,ext_fitness

class AlgSpea2():
    """Métodos para o algoritmo Spea 2. 
    Ex para importação: 
    import genetico as gn
    Ex uso 
    a=gn.AlgSpea2._gerar_populacao(100)
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

    @jit(nopython=True,nogil=True)
    def _ponto_dominado_minimizacao(resultado_fn):
        # f1=np.array([10,8,7,7,13])
        # f2=np.array([5,10,11,12,6])
        # resultado_fn=np.column_stack((f1,f2))
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

    @jit(nopython=True,nogil=True)
    def density_to_fitness(p_i_to_compare,population_points):
        num_points=len(population_points)
        # K Value
        k=int(math.sqrt(num_points+int(1)))
        distances=np.zeros(shape=population_points[:,0].shape)

        # Loop per points to compare
        for i in np.arange(0,num_points):
            distances[i]=np.linalg.norm(population_points[i]-p_i_to_compare)
        distances=np.sort(distances)
        density_i=1.0/(distances[k]+2.0)
        # distances_truncation=np.copy(distances[k-1:k+1])
        return density_i,np.copy(distances[k-1:k+1])

    def _calc_fitness(resultado_fn_cur_ext):
        """Calculates fitness according to Zitzler2001 (Spea2), note that the goal is to minimize the fitness, the smaller the fitness higher the better the individual

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            ext_resultado_fn (float): Array floats com os objetivos da população externa calculados

        Returns:
            [array]: Fitness of current population
            [array]: Fitness of external population
        """
        # f1=([0.5,1.5,2,2.1,2.8,0.7])
        # f2=([0.5,1.5,0.5,2.1,3,4])
        # resultado_fn=np.column_stack((f1,f2))

        # ext_resultado_fn=np.array([[4.2,0.7],[3,3.1],[1,4.2]])
            
        num_ind_tot=len(resultado_fn_cur_ext)

        mask_values_dominated=np.zeros(shape=(num_ind_tot,),dtype=float)

        num_dominates=np.zeros(shape=(num_ind_tot,),dtype=float)
        density=np.zeros(shape=(num_ind_tot,),dtype=float)
        distances_truncation_cur_ext=np.zeros(shape=(num_ind_tot,2),dtype=float)

        # Loop per individual i of the external population
        for i in range(0,num_ind_tot):
            # Concept of domination for minimization a dominates b if fia>fib and fja>=fjb
            mask_better_or_equal=(resultado_fn_cur_ext[i]<=resultado_fn_cur_ext).all(axis=1)
            mask_contains_any_dominant_value=(resultado_fn_cur_ext[i]<resultado_fn_cur_ext).any(axis=1)
            mask_distinct=(resultado_fn_cur_ext[i]!=resultado_fn_cur_ext).any(axis=1)
            mask_values_dominated_by_i=mask_better_or_equal&mask_contains_any_dominant_value&mask_distinct

            # # Concept of domination for maximization a dominates b if fia>fib and fja>=fjb
            # mask_better_or_equal=(resultado_fn_cur_ext[i]>=resultado_fn_cur_ext).all(axis=1)
            # mask_contains_any_dominant_value=(resultado_fn_cur_ext[i]>resultado_fn_cur_ext).any(axis=1)
            # mask_distinct=(resultado_fn_cur_ext[i]!=resultado_fn_cur_ext).any(axis=1)
            # mask_values_dominated_by_i=mask_better_or_equal&mask_contains_any_dominant_value&mask_distinct

            n_dominates=len(mask_values_dominated_by_i[mask_values_dominated_by_i==True])
            mask_values_dominated=np.column_stack((mask_values_dominated,mask_values_dominated_by_i))        
            num_dominates[i]=n_dominates
            # Calculating distance to all other points
            density_i,distances_truncation_i=AlgSpea2.density_to_fitness(resultado_fn_cur_ext[i],resultado_fn_cur_ext[mask_distinct])
            density[i]=density_i
            distances_truncation_cur_ext[i]=distances_truncation_i

        # Removes first dummy column
        mask_values_dominated=mask_values_dominated[:,1:]
        fitness_cur_ext=np.dot(mask_values_dominated,num_dominates)+density

        return fitness_cur_ext,distances_truncation_cur_ext

    @jit(nopython=True,nogil=True)
    def _distance_to_truncation(p_i_to_compare,population_points):
        num_points=len(population_points)
        # K Value
        k=int(math.sqrt(num_points+int(1)))
        distances=np.zeros(shape=population_points[:,0].shape)

        # Loop per points to compare
        for i in np.arange(0,num_points):
            distances[i]=np.linalg.norm(population_points[i]-p_i_to_compare)
        distances=np.sort(distances)
        return np.copy(distances[k-1:k+1])

    def _truncation(distances_truncation,num_to_select,ext_resultado_fn,ix_trunc):
        n_ind=len(ext_resultado_fn)
        # Creates masks removing the smallest distance 
        mask_not_min_distance=distances_truncation[:,1]!=np.min(distances_truncation[:,1])
        # Verifies if there is tie in smallest distance
        if sum(mask_not_min_distance==False)>1:
            ix_eval_second_distance=np.where(mask_not_min_distance==False)[0]
            min_distance=np.max(distances_truncation[:,0])
            ix_min=0
            for ix in ix_eval_second_distance:
                if distances_truncation[:,0][ix]<min_distance:
                    ix_min=ix
                    min_distance=distances_truncation[:,0][ix]
            mask_not_min_distance[ix_eval_second_distance[ix_eval_second_distance!=ix_min]]=True
            ix_trunc=ix_trunc[mask_not_min_distance]
        else:
            ix_trunc=ix_trunc[mask_not_min_distance]

        while len(ix_trunc)>num_to_select:
            # Updates the distance to truncation removing the 
            for j in range(0,n_ind):
                mask_distinct=(ext_resultado_fn[j]!=ext_resultado_fn).any(axis=1)
                distances_truncation_i=AlgSpea2._distance_to_truncation(ext_resultado_fn[j],ext_resultado_fn[mask_distinct])
                distances_truncation[j]=distances_truncation_i

            # Creates masks removing the smallest distance 
            
            # mask_not_min_distance=distances_truncation[mask_not_min_distance][:,1]!=np.min(distances_truncation[mask_not_min_distance][:,1])
            mask_not_min_distance=distances_truncation[mask_not_min_distance][:,1]!=np.min(distances_truncation[mask_not_min_distance][:,1])
            # Verifies if there is tie in smallest distance
            if sum(mask_not_min_distance==False)>1:
                ix_eval_second_distance=np.where(mask_not_min_distance==False)[0]

                min_distance=np.max(distances_truncation[:,0])
                ix_min=0
                for ix in ix_eval_second_distance:
                    if distances_truncation[mask_not_min_distance][:,0][ix]<min_distance:
                        ix_min=ix
                        min_distance=distances_truncation[mask_not_min_distance][:,0][ix]
                mask_not_min_distance[ix_eval_second_distance[ix_eval_second_distance!=ix_min]]=True
                ix_trunc=ix_trunc[mask_not_min_distance]
            else:
                ix_trunc=ix_trunc[mask_not_min_distance]
        return ix_trunc



class AlgMoead():
    """Métodos para o algoritmo Moead. 
    Ex para importação: 
    import genetico as gn
    Ex uso 
    a=gn.AlgMoead._gerar_populacao(100)
    """
    # lattice=genetico.AlgMoead._simplex_lattice(objectives.shape[1],num_pop_total)

    def _simplex_lattice(num_objectives,num_pop_total):
        # Finding closest number of h
        h=0
        n_comb=0
        while n_comb<num_pop_total:
            h+=1
            n_comb=math.comb(h+num_objectives-1,num_objectives-1)
            print(n_comb)
        print(h)
        
        a=[x/h for x in range(0,num_pop_total)]
        print(a)
        lattice=combinations(5,2)
        for lat in lattice:
            print(lat)
        lattice=combinations(a,num_objectives)
        for lat in lattice:
            print(lat)

        return lattice

    def _random_lattice(num_objectives,num_pop_total):
        # div=int(num_pop_total/10)
        # v_base=np.zeros(shape=(10,3))
        # v_base[:,0]=np.linspace(0,1,num=10,endpoint=True)
        # v_max=1-v_base[:,0]
        # v_base[:,1]=np.random.uniform(0,v_max)
        # v_base[:,2]=1-np.sum(v_base[:,0:2],axis=1)
        # v_base=np.tile(v_base,(div,1))

        v_base=np.zeros(shape=(num_pop_total,num_objectives))
        v_base[:,0]=np.random.uniform(0,1,size=num_pop_total)
        # v_base[:,0]=np.linspace(0,1,num=num_pop_total,endpoint=True)
        v_max=1-v_base[:,0]
        v_base[:,1]=np.random.uniform(0,v_max)
        v_base[:,2]=1-np.sum(v_base[:,0:2],axis=1)
        return v_base

    @jit(nopython=True,nogil=True)
    def _distance_between_points(p_i_to_compare,population_points,num_t_points):
        num_points=len(population_points)
        # K Value
        distances=np.zeros(shape=population_points[:,0].shape)
        # Loop per points to compare
        for i in np.arange(0,num_points):
            distances[i]=np.linalg.norm(population_points[i]-p_i_to_compare)
        distances=np.argsort(distances)[:num_t_points+1]
        return distances

    @jit(nopython=True,nogil=True)
    def _euclidean_dist_weight_vectors(weight_vectors,num_t_points):
        ind=len(weight_vectors)
        ix_t_closest_vectors=np.zeros(shape=(ind,num_t_points))
        for j in np.arange(0,ind):
            # mask_distinct=(weight_vectors[j]!=weight_vectors).any(axis=1)
            # ix_t_closest_vectors[i]=AlgSpea2._distance_between_points(weight_vectors[j],weight_vectors)
            distances=np.zeros(shape=(ind,))
            # Loop per points to compare
            for i in np.arange(0,ind):
                if i==j:
                    distances[i]=100
                    continue
                distances[i]=np.linalg.norm(weight_vectors[i]-weight_vectors[j])
            ix_t_closest_vectors[j]=np.argsort(distances)[:num_t_points]
        return ix_t_closest_vectors

    @jit(nopython=True,nogil=True)
    def _weighted_sum(objectives,weights,obj_max):
        # Normalizing sum
        # obj_max=np.max(objectives,axis=int(0))
        # obj_max=np.max(objectives,axis=0)
        objectives=objectives/obj_max
        ind=len(objectives)
        fitness=np.zeros(shape=(ind,))
        for i in np.arange(0,ind):
            fitness[i]=np.dot(objectives[i],weights[i])
        return fitness

    def _select_neighbors(ix_t_closest_weights,n_pais):
        # # Indexes de individuos da população e pesos de aptidao
        # indices_apt=np.arange(0,len(ix_t_closest_weights))    

        # First selects which parents will be selected for crossover
        max_ix=len(ix_t_closest_weights) 
        ixs_main=[random.randint(0,max_ix-1) for x in range(0,int(n_pais/2))]

        # Array para armazenar index de valores selecionados
        # idx_ganhadores=np.zeros(shape=(n_pais,1))
        idx_ganhadores=[]
        for i in ixs_main:
            # Seleção de indices de individuos para participação do torneio.
            idx_for_crossover = np.random.choice(ix_t_closest_weights[i],size=2,replace=False)
            # Retorna a linha com a maior aptidao
            idx_ganhadores.append(int(idx_for_crossover[0]))
            idx_ganhadores.append(int(idx_for_crossover[1]))
        return idx_ganhadores

    @jit(nopython=True,nogil=True)
    def _update_fitness(fitness,ix_to_crossover,fitness_cross_mut):
        ix_to_crossover_to_change=[]
        for i in np.arange(0,len(ix_to_crossover)):
            if fitness[ix_to_crossover[i]]>fitness_cross_mut[i]:
                fitness[ix_to_crossover[i]]=fitness_cross_mut[i]
                ix_to_crossover_to_change.append(i)
        return fitness,np.array(ix_to_crossover_to_change)
        
    # @jit(nopython=True,nogil=True)
    def _ponto_dominado_minimizacao(resultado_fn):
        # f1=np.array([10,8,7,7,13])
        # f2=np.array([5,10,11,12,6])
        # resultado_fn=np.column_stack((f1,f2))
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

    def _update_ext_population(resultado_fn,ext_resultado_fn):
        """Avalia as fronteiras de pareto para alocação de cada valor do individuo da população.

        Args:
            resultado_fn (float): Array de floats shape(m,n) com a solução dos valores de individuos avaliados em n funções Ex: f0(coluna 0), f1(coluna 1)...fn(coluna n)
            num_fronteiras (int): Número de fronteiras
        Returns:
            int: Array shape (n,1) com a classificação das fronteiras de pareto para cada individuo
        """
        row,col=resultado_fn.shape
        # # Definição de fronteiras
        # # mask_nao_classificados=np.ones(dtype=bool,shape=(row,))
        # ix_falta_classificar=np.arange(0,row)
        # fronteiras=np.empty(dtype=int,shape=(row,))

        # Loop por fronteiras exceto a ultima pois os valores remanescentes serão adicionados na ultima fronteira
        j=0
        existe_n_dominados=True
        while (j<num_fronteiras-1) & (existe_n_dominados):
        # for j in range(0,num_fronteiras-1):
            dominado=np.ones(shape=(row,))
            # # Verifica se ainda tenho pontos nao classificados
            # 0=Não dominados 1=Dominados Loop por pontos
            dominado[ix_falta_classificar]=AlgMoead._ponto_dominado_minimizacao(resultado_fn[ix_falta_classificar])
            ix_nao_dominados=np.where(dominado==0)[0]
            if len(ix_nao_dominados)==0:
                # print("Não encontrei não dominados")
                existe_n_dominados=False
                continue
            fronteiras[ix_nao_dominados]=j
            # print(sum([1 for x in ix_falta_classificar]))
            ix_falta_classificar=np.delete(ix_falta_classificar,np.where(np.isin(ix_falta_classificar,ix_nao_dominados))[0])
            # print(sum([1 for x in ix_falta_classificar]))
            j+=1
        # Adiciona todos os outros pontos na última fronteira
        fronteiras[ix_falta_classificar]=j


        return fronteiras

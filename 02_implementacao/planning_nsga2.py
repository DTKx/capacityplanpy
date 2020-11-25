# Standard Libraries
import cProfile
import concurrent.futures
import random
import copy
import time
import datetime
import timeit
from itertools import product
# Third Parties
import pandas as pd
import numpy as np
from numba import jit
from pygmo import *

# Local
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\02_trabalho_2\\02_implementacao\\01_parte1\\')
import genetico_permutacao as genetico



class Alocacao():
    # Class Variables

    # Number of Pareto Fronts
    num_fronteiras=3
    # Big Dummy para crowding distance of extremes
    big_dummy=10000

    # Reference Point for Hypervolume
    ref_point=[200,200,7500]
    # ref_point=[-0.01,-0.01,-0.01]
    volume_max=ref_point[0]*ref_point[1]*ref_point[2]

    # Methods
    @jit(nopython=True,nogil=True)
    def pull_requerimentos(pop_tarefa,pop_processador,grafo_e,ind_low_high):
        n_ind,n_tarefas=pop_tarefa.shape

        # Iniciando o loop por individuos(Organizações de tarefas)
        for j in np.arange(0,n_ind):
            pop_processador[j]=pop_processador[j][ind_low_high[j]]
            pop_tarefa[j]=pop_tarefa[j][ind_low_high[j]]

            # Iniciando o loop por tarefa pulo a tarefa 0, pois sempre inicio com ela no processador 0
            for i in np.arange(1,n_tarefas):
                # Retorna mask de tarefas requeridas para iniciar
                requere_nodes=grafo_e[np.where(grafo_e[:,1]==pop_tarefa[j][i])][:,0]

                requere_ix=np.where(np.array([x in set(requere_nodes) for x in pop_tarefa[j]]))[0]
                requere_ix=requere_ix[requere_ix>i]

                if len(requere_ix)>0:
                    requere_ix_max=max(requere_ix)
                else:
                    requere_ix_max=0
                # Sai do loop apenas se a tarefa já está localizada na frente de seus requerimentos independente de processador
                while i<requere_ix_max:
                    # Adiciona as tarefas requeridas em i
                    for k in np.arange(0,len(requere_ix)): 
                        # print(f"ix {ix} nodes {requere_nodes}\n tarefa \n{pop_tarefa[j]}")
                        p_2=np.append(pop_tarefa[j][requere_ix[k]],pop_tarefa[j][np.delete(np.arange(i,n_tarefas),requere_ix[k]-i)])
                        pop_tarefa[j]=np.append(pop_tarefa[j][:i],p_2)
                        # print(f"Tarefa \n{pop_tarefa[j]}")

                        p_2=np.append(pop_processador[j][requere_ix[k]],pop_processador[j][np.delete(np.arange(i,n_tarefas),requere_ix[k]-i)])
                        pop_processador[j]=np.append(pop_processador[j][:i],p_2)

                    # Retorna mask de tarefas requeridas para iniciar
                    requere_nodes=grafo_e[np.where(grafo_e[:,1]==pop_tarefa[j][i])][:,0]
                    # Retorna a maior localização de nós requeridos na população de tarefas


                    requere_ix=np.where(np.array([x in set(requere_nodes) for x in pop_tarefa[j]]))[0]
                    requere_ix=requere_ix[requere_ix>i]

                    if len(requere_ix)>0:
                        requere_ix_max=max(requere_ix)
                    else:
                        requere_ix_max=0

        return pop_tarefa,pop_processador



    # @jit(nopython=True,nogil=True)
    # @staticmethod
    def corrigir_requerimentos(pop_tarefa,pop_processador,grafo_e):
        n_ind,n_tarefas=pop_processador.shape
        pop_tarefa=pop_tarefa[pop_tarefa!=0].reshape(-1,n_tarefas-1)

        # Correção para iniciar com tarefa 0
        pop_tarefa=np.append(np.zeros(shape=(n_ind,1),dtype=np.int64),pop_tarefa,axis=1)

        # Correção para iniciar no processador 0
        pop_processador[:,0]=0

        # Ranking array with index from lowest to highest value
        ind_low_high = np.argsort(pop_processador)

        pop_tarefa,pop_processador=Alocacao.pull_requerimentos(pop_tarefa,pop_processador,grafo_e,ind_low_high)

            # # Iniciando o loop por tarefa pulo a tarefa 0, pois sempre inicio com ela no processador 0
            # for i in np.arange(1,n_tarefas):
            #     # Retorna mask de tarefas requeridas para iniciar
            #     requere_nodes=grafo_e[np.where(grafo_e[:,1]==pop_tarefa[j][i])][:,0]
            #     try:
            #         # Retorna a maior localização de nós requeridos que estão a frente do index i na população de tarefas
            #         requere_ix=np.where(np.array([x in set(requere_nodes) for x in pop_tarefa[j]]))[0]
            #         requere_ix=requere_ix[requere_ix>i]
            #         requere_ix_max=max(requere_ix)
            #     except:
            #     # except ValueError:
            #         requere_ix_max=0
            #     # Sai do loop apenas se a tarefa já está localizada na frente de seus requerimentos independente de processador
            #     while i<requere_ix_max:

            #         # Adiciona as tarefas requeridas em i
            #         for k in np.arange(0,len(requere_ix)): 
            #             # print(f"ix {ix} nodes {requere_nodes}\n tarefa \n{pop_tarefa[j]}")
            #             p_2=np.append(pop_tarefa[j][requere_ix[k]],pop_tarefa[j][np.delete(np.arange(i,n_tarefas),requere_ix[k]-i)])
            #             pop_tarefa[j]=np.append(pop_tarefa[j][:i],p_2)
            #             # print(f"Tarefa \n{pop_tarefa[j]}")

            #             p_2=np.append(pop_processador[j][requere_ix[k]],pop_processador[j][np.delete(np.arange(i,n_tarefas),requere_ix[k]-i)])
            #             pop_processador[j]=np.append(pop_processador[j][:i],p_2)

            #         # Retorna mask de tarefas requeridas para iniciar
            #         requere_nodes=grafo_e[np.where(grafo_e[:,1]==pop_tarefa[j][i])][:,0]
            #         # Retorna a maior localização de nós requeridos na população de tarefas
            #         requere_ix=np.where(np.array([x in set(requere_nodes) for x in pop_tarefa[j]]))
            #         try:
            #             # Retorna a maior localização de nós requeridos que estão a frente do index i na população de tarefas
            #             requere_ix=np.where(np.array([x in set(requere_nodes) for x in pop_tarefa[j]]))[0]
            #             requere_ix=requere_ix[requere_ix>i]
            #             requere_ix_max=max(requere_ix)
            #         except:
            #             requere_ix_max=0
            # # j+=1
        return pop_tarefa,pop_processador

    # JIT está funcionando
    @staticmethod
    @jit(nopython=True,nogil=True)   
    # def t_finalizacao_i(list_t_i,grafo_e,tarefa_pi,tarefas_pj,t_i,list_t_j,processador,t_exec_tarefas,i):
    def t_finalizacao_i(list_t_i,grafo_e,tarefa_pi,tarefas_pj,t_i,list_t_j,processador,t_exec_tarefas,i,kinat,kcomu):

        # Retorna index das tarefas requeridas para iniciar
        requere_nodes=grafo_e[np.where(grafo_e[:,1]==tarefa_pi)][:,0]
        requere_no_outro_proc=np.array([x in set(requere_nodes) for x in tarefas_pj])

        if np.sum(requere_no_outro_proc)>0:
            # Verifica os indices dos nós requeridos
            ix_nos_requeridos=np.where(requere_no_outro_proc)[0]

            requere_nodes_p=tarefas_pj[ix_nos_requeridos]

            t_comunicacao_list=np.array([grafo_e[:,2][(grafo_e[:,0]==no)&(grafo_e[:,1]==tarefa_pi)][0] for no in requere_nodes_p])

            # # Verifica o máximo dos valores dos tempos corridos + tempo de comunicação de todos as tarefas dependentes que estão no outro processador

            max_t_comunicacao_requere=max(t_comunicacao_list+list_t_j[ix_nos_requeridos])
            t_comunica=np.sum(t_comunicacao_list)

            t_ocioso_i=int(max_t_comunicacao_requere-t_i)
            if t_ocioso_i<0:
                t_ocioso_i=0

        # Considera o processador atual
        else:
            t_ocioso_i=0
            t_comunica=0

        # Tempo execucao
        t_exec=t_exec_tarefas[i]

        # Tempo corrido
        t_i=t_i+t_ocioso_i+t_exec

        list_t_i=np.append(list_t_i,t_i)
        # return list_t_i,t_i,t_ocioso_i,t_comunicacao_list
        energia=t_ocioso_i*kinat+t_comunica*kcomu
        # energia=sum(t_ocioso_i)*kinat+sum(t_comunicacao_list)*kcomu
        return list_t_i,t_i,energia

    @staticmethod
    # @jit(nopython=True)   
    def avaliar_fitness(pop_tarefa,pop_processador,grafo_v,grafo_e):
        # Constantes
        kativ=int(100)
        kinat=int(5)
        kcomu=int(20)

        fit_t_exec=[]
        fit_t_exec_medio=[]
        fit_energia=[]

        # Tarefas que são realizadas em cada processador
        pop_t_exec=grafo_v[:,1][pop_tarefa]
        ener_exec_all=sum(pop_t_exec[0])*kativ

        for tarefas,processadores,t_exec_tarefas in zip (pop_tarefa,pop_processador,pop_t_exec):
            energia_0,energia_1=0,0

            tarefas_p0=tarefas[processadores==0]
            tarefas_p1=tarefas[processadores==1]

            # Tempo corrido
            t_0=t_exec_tarefas[0]
            t_1=0

            list_t_0=np.array([t_0])
            list_t_1=np.array([])


            # Loop por tarefa
            for i in range(1,len(tarefas)):
                if processadores[i]==0:
                    # Verifica o tempo corrido
                    list_t_0,t_0,energia_0=Alocacao.t_finalizacao_i(list_t_0,grafo_e,tarefas[i],tarefas_p1,t_0,list_t_1,processadores[i],t_exec_tarefas,i,kinat,kcomu)
                    # list_t_0,t_0,t_ocioso_0,t_comunicacao_list_0=Alocacao.t_finalizacao_i(list_t_0,grafo_e,tarefas[i],tarefas_p1,t_0,list_t_1,processadores[i],t_exec_tarefas,i)
                # Considera o caso em que o processador é o 1
                else:
                    # Verifica o tempo ocioso
                    list_t_1,t_1,energia_1=Alocacao.t_finalizacao_i(list_t_1,grafo_e,tarefas[i],tarefas_p0,t_1,list_t_0,processadores[i],t_exec_tarefas,i,kinat,kcomu)
                    # list_t_1,t_1,t_ocioso_1,t_comunicacao_list_1=Alocacao.t_finalizacao_i(list_t_1,grafo_e,tarefas[i],tarefas_p0,t_1,list_t_0,processadores[i],t_exec_tarefas,i)
            fit_t_exec.append(max(t_0,t_1))
            fit_t_exec_medio.append((t_0+t_1)/2.0)
            fit_energia.append(ener_exec_all+energia_0+energia_1)
            # energia=ener_exec_all+(sum(t_ocioso_0)+sum(t_ocioso_1))*kinat+(sum(t_comunicacao_list_0)+sum(t_comunicacao_list_1))*kcomu
        return np.array(fit_t_exec),np.array(fit_t_exec_medio),np.array(fit_energia)

    @staticmethod
    def calcula_crowding_dist(resultado_f1_f2):
        """Calcula a crowding distance

        Args:
            populacao (float): Array of float
        """
        # # Avalio o tempo fitness da minha solução
        # resultado_f1_f2=Alocacao.calculo_funcao(populacao)

        # Inicio da fronteira 1
        # Classificação dos individuos dominancia Individuos não dominados
        # fronteiras=genetico.AlgNsga2._fronteiras(resultado_f1_f2,Alocacao.num_fronteiras)
        fronteiras=genetico.AlgNsga2._fronteiras(copy.deepcopy(resultado_f1_f2),Alocacao.num_fronteiras)

        # Calcula o crowding_fitness
        # crowding_dist=genetico.AlgNsga2._crowding_distance(resultado_f1_f2,fronteiras,Alocacao.big_dummy)
        crowding_dist=genetico.AlgNsga2._crowding_distance(resultado_f1_f2,copy.deepcopy(fronteiras),Alocacao.big_dummy)
        return crowding_dist,fronteiras


    def main(self,n_execucao,num_pop_total,ger_max,n_tour,perc_crossover,proba_mutacao,nome_grafo):
        start=time.perf_counter()
        # Seleção grafo
        # Seleção grafo
        if nome_grafo=="gauss18":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,4],[4,4],[5,4],[6,6],[7,3],[8,3],[9,3],[10,3],[11,4],[12,2],[13,2],[14,2],[15,2],[16,1],[17,1]],dtype=int)
            grafo_e=np.array([[0,1,12],[0,2,12],[0,3,12],[0,4,12],[0,5,12],[0,6,12],[2,6,8],[2,7,8],[3,8,8],[4,9,8],[5,10,8],[6,7,12],[6,8,12],[6,9,12],[6,10,12],[6,11,12],[8,11,8],[8,12,8],[9,13,8],[10,14,8],[11,12,12],[11,13,12],[11,14,12],[11,15,12],[13,15,8],[13,16,8],[14,17,8],[15,16,12],[15,17,12]],dtype=int)
        elif nome_grafo=="p11_a":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,4],[4,4],[5,4],[6,6],[7,3],[8,3],[9,3],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,8],[0,2,12],[1,3,12],[1,4,12],[2,5,8],[2,6,12],[3,7,8],[4,7,8],[4,8,8],[5,8,8],[5,9,8],[6,9,8],[7,10,12],[8,10,8],[9,10,12]],dtype=int)
        elif nome_grafo=="p11_b":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,6],[4,4],[5,4],[6,3],[7,4],[8,3],[9,2],[10,3]],dtype=int)                             
            grafo_e=np.array([[0,1,8],[0,2,8],[0,3,8],[1,4,12],[2,4,8],[2,5,12],[2,6,8],[3,6,12],[4,7,8],[4,8,8],[5,8,8],[5,9,8],[6,10,12]],dtype=int)
        elif nome_grafo=="p11_c":
            grafo_v=np.array([[0,8],[1,3],[2,6],[3,4],[4,2],[5,4],[6,6],[7,3],[8,4],[9,3],[10,2]],dtype=int)
            grafo_e=np.array([[0,1,10],[0,2,10],[1,3,8],[1,4,8],[2,5,8],[2,6,8],[3,7,12],[4,8,8],[5,9,10],[6,10,8]],dtype=int)
        elif nome_grafo=="p11_d":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,4],[4,2],[5,1],[6,6],[7,3],[8,4],[9,4],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,10],[0,2,10],[0,3,10],[0,4,10],[0,6,10],[1,5,12],[2,5,8],[3,6,8],[3,7,8],[4,7,12],[5,8,8],[6,8,10],[6,9,10], [7,9,8],[8,10,8],[9,10,8]],dtype=int)
        elif nome_grafo=="p11_e":
            grafo_v=np.array([[0,8],[1,2],[2,3],[3,4],[4,2],[5,4],[6,6],[7,3],[8,4],[9,2],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,12],[0,2,12],[0,3,12],[0,4,12],[1,3,8],[1,6,8],[2,4,8],[2,7,8],[3,5,10],[4,5,10],[4,7,10],[5,6,8],[5,8,8],[5,9,8],[6,8,10],[7,9,10], [8,10,4],[9,10,4]],dtype=int)
        elif nome_grafo=="p11_f":
            grafo_v=np.array([[0,8],[1,4],[2,3],[3,4],[4,2],[5,4],[6,6],[7,1],[8,3],[9,2],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,12],[0,2,12],[0,3,12],[0,4,12],[1,5,10],[2,5,4],[2,6,4],[2,7,4],[3,7,10],[3,8,10],[3,9,10],[4,9,8],[4,10,8]],dtype=int)
        elif nome_grafo=="p11_g":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,4],[4,4],[5,4],[6,6],[7,3],[8,3],[9,3],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,12],[0,2,12],[0,3,12],[1,4,8],[2,4,10],[2,5,10],[2,7,10],[3,6,8],[4,7,4],[5,8,4],[5,9,4],[6,8,4],[6,9,4],[7,10,4],[8,10,4],[9,10,4]],dtype=int)
        elif nome_grafo=="p11_h":
            grafo_v=np.array([[0,8],[1,4],[2,4],[3,4],[4,2],[5,2],[6,3],[7,3],[8,6],[9,4],[10,4]],dtype=int)
            grafo_e=np.array([[0,1,10],[0,2,10],[0,3,10],[0,4,10],[1,5,8],[1,8,8],[2,5,12],[2,6,12],[3,6,10],[3,7,10],[4,7,12],[4,10,12],[5,8,4],[5,9,4],[6,9,4],[7,9,4],[7,10,4]],dtype=int)
        elif nome_grafo=="p11_i":
            grafo_v=np.array([[0,8],[1,6],[2,4],[3,3],[4,4],[5,2],[6,4],[7,4],[8,6],[9,5],[10,2]],dtype=int)
            grafo_e=np.array([[0,1,8],[0,2,8],[0,3,8],[0,4,8],[0,5,8],[1,6,6],[2,6,6],[2,7,6],[3,7,6],[3,8,6],[4,8,6],[4,9,6],[5,9,6],[6,10,10],[9,10,10]],dtype=int)
        elif nome_grafo=="p11_j":
            grafo_v=np.array([[0,8],[1,3],[2,4],[3,3],[4,1],[5,2],[6,4],[7,4],[8,6],[9,2],[10,3]],dtype=int)
            grafo_e=np.array([[0,1,10],[0,2,10],[0,3,10],[0,5,10],[0,7,10],[1,4,8],[2,5,8],[2,6,8],[3,7,8],[4,8,10],[4,9,10],[5,9,4],[6,9,4],[6,10,4],[7,10,12]],dtype=int)

        # Define o número da tarefa
        num_tarefa=(len(grafo_v))

        # Gerar população inicial de atividades de atividades sem repetição e processadores iniciando sempre na atividade 0 no processador 0 
        pop_tarefa,pop_processador=genetico._gerar_pop_tarefa_proc_permutacao(num_pop_total,num_tarefa,False)
        # print("out gerar pop")

        # Corrige as tarefas e processador 
        pop_tarefa,pop_processador=Alocacao.corrigir_requerimentos(pop_tarefa,pop_processador,grafo_e)
        # print("out corrigir requerimentos")

        # Avalio o tempo fitness da minha solução
        fitness_t_exec,fit_t_exec_medio,fit_energia=self.avaliar_fitness(pop_tarefa,pop_processador,copy.deepcopy(grafo_v),copy.deepcopy(grafo_e))
        objectives=np.column_stack((fitness_t_exec,fit_t_exec_medio,fit_energia))

        # print("out avaliar fitness")
        # print("In Crowding")
        # Calcula Crowding Distance
        crowd_dist,fronteiras=Alocacao.calcula_crowding_dist(objectives)

        # print("Pop inicial")
        # ix_front=np.where(fronteiras==0)[0]
        # ix_best=np.argmin(objectives[ix_front][:,0])
        # ix_best_crowd=np.argmax(crowd_dist[ix_front])
        # print(f"best exe exec {objectives[ix_front][ix_best]} crowd {crowd_dist[ix_front][ix_best]} front {fronteiras[ix_front][ix_best]}")
        # print(f"best cro exec {objectives[ix_front][ix_best_crowd]} crowd {crowd_dist[ix_front][ix_best_crowd]} front {fronteiras[ix_front][ix_best_crowd]}")

        # Número de pais
        n_pais = int(num_pop_total * perc_crossover)
        # Verifico que tenho um valor par de pares de pais
        if n_pais % 2 == 1:
            n_pais = n_pais + 1

        ger=0
        convergiu=0
        # Inicio as minhas gerações
        while ger<ger_max:            
            # print("ger",ger)
            # Seleção para crossover 
            # Roleta utilizando a inversão pelo pior valor do fitness
            # print("in torneio")
            # Torneio simples
            ix_to_crossover=genetico.AlgNsga2._torneio_simples_nsga2(copy.deepcopy(pop_tarefa),copy.deepcopy(pop_processador),copy.deepcopy(crowd_dist),copy.deepcopy(fronteiras),n_pais,n_tour)
            # print("out torneio")

            # Crossover ciclico
            pop_tarefa_cross,pop_processador_cross=genetico._crossover_ciclico(copy.deepcopy(pop_tarefa[ix_to_crossover]),copy.deepcopy(pop_processador[ix_to_crossover]))
            # print("out cross")

            # Mutação
            pop_tarefa_cross_mut,pop_processador_cross_mut=genetico._mutacao_2_genes(pop_tarefa_cross,pop_processador_cross,proba_mutacao)
            # print("out mutacao")

            # Corrige as tarefas e processador
            pop_tarefa_cross_mut_cor,pop_processador_cross_mut_cor=Alocacao.corrigir_requerimentos(pop_tarefa_cross_mut,pop_processador_cross_mut,grafo_e)
            # print("out corrigir")

            # Avalio o tempo fitness da minha solução
            fitness_t_exec_cross_mut,fit_t_exec_medio_cross_mut,fit_energia_cross_mut=Alocacao.avaliar_fitness(pop_tarefa_cross_mut_cor,pop_processador_cross_mut_cor,copy.deepcopy(grafo_v),copy.deepcopy(grafo_e))

            objectives_cross_mut=np.column_stack((fitness_t_exec_cross_mut,fit_t_exec_medio_cross_mut,fit_energia_cross_mut))

            # print("out fitness")
            # Calcula Crowding Distance
            crowd_dist_cross_mut,fronteiras_cross_mut=Alocacao.calcula_crowding_dist(objectives_cross_mut)

            # print("Cross_Mut")
            # ix_front=np.where(fronteiras_cross_mut==0)[0]
            # ix_best=np.argmin(objectives_cross_mut[ix_front][:,0])
            # ix_best_crowd=np.argmax(crowd_dist_cross_mut[ix_front])
            # print(f"best exe exec {objectives_cross_mut[ix_front][ix_best]} crowd {crowd_dist_cross_mut[ix_front][ix_best]} front {fronteiras_cross_mut[ix_front][ix_best]}")
            # print(f"best cro exec {objectives_cross_mut[ix_front][ix_best_crowd]} crowd {crowd_dist_cross_mut[ix_front][ix_best_crowd]} front {fronteiras_cross_mut[ix_front][ix_best_crowd]}")

            # Combino as populacoes e fitness
            pop_tarefa = np.vstack((pop_tarefa,pop_tarefa_cross_mut_cor))
            pop_processador = np.vstack((pop_processador,pop_processador_cross_mut_cor))
            crowd_dist=np.concatenate((crowd_dist,crowd_dist_cross_mut))
            fronteiras=np.concatenate((fronteiras,fronteiras_cross_mut))
            objectives=np.concatenate((objectives,objectives_cross_mut))

            # Reinserção Linear

            # Drop duplicates
            unique_tar,unique_ix=np.unique(pop_tarefa,return_index=True,axis=0)
            duplicates_ix=np.setxor1d(np.arange(0,n_pais+num_pop_total),unique_ix)
            if len(duplicates_ix)>0:
                # Check if they are really duplicates or the processor array is equal
                dupl_verify=np.append(pop_tarefa[duplicates_ix],pop_processador[duplicates_ix],axis=1)
                unique_tar_proc,unique_tar_proc_ix=np.unique(dupl_verify,return_index=True,axis=0)
                unique_ix=np.append(unique_ix,duplicates_ix[unique_tar_proc_ix])

                pop_tarefa = pop_tarefa[unique_ix]
                pop_processador = pop_processador[unique_ix]
                crowd_dist = crowd_dist[unique_ix]
                fronteiras = fronteiras[unique_ix]
                objectives=objectives[unique_ix]

            ix_prox_ger=genetico.AlgNsga2._index_reinsercao_ordenada_nsga(crowd_dist,fronteiras,num_pop_total)
            # Atualização da população com base na reinserção linear

            pop_tarefa = pop_tarefa[ix_prox_ger]
            pop_processador = pop_processador[ix_prox_ger]
            crowd_dist = crowd_dist[ix_prox_ger]
            fronteiras = fronteiras[ix_prox_ger]
            objectives=objectives[ix_prox_ger]


            # # Calcula o hypervolume
            # hv = hypervolume(points = objectives)
            # volume_ger=hv.compute(Alocacao.ref_point)
            # hv_vol_norma=volume_ger/Alocacao.volume_max
            # print(f"Hypervolume {hv_vol_norma}")

            # uniq,counts=np.unique(fronteiras,return_counts=True)
            # ix_front=np.where(fronteiras==0)[0]
            # ix_best=np.argmin(objectives[ix_front][:,0])
            # ix_best_crowd=np.argmax(crowd_dist[ix_front])
            # print(f"Ger {ger} {uniq,counts}")
            # print(f"best exe exec {objectives[ix_front][ix_best]} crowd {crowd_dist[ix_front][ix_best]} front {fronteiras[ix_front][ix_best]}")
            # print(f"best cro exec {objectives[ix_front][ix_best_crowd]} crowd {crowd_dist[ix_front][ix_best_crowd]} front {fronteiras[ix_front][ix_best_crowd]}")


            # if sum(counts)>100:
            #     print("Problem")


            # print(f"Ger {ger},best {np.amin(fitness_t_exec)}, worst {np.amax(fitness_t_exec)}, unicos {np.unique(pop_tarefa,axis=0).shape[0]}")
            ger+=1
            # if np.amin(fitness_t_exec)==44.0:
            #     break

        # Calcula o hypervolume
        # ref_point=nadir(objectives)+10.0
        hv = hypervolume(points = objectives)
        volume_ger=hv.compute(Alocacao.ref_point)
        # hv_vol_norma=volume_ger
        hv_vol_norma=volume_ger/Alocacao.volume_max
        # print(f"Hypervolume {hv_vol_norma}")

        # Verifica o melhor individuo da execução e adiciona na lista
        ix_best=np.argmin(objectives[:,0])

        finish=time.perf_counter()
        # print(f'{n_execucao}, fitness {str(fitness_t_exec[ix_best])}, fitness médio {str(np.mean(fitness_t_exec))},tempo {finish-start}')
        return (n_execucao,str(objectives[:,0][ix_best]),str(pop_tarefa[ix_best]),str(pop_processador[ix_best]),np.mean(objectives[:,0]),np.mean(objectives[:,1]),np.mean(objectives[:,2]),finish-start,hv_vol_norma)



    def main_parallel(self):
        # nome_grafo=["gauss18"]
        # Deadlocks com Thread e Process
        # nome_grafo=["p11_g","p11_h"]
        # Deadlocks com Thread
        # nome_grafo=["p11_i","p11_j"]
        # t_otimo=["44.0","40.0","31.0","32.0","38.0","40.0","28.0","37.0","33.0","32.0","30.0"]
        nome_grafo=["gauss18","p11_a","p11_b","p11_c","p11_d","p11_e","p11_f","p11_g","p11_h","p11_i","p11_j"]


        num_pop_i=[100]
        num_ger_i=[200]
        n_tour_i=[2]
        perc_crossover_i=[0.6]
        probab_mutacao_i=[0.3]

        # num_pop_i=[50,100,200]
        # num_ger_i=[200,300,500]
        # n_tour_i=[2,3,4]
        # perc_crossover_i=[0.6]
        # probab_mutacao_i=[0.3]

        variantes = list(product(*[num_pop_i,num_ger_i,n_tour_i,perc_crossover_i,probab_mutacao_i,nome_grafo]))
        for param in variantes:
            if param[5]=="gauss18":t_otimo="44.0"
            elif param[5]=="p11_a":t_otimo="40.0"
            elif param[5]=="p11_b":t_otimo="31.0"
            elif param[5]=="p11_c":t_otimo="32.0"
            elif param[5]=="p11_d":t_otimo="38.0"
            elif param[5]=="p11_e":t_otimo="40.0"
            elif param[5]=="p11_f":t_otimo="28.0"
            elif param[5]=="p11_g":t_otimo="37.0"
            elif param[5]=="p11_h":t_otimo="33.0"
            elif param[5]=="p11_i":t_otimo="32.0"
            elif param[5]=="p11_j":t_otimo="30.0"

            t_0_execs=time.perf_counter()
            # Numero de execuções
            input_n_execucoes = 20
            # Lista iterável para chamar função
            lista_n_exec=range(0,input_n_execucoes)
            resultado_final=[]
            convergiu=0
            aptidao_media_n_ger=[]
            hv_medio_media_n_ger=[]
            pior_apt=0

            # with concurrent.futures.ThreadPoolExecutor() as executor:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for resultado in (executor.map(Alocacao().main,lista_n_exec,[param[0]]* len(lista_n_exec),
                [param[1]]* len(lista_n_exec),[param[2]]* len(lista_n_exec),[param[3]]* len(lista_n_exec),
                [param[4]]* len(lista_n_exec),[param[5]]* len(lista_n_exec))):
                    print(resultado)
                    resultado_final.append(resultado)
                    if resultado[1]==t_otimo:
                        convergiu=convergiu+1
                    if resultado[5]>pior_apt:
                        pior_apt=resultado[5]
                    aptidao_media_n_ger.append(resultado[4])                   
                    hv_medio_media_n_ger.append(resultado[8])                   

            # Verificar o resultado
            num_pop=f"_pop{param[0]}"
            num_ger=f"_ger{param[1]}"
            tipo_selecao=f"_tour{param[2]}"
            tipo_crossover="_ciclico"
            perc_crossover=f"_%cross{param[3]}"
            probab_mutacao=f"_%mutacao{param[4]}"
            nome_tipo_exec=param[5]+num_pop+num_ger+tipo_selecao+tipo_crossover+perc_crossover+probab_mutacao
            root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\02_trabalho_2\\01_dados\\02_raw_data_parte_2\\02_nsga2\\"
            file_name = "nsga2_p2_por_ger.csv"
            path = root_path + file_name

            resultado_final_pd=pd.DataFrame(resultado_final)
            resultado_final_pd["Tipo"]=nome_tipo_exec
            # header=['Execucao','Fitness','Tarefas',"Processadores","Tempo [s],"Tipo"]
            resultado_final_pd.to_csv(path, mode='a',header=False,index=False)

            t_1_execs=time.perf_counter()

            # print(f'\n{nome_tipo_exec} Tempo total {t_1_execs-t_0_execs}, tempo médio/execucao {(t_1_execs-t_0_execs)/input_n_execucoes} convergiu {convergiu}  {(convergiu/input_n_execucoes)*100}% \n')
            # header=['Grafo','Tpop','Nger','Tour','Converg','Média','Pior',"% convergencia","Tempo total [s]","Tempo/execucao [s]","Crossover","% Crossover","% Mutacao","Média hipervolume"
            line_resumo=(f'\n{param[5]},{param[0]},{param[1]},{param[2]},{convergiu},{sum(aptidao_media_n_ger)/len(aptidao_media_n_ger)},{pior_apt},{convergiu/input_n_execucoes},{t_1_execs-t_0_execs},{(t_1_execs-t_0_execs)/input_n_execucoes},{tipo_crossover},{param[3]},{param[4]},{sum(hv_medio_media_n_ger)/len(hv_medio_media_n_ger)}')
            print(line_resumo)
            print(convergiu)

            root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\02_trabalho_2\\01_dados\\02_raw_data_parte_2\\02_nsga2\\"
            file_name = "nsga2_p2_resultados.csv"
            path = root_path + file_name

            file = open(path,'a')
            file.write(line_resumo)
            file.close()
        print("Termino todas variantes")

    def main_cprofile(self):
        """Retorna o cProfile com uma execução, para avaliação de tempo
        """
        # nome_grafo=["p11_h"]
        # nome_grafo=["p11_g","p11_h"]

        # nome_grafo=["p11_g","p11_h","p11_i","p11_j"]
        nome_grafo=["gauss18"]
        # nome_grafo=["gauss18","p11_a","p11_b","p11_c","p11_d","p11_e","p11_f","p11_g","p11_h","p11_i","p11_j"]

        tipo_crossover="ciclico"

        num_pop_i=[100]
        num_ger_i=[200]
        n_tour_i=[2]
        perc_crossover_i=[0.6]
        probab_mutacao_i=[0.3]

        variantes = list(product(*[num_pop_i,num_ger_i,n_tour_i,perc_crossover_i,probab_mutacao_i,nome_grafo]))
        for param in variantes:
            if param[5]=="gauss18":t_otimo="44.0"
            elif param[5]=="p11_a":t_otimo="40.0"
            elif param[5]=="p11_b":t_otimo="31.0"
            elif param[5]=="p11_c":t_otimo="32.0"
            elif param[5]=="p11_d":t_otimo="38.0"
            elif param[5]=="p11_e":t_otimo="40.0"
            elif param[5]=="p11_f":t_otimo="28.0"
            elif param[5]=="p11_g":t_otimo="37.0"
            elif param[5]=="p11_h":t_otimo="33.0"
            elif param[5]=="p11_i":t_otimo="32.0"
            elif param[5]=="p11_j":t_otimo="30.0"

            t_0_execs=time.perf_counter()

            # Numero de execuções
            input_n_execucoes = 20
            # Lista iterável para chamar função
            lista_n_exec=range(0,input_n_execucoes)
            resultado_final=[]
            convergiu=0
            aptidao_media_n_ger=[]
            pior_apt=0
            hv_medio_media_n_ger=[]

            # return (n_execucao 1,str(objectives[:,0][ix_best])2,str(pop_tarefa[ix_best])3,4str(pop_processador[ix_best]),5np.mean(objectives[:,0]),6np.mean(objectives[:,1]),7np.mean(objectives[:,2]),8finish-start,9np.unique(fronteiras,return_counts=True),10HV)

            # num_pop_total,ger_max,n_tour,perc_crossover,proba_mutacao=param[0],param[1],param[2],param[3],param[4]
            for resultado in (map(Alocacao().main,lista_n_exec,[param[0]]* len(lista_n_exec),
                [param[1]]* len(lista_n_exec),[param[2]]* len(lista_n_exec),[param[3]]* len(lista_n_exec),
                [param[4]]* len(lista_n_exec),[param[5]]* len(lista_n_exec))):           
                print(resultado)
                resultado_final.append(resultado)
                if resultado[1]==t_otimo:
                    convergiu=convergiu+1
                if resultado[5]>pior_apt:
                    pior_apt=resultado[5]
                aptidao_media_n_ger.append(resultado[4])
                hv_medio_media_n_ger.append(resultado[8])                   


            # Verificar o resultado
            num_pop=f"_pop{param[0]}"
            num_ger=f"_ger{param[1]}"
            tipo_selecao=f"_tour{param[2]}"
            tipo_crossover="_ciclico"
            perc_crossover=f"_%cross{param[3]}"
            probab_mutacao=f"_%mutacao{param[4]}"
            nome_tipo_exec=param[5]+num_pop+num_ger+tipo_selecao+tipo_crossover+perc_crossover+probab_mutacao
            root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\02_trabalho_2\\01_dados\\02_raw_data_parte_2\\02_nsga2\\"

            file_name = "nsga2_p2_bd_por_geracao.csv"
            path = root_path + file_name

            resultado_final_pd=pd.DataFrame(resultado_final)
            resultado_final_pd["Tipo"]=nome_tipo_exec
            # header=['Execucao','Fitness','Tarefas',"Processadores","Tempo [s],"Tipo"]
            resultado_final_pd.to_csv(path, mode='a',header=False,index=False)

            t_1_execs=time.perf_counter()

            # print(f'\n{nome_tipo_exec} Tempo total {t_1_execs-t_0_execs}, tempo médio/execucao {(t_1_execs-t_0_execs)/input_n_execucoes} convergiu {convergiu}  {(convergiu/input_n_execucoes)*100}% \n')
            # header=['Grafo','Tpop','Nger','Tour','Converg','Média','Pior',"% convergencia","Tempo total [s]","Tempo/execucao [s]","Crossover","% Crossover","% Mutacao"
            line_resumo=(f'\n{param[5]},{param[0]},{param[1]},{param[2]},{convergiu},{sum(aptidao_media_n_ger)/len(aptidao_media_n_ger)},{pior_apt},{convergiu/input_n_execucoes},{t_1_execs-t_0_execs},{(t_1_execs-t_0_execs)/input_n_execucoes},{tipo_crossover},{param[3]},{param[4]},{sum(hv_medio_media_n_ger)/len(hv_medio_media_n_ger)}')
            print(line_resumo)
            print(convergiu)
            # return (n_execucao,str(objectives[:,0][ix_best]),str(pop_tarefa[ix_best]),str(pop_processador[ix_best]),np.mean(objectives[:,0]),np.mean(objectives[:,1]),np.mean(objectives[:,2]),finish-start,np.unique(fronteiras,return_counts=True))

            root_path = "C:\\Users\\Debora\\Documents\\01_UFU_local\\01_comp_evolutiva\\02_trabalho_2\\01_dados\\02_raw_data_parte_2\\02_nsga2\\"
            file_name = "nsga2_p2_resultados.csv"
            path = root_path + file_name

            file = open(path,'a')
            file.write(line_resumo)
            file.close()
            print("Termino todas variantes")

            # def main(self,n_execucao,num_pop_total,ger_max,n_tour,perc_crossover,proba_mutacao,nome_grafo):
            # print(f"Alocacao().main(0,{param[0]},{param[1]},{param[2]},{param[3]},{param[4]},gauss18)")
            # cProfile.runctx("Alocacao().main(0,100,200,2,0.6,0.3,gauss18)",globals(),locals())
            # cProfile.runctx(f"Alocacao().main(0,{param[0]},{param[1]},{param[2]},{param[3]},{param[4]},gauss18)",globals(),locals())
            # cProfile.runctx(f"Alocacao().main(2,{param[0]},{param[1]},{param[2]},{param[3]},{param[4]},{param[5]})",globals(),locals())

            # cProfile.runctx(f"Alocacao().main(0,{[param[0]]},{[param[1]]},{[param[2]]},{[param[3]]},{[param[4]]},{[param[5]]})",globals(),locals())
            # cProfile.runctx(f"Alocacao().main(0,{num_pop_total},{ger_max},{n_tour},{perc_crossover},{proba_mutacao},{nome_grafo[0]})",globals(),locals())


if __name__=="__main__":
    # Alocacao().main_cprofile()
    Alocacao().main_parallel()

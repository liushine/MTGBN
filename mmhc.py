#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:33:53 2022

@author: shine
"""
import pandas as pd
import numpy as np
import copy
import math
import networkx as nx

from hmc import log_multi_gamma,log_iw_const
from utils import makedecompgraph 


def change_form_dic_to_matrix2(dic):
    key=list(dic.keys())
    value=list(dic.values())
    s = (len(dic),len(dic))
    matrix=np.zeros(s)
    df=pd.DataFrame(matrix, columns=key, index=key) 
    for i in range(0,len(key)):
        if len(value[i])<=0 :
             continue
        ase=value[i]
        for j in ase:
              df.loc[j,key[i]]=1 
    df=np.array(df,'float32')
    return df

def cond_density(v0,p,ni,sigma_h,S,cond):
    gnt_conditional_density=v0/2*math.log(np.linalg.det(sigma_h[cond][:,cond] ))+ \
    log_multi_gamma(p,((v0+ni)/2))-(ni*p)/2*math.log(math.pi)-(v0+ni)/2*math.log(np.linalg.det(S[cond][:,cond]))-log_multi_gamma(p,v0/2)
    return gnt_conditional_density
# The score function concludes two parts: 1) logP(D_i|B_i, \sigma_i^h); 2) logP(B_i|, \sigma_i^h)

#The function score_gbn_D is used to calculate logP(D_i|B_i, \sigma_i^h)    
def score_gbn_D(dag,sigma_h,v0,p,ni,S):
    numbernodes=len(dag)
    dgraph = nx.from_numpy_matrix(dag, create_using=nx.DiGraph)
    par= np.empty((numbernodes),dtype=object)
    v_and_p=np.empty((numbernodes),dtype=object)
    for i in range(0,numbernodes):
        vertice=[i]
        parent=list(dgraph.predecessors(i))
        par[i]=parent
        vertice_and_parent=sorted(vertice+parent)
        v_and_p[i]=vertice_and_parent
    score_part1=0
    for i in range(0,numbernodes):
        score_part1=score_part1+cond_density(v0,p,ni,sigma_h,S,v_and_p[i])-cond_density(v0,p,ni,sigma_h,S,par[i])
    return score_part1 

#The function score_gbn_D is used to calculate logP(B_i|, \sigma_i^h)
def score_gbn_B(dag,sigma_h,v0,p):
    Gi=makedecompgraph(dag)
    cliques=Gi[:,0]
    separators=Gi[:,1]
    numbercliques=len(cliques)
    score_part2=log_iw_const((v0+len(cliques[0])-1),(sigma_h[cliques[0]][:,cliques[0]])/2,p)
    if numbercliques >1:
           for i in range(1,numbercliques-1):
                score_part2 = score_part2+log_iw_const((v0+len(cliques[i])-1),(sigma_h[cliques[i]][:,cliques[i]])/2,len(cliques[i]))- \
                log_iw_const((v0+len(separators[i])-1),(sigma_h[separators[i]][:,separators[i]])/2,len(separators[i]))
    else :
           score_part2=score_part2
    return score_part2                   
def score_gbn(dag,sigma_h,v0,p,ni,S):
    score=score_gbn_D(dag,sigma_h,v0,p,ni,S)+score_gbn_B(dag,sigma_h,v0,p)
    return score

def score_sum(G,data,sigma_h):
    scoresum=0
    m=len(G)
    for i in range(1,m+1):
        Gi=G[i]
        data=dataset[i]
        ni=data.shape[0]
        si=data.T.dot(data)/ni
        S=sigma_h+ni*si
        scoresum=scoresum+score_gbn(dag,sigma_h,v0,p,ni,S)
    return scoresum  

# The function maximize_Qfunction2 is used to calclulate Q function 
    
def maximize_Qfunction2 (Gi,datai,v0,hms_sample):
    maximize_Q2=0
    N=hms_sample.shape[0]
    p=len(Gi)
    ni=datai.shape[0]
    si=datai.T.dot(datai)/ni
    for i in range (0,N):
        sigma_h=hms_sample[i]
        S=sigma_h+ni*si
        maximize_Q2=maximize_Q2+score_gbn(Gi,sigma_h,v0,p,ni,S)
    maximize_Q2=maximize_Q2/N
    return maximize_Q2  

def change_form_dic_to_matrix2(dic):
    key=list(dic.keys())
    value=list(dic.values())
    s = (len(dic),len(dic))
    matrix=np.zeros(s)
    df=pd.DataFrame(matrix, columns=key, index=key) 
    for i in range(0,len(key)):
        if len(value[i])<=0 :
             continue
        ase=value[i]
        for j in ase:
              df.loc[j,key[i]]=1 
    df=np.array(df,'float32')
    return df

def hc(pc,dataset,v0,hms_sample):  
    """Use Max-Min Hill-Climing algorithm to search the Bayesian network structure 
    Args：
        pc: the network structure learn from mmpc
        dataset: observational datasets
        v0: degree of freedom in the inverse Wishart distribution
        hms_sample: samples of \sigma_{h} generated from hmc
        
    Returns:
        Bayesian network structure B
    """
    
    
    data=pd.DataFrame(dataset)
    dataset=np.array(dataset,'float32')
    gra = {}
    gra_temp = {}
    iteration=0
    for node in data:
        gra[node] = []
        gra_temp[node] = []
    diff = 1
    # attempt to find better graph until no difference could make
    while diff > 1e-10:

        diff = 0
        edge_candidate = []
        gra_temp = copy.deepcopy(gra)

        cyc_flag = False

        for tar in data:
            # attempt to add edges
            for pc_var in pc[tar]:
                underchecked = [pc_var]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = copy.deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if gra[gra_par]:
                            if tar in gra[gra_par]:
                                cyc_flag = True
                                break
                            else:
                                for key in gra[gra_par]:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    gra_temp[tar].append(pc_var)
                    score_diff_temp = maximize_Qfunction2 (change_form_dic_to_matrix2(gra_temp),dataset,v0,hms_sample)-\
                                      maximize_Qfunction2 (change_form_dic_to_matrix2(gra),dataset,v0,hms_sample)
                    if (score_diff_temp - diff > -1e-10):
                        diff = score_diff_temp
                        edge_candidate = [tar, pc_var, 'a']
                        iteration=iteration+1
                    gra_temp[tar].remove(pc_var)
                #print(pc_var)
            
            #print(iteration)
            for par_var in gra[tar]:
                # attempt to reverse edges
                gra_temp[par_var].append(tar)
                gra_temp[tar].remove(par_var)
                underchecked = [tar]
                checked = []
                while underchecked:
                    if cyc_flag:
                        break
                    underchecked_copy = copy.deepcopy(underchecked)
                    for gra_par in underchecked_copy:
                        if gra_temp[gra_par]:
                            if par_var in gra_temp[gra_par]:
                                cyc_flag = True
                                break
                            else:
                                for key in gra_temp[gra_par]:
                                    if key not in checked:
                                        underchecked.append(key)
                        underchecked.remove(gra_par)
                        checked.append(gra_par)

                if cyc_flag:
                    cyc_flag = False
                else:
                    score_diff_temp = maximize_Qfunction2 (change_form_dic_to_matrix2(gra_temp),dataset,v0,hms_sample)-\
                                      maximize_Qfunction2 (change_form_dic_to_matrix2(gra),dataset,v0,hms_sample)
                    if score_diff_temp - diff > 1e-10:
                        diff = score_diff_temp
                        edge_candidate = [tar, par_var, 'r']
                        iteration=iteration+1
                gra_temp[par_var].remove(tar)
                #print(iteration)
                # attempt to delete edges
                score_diff_temp = maximize_Qfunction2 (change_form_dic_to_matrix2(gra_temp),dataset,v0,hms_sample)-\
                                  maximize_Qfunction2 (change_form_dic_to_matrix2(gra),dataset,v0,hms_sample)
                if (score_diff_temp - diff > -1e-10):
                    diff = score_diff_temp
                    edge_candidate = [tar, par_var, 'd']
                    iteration=iteration+1    
                gra_temp[tar].append(par_var)
                #print(iteration)
        # print(diff)
        # print(edge_candidate)
        if edge_candidate:
            if edge_candidate[-1] == 'a':
                gra[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                gra[edge_candidate[1]].append(edge_candidate[0])
                gra[edge_candidate[0]].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                gra[edge_candidate[0]].remove(edge_candidate[1])
                pc[edge_candidate[0]].append(edge_candidate[1])
                pc[edge_candidate[1]].append(edge_candidate[0])
    dag = {}
    for var in gra:
        dag[var] = {}
        dag[var]['par'] = gra[var]
        dag[var]['nei'] = []
    score=maximize_Qfunction2 (change_form_dic_to_matrix2(gra),dataset,v0,hms_sample)#zengjia
    #print(maximize_Qfunction2 (change_form_dic_to_matrix2(gra),dataset,v0,hms_sample))  
    #print(score)
    #print(gra)
    matrix=change_form_dic_to_matrix2(gra)
    return matrix,score


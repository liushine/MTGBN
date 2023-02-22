#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:36:13 2022

@author: shine
"""


import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout as layout
from networkx.utils import not_implemented_for
from networkx.algorithms import moral, complete_to_chordal_graph, chordal_graph_cliques
from random import randint as rd
import numpy as np

def makedecompgraph(DAG):
    # clique_graph = nx.Graph()
    dgraph = nx.from_numpy_matrix(DAG, create_using=nx.DiGraph)
    Moral_G = moral.moral_graph(dgraph)
    Chordal_G, _ = complete_to_chordal_graph(Moral_G)
    cliques = [(sorted(i)) for i in chordal_graph_cliques(Chordal_G)]
    separators= [None for _ in range(len(cliques))]
    separators[0]=[]
    for i in range(1,len(cliques)):
        cliques0=set(cliques[i-1])
        cliques1=set(cliques[i])
        separators[i]= (sorted(cliques0.intersection(cliques1)))
    G = np.empty((len(cliques),2),dtype=object)
    G[:,0]=cliques
    G[:,1]=separators
    return G

def DagMake(node):
    n=int(node)
    node=range(1,n+1)
    node=list(node)
    m=rd(n-1,(n*(n-1))/2+1)
    DAG=np.zeros((n,n))
    for i in range(0,m):
        p1=rd(1,n-1)
        p2=rd(p1+1,n)
        x=node[p1-1]
        y=node[p2-1]
        #l=np.random.randint(10,20)
        if DAG[x-1][y-1]!=0:
            continue
        else:
            DAG[x-1][y-1]=1
    return DAG

def symmetry(pc):

    for var in pc:
        pc_remove = []
        for par in pc[var]:
            if var not in pc[par]:
                pc_remove.append(par)
        if pc_remove:
            for par in pc_remove:
                pc[var].remove(par)
    return pc

def shrinkage(sigma,p,s):
    omega=np.linalg.inv(sigma)
    for i in range(0,p):
        for j in range (0,p):
                    if abs(omega[i,j])<=s:
                        omega[i,j]=0
    return omega
def structure(omega,p):
    G=omega
    for i in range(0,p):
        for j in range (0,p):
            if i==j:
                G[i,j]=0
            else:
                if G[i,j]!=0:
                    G[i,j]=1               
    return G
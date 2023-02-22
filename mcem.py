#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:21:45 2022

@author: shine
"""
import numpy as np
import copy
import pandas as pd
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from scipy.stats import invwishart, invgamma,multivariate_normal
from utils import DagMake
from hmc import hamilltonian_monte_carlo
from mmpc import mmpc
from mmhc import hc

# reparametrizing sigma_h to unconstrained parameters
sigma_h_to_V= tfb.Chain([  
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
    tfb.Invert(tfb.CholeskyOuterProduct()),
])
flatten = tfb.Chain([
    tfb.Invert(tfb.FillTriangular()),    
])


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


def mcem(m,v0,p,V,stepsize,iteration,dataset,initial_G,epsilon):   
    """Use Monte-Carlo Expectation-Maximation (MCEM) algorithm to search for the BN structures 
    Args:
         m: the number of tasks
         v0: degree of freedom in the inverse Wishart distribution 
         p: dimension 
         V: initial V
         stepsize: the step length of HMC
         iteration: the number of iterations of HMC
         dataset: observational datasets
         initial_G: initial BN structures
         epsilon: tolerance
    
    Returns:
          BN structures B
    """
    B=initial_G
    difference=1
    Q_score=0
    hms_sample=hamilltonian_monte_carlo(m,v0,p,B,dataset,V,stepsize,iteration)
    for i in range(1,m+1):
        data=pd.DataFrame(dataset[i])
        mmpc_B=symmetry(mmpc(data,prune = False, threshold =0.05))
        Bc=hc(mmpc_B,data,v0,hms_sample)
        B[i]=Bc[0]
        Q_score=Q_score+Bc[1]
    Q_score_past=Q_score
    while difference > epsilon:
        difference=0
        Q_score=0
        hms_sample=hamilltonian_monte_carlo(m,v0,p,B,dataset,V,stepsize,iteration)
        for i in range(1,m+1):
            data=pd.DataFrame(dataset[i])
            mmpc_B=symmetry(mmpc(data,prune = False, threshold =0.05))
            Bc=hc(mmpc_B,data,v0,hms_sample)
            B[i]=Bc[0]
            Q_score=Q_score+Bc[1]
        Q_score_current=Q_score
        difference=abs(Q_score_current- Q_score_past)
        Q_score_past=Q_score_current
        #print(difference)
        Q_score_past=Q_score_current 
        #print(Q_score_past) 
    return B
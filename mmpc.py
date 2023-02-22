#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:50:18 2022

@author: shine
"""

import numpy as np
from scipy.stats import pearsonr
import operator
import itertools
import pandas as pd
import time
def mmpc_forward(tar, pc, can, data, prune, threshold):
    # tar: target variable
    # pc: parents and children set of the target variable
    # can: candidate variable for the pc set of the target variable
    # data: input data
    # prune: prune the target variable in the candidate variable set for those variables which are independent with the target one
    # run until no candidate variable for current variable
    p_value = {}
    for can_var in can[tar]:
        p_value[can_var] = 0

    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test(p_value, tar, pc, can[tar], data)
        #print(p_value)
        # update pc set and candidate set
        pc, can = update_forward(p_value, tar, pc, can, prune, threshold)
    return pc, can
def mmpc_backward(tar, pc, can, data, prune, threshold):
    # tar: target variable
    # pc: parents and children set of the target variable
    # can: candidate variable for the pc set of the target variable
    # data: input data
    # prune: prune the target variable in the candidate variable set for those variables which are independent with the target one
    # transfer the variable in pc set to candidate set except the last one
    can[tar] = pc[0: -1]
    pc_output = []
    pc_output.append(pc[-1])
    can[tar].reverse()
    
    while can[tar]:
        # run conditional independence test between each candidate varialbe and target variable
        p_value = independence_test({}, tar, pc_output, can[tar], data)
        # update pc set and candidate set
        pc_output, can = update_backward(p_value, tar, pc_output, can, prune, threshold)

    return pc_output, can
def independence_test(p_value, tar, pc, can, data):
    for can_var in can:
        if can_var not in p_value.keys():
            p_value[can_var] = 0
        if len(pc) == 0:
            p_value[can_var]= max(pearsonr(data.loc[:, tar], data.loc[:, can_var])[1], p_value[can_var])   
        else:
            for r in range(len(pc)):
                for pc_sub in itertools.combinations(pc[0 : -1], r):
                    pc_con = list(pc_sub)
                    pc_con.append(pc[-1])
                    X_coef = np.linalg.lstsq(data.loc[:, pc_con], data.loc[:, tar], rcond=None)[0]
                    Y_coef = np.linalg.lstsq(data.loc[:,  pc_con], data.loc[:, can_var], rcond=None)[0]
                    residual_X = data.loc[:,tar] - data.loc[:, pc_con].dot(X_coef)
                    residual_Y = data.loc[:, can_var] - data.loc[:, pc_con].dot(Y_coef)
                    p_value[can_var]=max(pearsonr(residual_X, residual_Y)[1], p_value[can_var]) 
                    
    return p_value
def update_forward(p_value, tar, pc, can, prune, threshold):
    # add the variable with lowest p-value to pc set and remove it from the candidate set
    sorted_p_value = sorted(p_value.items(), key = operator.itemgetter(1))

    if sorted_p_value[0][1] <= threshold:
        pc.append(sorted_p_value[0][0])
        can[tar].remove(sorted_p_value[0][0])
        p_value.pop(sorted_p_value[0][0], None)

    # remove independent variables from candidate set
    independent_can = [x for x in sorted_p_value if x[1] > threshold]
    for ind in independent_can:
        can[tar].remove(ind[0])
        p_value.pop(ind[0])
        # prune the target variable from the candidate set of the candidate variable if they are independent
        if prune:
            if tar in can[ind[0]]:
                can[ind[0]].remove(tar)
    return pc, can

# pc and candidate set update function for backward phase
def update_backward(p_value, tar, pc, can, prune, threshold):

    # initialise the output candidate set
    can_output = []

    # signal of import variable
    sig_import = 1
    for can_var in can[tar]:
        if p_value[can_var] <= threshold:
            if sig_import:
                pc.append(can_var)
                sig_import = 0
            else:
                can_output.append(can_var)
        else:
            if prune:
                if tar in can[can_var]:
                    can[can_var].remove(tar)

    can[tar] = can_output

    return pc, can
def mmpc(data,prune = False, threshold = 0.05):    
    pc = {}
    # initialise the candidate set for variables
    can = {}
    for tar in data:
        can[tar] = list(data.columns)
        can[tar].remove(tar)

    start_time = time.time()
    # run MMPC on each variable
    # forward_time = 0
    # backward_time = 0
    for tar in data:
        # forward phase
        pc[tar] = []
        # start_time = time.time()
        pc[tar], can = mmpc_forward(tar, pc[tar], can, data, prune, threshold)
        if pc[tar]:
            pc[tar], can = mmpc_backward(tar, pc[tar], can, data, prune, threshold)
    return pc


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:31:04 2022

@author: shine
"""
import math
import numpy as np
import copy
import random
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from utils import makedecompgraph 

# reparametrizing sigma_h to unconstrained parameters
sigma_h_to_V= tfb.Chain([  
    tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
    tfb.Invert(tfb.CholeskyOuterProduct()),])
flatten = tfb.Chain([
    tfb.Invert(tfb.FillTriangular()),])


#Calculating the log-density fuction,the log-density including four parts
def log_multi_gamma(card,v):
    f=0
    for i in range(1,card+1):
        f=math.log(math.pi)+math.log(math.gamma(v-((i-1)/2)))
    f=(card*(card-1)/4)*f 
    return f

def log_iw_const(df,S,card):
    iwc=(df/2)*math.log(np.linalg.det(S))-log_multi_gamma(card,df/2)
    return iwc 

# The logdensity_part1 is used to calculate logh(\sigma_{h}+niSi, Bi)
def logdensity_part1(Gi,v0,ni,S,p):
    cliques=Gi[:,0]
    separators=Gi[:,1]
    numbercliques=len(cliques)     
    logdensity1=log_iw_const((v0+ni+len(cliques[0])-1),(S[cliques[0]][:,cliques[0]])/2,len(cliques[0]))
    if numbercliques >1:
           for i in range(1,numbercliques-1):
                logdensity1 = logdensity1+log_iw_const((v0+ni+len(cliques[i])-1),(S[cliques[i]][:,cliques[i]])/2,len(cliques[i]))+ \
               -log_iw_const((v0+ni+len(separators[i])-1),(S[separators[i]][:,separators[i]])/2,len(separators[i]))
    else :
           logdensity1=logdensity1
    return logdensity1  

# The logdensity_part2 is used to calculate (v0+ni)/2*log|\sigma_{h}+niSi| 
def logdensity_part2(v0,ni,S):
    density=np.linalg.det(S)*(v0+ni)/2
    logdensity2=math.log(density)
    return (logdensity2)

# The logdensity_part2 is used to calculate m*v0+p-i+2
def logdensity_part3(m,v0,p,V):
    logdensity3=0
    for i in range(p):
        logdensity3 =logdensity3+(m*v0+p-i+2)*V[i,i]
    return logdensity3

def logdensity_const(v0,ni,p):
    logdensityconst=(-ni*p)/2*math.log(math.pi)+log_multi_gamma(p,((v0+ni)/2))-log_multi_gamma(p,(v0/2))
    return logdensityconst

# The logdensity_sum is used to calculate log{P(V|B,D)}
def logdensity_sum(m,v0,p,G,dataset,V):
    logdensitysum=0
    for i in range(1,m+1):
        Gi=G[i]
        Gi=np.array(Gi,dtype='float32')
        Gi=makedecompgraph(Gi)
        data=dataset[i]
        data=np.array(data,dtype='float32')
        ni=data.shape[0]
        si=data.T.dot(data)/ni
        sigma_h=sigma_h_to_V.inverse(V).numpy()
        S=sigma_h+ni*si
        logdensitysum=logdensitysum+logdensity_part1(Gi,v0,ni,S,p)+logdensity_part2(v0,ni,S)
    logdensitysum=logdensitysum+logdensity_part3(m,v0,p,V)+logdensity_const(v0,ni,p)
    return logdensitysum


#Calculating the gradient of log-density fuction, which including three parts
def grad_L_part1(C_OR_S,S,L,v0,ni):
    I=np.identity(S.shape[0])
    A=(v0+ni+len(C_OR_S,)-1)/2
    B=np.transpose(I[C_OR_S])
    C=np.linalg.inv(S[ C_OR_S][:,C_OR_S])
    D=I[C_OR_S]
    E=L
    grad=2*np.tril((np.linalg.multi_dot([B,C,D,E])))
    grad=A*grad
    return grad

def gradient_part1(Gi,S,L,v0,ni):
    grad_L=[]
    cliques=Gi[:,0]
    separators=Gi[:,1]
    numbercliques=len(cliques)
    grad_L= grad_L_part1(cliques[0],S,L,v0,ni)
    if numbercliques >1:
           for i in range(1,numbercliques-1):
                grad_L=grad_L+grad_L_part1(cliques[i],S,L,v0,ni)-grad_L_part1(separators[i],S,L,v0,ni)
    else:
          grad_L=grad_L
    for i in range(grad_L.shape[0]):
         for j in range(grad_L.shape[0]) :
                if i==j:
                    grad_L[i,j]=grad_L[i,j]*L[i,j]
                else:
                    grad_L[i,j]=grad_L[i,j]
    grad_V1=grad_L
    return(grad_V1) 

def gradient_part2(S,L,v0,ni):  
    grad_L=(v0+ni)*(np.tril(np.linalg.inv(S)*L))
    for i in range(grad_L.shape[0]):
        for j in range(grad_L.shape[0]) :
                if i==j:
                    grad_L[i,j]=grad_L[i,j]*L[i,j]
                else:
                    grad_L[i,j]=grad_L[i,j]
    grad_V2=grad_L
    return(grad_V2)
       
def gradient_part3(m,v0,p,V):
    coff=0
    grad_V3 = copy.copy(V) 
    for i in range(p):
        for j in range(p): 
            coff += m*v0+p-i+1
            if i==j:
                grad_V3[i,i]=coff
            else:
                grad_V3[i,j]=0
    return grad_V3       
    
def gradient_sum(m,v0,p,G,dataset,V): 
    for i in range(1,m+1):
        Gi=G[i]
        Gi=np.array(Gi,dtype='float32')
        Gi=makedecompgraph(Gi)
        data=dataset[i]
        data=np.array(data,dtype='float32')
        ni=data.shape[0]
        si=data.T.dot(data)/ni
        sigma_h=sigma_h_to_V.inverse(V).numpy()
        S=sigma_h+ni*si
        cholesky=tfb.Invert(tfb.CholeskyOuterProduct())
        L=cholesky.forward(S).numpy()
        gradientsum=gradient_part1(Gi,S,L,v0,ni)+gradient_part2(S,L,v0,ni)
    gradientsum=gradientsum+gradient_part3(m,v0,p,V)
    return gradientsum 


def hamilltonian_monte_carlo(m,v0,p,G,dataset,V,stepsize,iteration):
    
    """Use the HMC to obtain a set of sample V through iterations
    Args:
        m: the number of subjects
        v0: degree of freedom in the inverse Wishart distribution 
        p: dimension 
        G: BN strctures
        dataset: obsevational datasets
        V: initial V
        stepsize: the step length
        iteration: the number of iterations
     
    Returns:
        samples of \sigma_{h}
        
    """
    accepted = 0.0
    sigma_hp= np.empty((iteration,p,p),dtype='float32')
    for n in range(iteration): 
            old_V = V
            old_energy =-logdensity_sum(m,v0,p,G,dataset,old_V)
            old_grad = -gradient_sum(m,v0,p,G,dataset,old_V )      
            new_V = copy.copy(old_V)  
            new_grad  = copy.copy(old_grad)
            M = np.eye(p, dtype=np.float32)
            pp=random.normalvariate(0, M)
            pp=np.tril(pp)
            vector_p= flatten.forward(pp).numpy() 
            H= np.dot(vector_p,vector_p)/2 + old_energy
            for tau in range(5):
                pp =pp-stepsize*new_grad/2
                new_V = new_V + stepsize*np.linalg.inv(M)@pp
                new_grad  = -gradient_sum(m,v0,p,G,dataset,new_V)
                pp= pp - stepsize*new_grad/2  
            vector_p=flatten.forward(pp).numpy()   
            new_energy = -logdensity_sum(m,v0,p,G,dataset,new_V)
            newH= np.dot(vector_p,vector_p)/2 + new_energy 
            dH= newH - H
            a = min(1,np.exp(-dH))
            u = np.random.uniform()
            if u < a: 
                v1=new_V 
                accepted = accepted + 1.0
            else:
                v1=old_V
            acceptance_prob = accepted/ iteration
            v1=sigma_h_to_V.inverse(v1).numpy()
            sigma_hp[n,:,:]=np.array(v1,'float32')
    #print(acceptance_prob)
    return (sigma_hp)
















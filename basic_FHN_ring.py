#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:20:48 2020
A system of Fitzhugh-Nagumo Neurons on a ring
@author: Sneha Kachhara
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import networkx as nx

#parameters
I = 0.23
a = -0.3
b = 0.8
tau = 20
tf = 1000 #time upto which to compute
time_step = 0.05
N = 2 #number of variables in the system (2 for FHN)

#function to integrate
def fitz_numo(state,t):
    v = state[0]
    w = state[1]
    dv = v*(1-v**2)-w+I
    dw = (v-a-b*w)/tau
    state = np.array([dv,dw])
    return state

#functions
def F(X,):
    #X is a 2-D array of v,w. Each row is a oscillator and each column a variable
    v = X.T[0] 
    w = X.T[1]
    dv = v*(1-v**2)-w+I
    dw = (v-a-b*w)/tau
    state = np.array([dv,dw])
    return state

def H(X,A,N3,N,e): #X is the set of all initial conditions
    S = np.zeros([N3,N]) #the coupling element
    for k in range(N3):
        s = np.asarray(np.sum(np.multiply(A[k],(X-X[k]).T),axis=1)).T
        S[k][0],S[k][1]=s[0]
    return S

def system_of_FHN(X0,t,A,G,N3,N,e):
    X0 = X0.reshape(N3,N) #to get back initial values in respective dimensions
    inherent_dynamics = F(X0).T #returns a (N3,N) matrix
    coupling_factor = e*G[0] #dimensions (1,N)
    coupling_dynamics = coupling_factor*H(X0,A,N3,N,e)
    X_f = inherent_dynamics+coupling_dynamics
    return X_f.flatten()

#to see the ring
def see_graphs(G,spring=False):
    plt.figure()
    if spring==True:
        pos = nx.spring_layout(G,iterations=100)
    else:
        pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos)
    return None

#See results
def plot_results(N3,e,t_values,X_final,ts_plots = False):
    if ts_plots==True:
        plt.figure().set_size_inches(15,10)
        plt.suptitle('time series plots')
        for osc in range(0,N3):
            osc_V = osc*2
            osc_W = osc*2+1
            plt.subplot(211)
            plt.plot(t_values,X_final[:,osc_V],marker='o',markersize=0.5,linewidth=0.2,alpha=0.2,color='k')
            plt.ylabel('v')
            plt.subplot(212)
            plt.plot(t_values,X_final[:,osc_W],marker='o',markersize=0.5, linewidth=0.2,alpha=0.2,color='k')
            plt.ylabel('w')
        plt.xlabel('time')
        plt.title('coupling strength: '+str(e)+', I_ext: '+str(I))
        plt.tight_layout()
    #spatio-temporal plots
    plt.figure().set_size_inches(10,20)
    plt.subplot(121)
    plt.title('V')
    plt.imshow(X_final[-10000::10,::2],origin='lower',aspect='auto')
    plt.ylabel('time')
    plt.xlabel('oscillator number')
    plt.subplot(122)
    plt.title('W')
    plt.imshow(X_final[-10000::10,1::2],origin='lower',aspect='auto')
    plt.ylabel('time')
    plt.xlabel('oscillator number')
    #plt.colorbar()
    #plt.tight_layout()
    return None

#by default takes a ring of FHN
def create_system(N3=50,e=0.1,simple_ring=True):
    #initial state
    X_ini = np.random.rand(N3,N)#range of distribution of initial conditions
    gr = nx.cycle_graph(N3)
    A = nx.to_numpy_matrix(gr)
    #coupling matrix
    G = np.zeros([N,N]) #controls which variables to be coupled.
    G[0,0] = 1 #indicates only x variables are coupled.
    #get even distribution of both positive and negative initial values.
    #making half of them negative
    X_ini[:int(N3/2),:] = -1*X_ini[:int(N3/2),:]
     
    t_values = np.arange(0,1000,0.05)#times at which to compute the solution
    X_final = scipy.integrate.odeint(func=system_of_FHN,y0=X_ini.flatten(),t=t_values,args=(A,G,N3,N,e))
    plot_results(N3,e,t_values,X_final)
    return X_final

def create_system(A,e=0.1):
    N3 = len(A)
    #initial state
    X_ini = np.random.rand(N3,N)#range of distribution of initial conditions
    #coupling matrix
    G = np.zeros([N,N]) #controls which variables to be coupled.
    G[0,0] = 1 #indicates only x variables are coupled.
    #get even distribution of both positive and negative initial values.
    #making half of them negative
    X_ini[:int(N3/2),:] = -1*X_ini[:int(N3/2),:]
     
    t_values = np.arange(0,1000,0.05)#times at which to compute the solution
    X_final = scipy.integrate.odeint(func=system_of_FHN,y0=X_ini.flatten(),t=t_values,args=(A,G,N3,N,e))
    plot_results(N3,e,t_values,X_final)
    return X_final
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:38:21 2020

@author: Sneha Kachhara
"""
import basic_FHN_ring as fhn
import networkx as nx

#you can give a custom adjacency matrix as well.
Num = 100 #number of oscillators
nbrs = 4 #number of neighbours for Watts-Strogatz graph
prob = 0.2 #probability of rewiring for Watts-Strogatz graph
gr = nx.watts_strogatz_graph(100,4,0.1)
A = nx.to_numpy_matrix(gr)

coupling_strength = 0.1
fhn.create_system(A,e=coupling_strength)
fhn.see_graphs(gr)
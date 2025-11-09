#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:48:01 2018

Utils: functions to support jupyter notebooks for TDS

@author: Óscar Barquero Pérez y Rebeca Goya Esteban
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.signal as sig
import scipy.io.wavfile as wf
import os


def energia2(s,w):
    """
    Function that computes localized energy from a signal s, given a window w.
    w should be os length smaller than s
    """
    s = s - np.mean(s)
    Energy = []
    #for over the signal in steps of len(w)
    for n in range(0,len(s)-len(w),len(w)):
    #for n in range(0,len(s)-len(w)):
       
        #print(n,':',n+len(w))
        #print(len(s))
        trama = s[n:n+len(w)] * w #actual windowed segment
        
        Energy.append(np.sum(trama**2)/len(w))
        
    Energy = np.array(Energy)
    return Energy/np.max(Energy) 


def zcr2(s,w):
    """
    Function that computes l zero-crossing rate from a signal, given a window w.
    w should be os length smaller than sig
    """
    zcr_a = []
    
    for n in range(0,len(s)-len(w),len(w)):
        trama = s[n:n+len(w)]
        
        zcr_aux = np.sum((0.5/len(trama))*(np.abs(np.sign(trama[1:])-np.sign(trama[:-1]))))
        
        zcr_a.append(zcr_aux)
        
    return zcr_a

     

        
#----------------- ooooooo----------------------------------
# The following code is from https://github.com/pierre-rouanet/dtw/blob/master/dtw.py

from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist

def dtw_Endp(ref,test,dist = 'euclidean',omit_left=0,omit_right = 0,warp = 1):
    """
     TO_DO Implementing dtw with Sakoe restrictions
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    refs:
        : Theodoridis
        : https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
        : https://nipunbatra.github.io/blog/2014/dtw.html
    """
    
    #check input vector dimensions
    if ndim(ref) == 1:
        ref = ref[:,np.newaxis]
    if ndim(test) == 1:
        test = test[:,np.newaxis]
        
    I = len(ref)
    J = len(test)
    

    
    #1) 2d matrix to compute distances between all pairs of x and y
    
    NodeCost = cdist(test,ref,metric = dist)
    
    
    
    #2) Accumulated cost
    D = np.zeros(np.shape(NodeCost))
    
    
    #Initilization 1st column
    #TO_DO: use by defualt parameters
    
    for n in range(omit_left+1): #due to the left endpoint constraint
        D[n,0] = NodeCost[n,0]
        
    for n in range(omit_left+1,len(test)): #due to the left endpoint constraint
        D[n,0] = D[n-1,0] + NodeCost[n,0]
        
    #Initizalization 1st row
    for n in range(1,len(ref)):
        D[0,n] = D[0, n-1] + NodeCost[0,n]
        
    
    #we started at zero
    D[0,0] = NodeCost[0,0]
    #inizialization
  
    #Compute the distance for every pair of points
    #double-for over length of ref and test:
    for i in range(1, len(test)):
        for j in range(1, len(ref)):
            #Sakoe-Chiba local path constraints
            D[i, j] = min(D[i-1, j-1], D[i-1, j], D[i, j-1]) + NodeCost[i, j]
    
    #Backtracking applying right endpoint constraint
   # MatchingCost = np.inf
    """
    for k in range(len(test),len(test)-omit_right,-1):
        print(k)
        path, cost = back_tracking_path_cost(ref,test,D,NodeCost,start_)
    """
    #back_tracking to obtain de path
    path, cost, matching_cost = back_tracking_path_cost_2(ref,test,D,NodeCost,start_node_row = len(test)-omit_right-1,start_node_col = len(ref)-1)
    
    return NodeCost, D, path, cost, matching_cost

def dtw(ref,test,dist = 'euclidean', warp = 1):
    """
     TO_DO Implementing dtw with Sakoe restrictions
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    refs:
        : Theodoridis
        : https://github.com/pierre-rouanet/dtw/blob/master/dtw.py
        : https://nipunbatra.github.io/blog/2014/dtw.html
    """
    
    #check input vector dimensions
    if ndim(ref) == 1:
        ref = ref[:,np.newaxis]
    if ndim(test) == 1:
        test = test[:,np.newaxis]
    
    #1) 2d matrix to compute distances between all pairs of x and y
    
    NodeCost = cdist(test,ref,metric = dist)
    
    
    
    #2) Accumulated cost
    accumulated_cost = np.zeros(np.shape(NodeCost))
    
    
    #we started at zero
    accumulated_cost[0,0] = NodeCost[0,0]
    #inizialization
    for i in range(1, len(ref)):
        accumulated_cost[0,i] = NodeCost[0,i] + accumulated_cost[0, i-1]   
    for i in range(1, len(test)):
        accumulated_cost[i,0] = NodeCost[i, 0] + accumulated_cost[i-1, 0]   
    #double-for over length of ref and test:
    for i in range(1, len(test)):
        for j in range(1, len(ref)):
            #Sakoe-Chiba local path constraints
            accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + NodeCost[i, j]
    
    #back_tracking to obtain de path
    path, cost, matching_cost = back_tracking_path_cost(ref,test,accumulated_cost,NodeCost)
    
    return NodeCost, accumulated_cost, path, cost, matching_cost

def back_tracking_path_cost_2(ref,test,accumulated_cost, distances,start_node_row = None,start_node_col = None):
    """
    Performs backtracking on a matrix of node predecessors and returns the
    extracted best path starting from node .
    """
    
    if start_node_row == None:
        start_node_row = len(test)-1
    if start_node_col == None:
        start_node_col = len(ref)-1
        
    cost = 0
    path = [[start_node_row, start_node_col]]
    test_i = start_node_row # test
    ref_j = start_node_col#j ref
    while test_i>0 and ref_j>0:
        if test_i==0:
            ref_j = ref_j - 1
        elif ref_j==0:
            test_i = test_i - 1
        else:
            if accumulated_cost[test_i-1, ref_j] == min(accumulated_cost[test_i-1, ref_j-1], accumulated_cost[test_i-1, ref_j], accumulated_cost[test_i, ref_j-1]):
                test_i = test_i - 1
            elif accumulated_cost[test_i, ref_j-1] == min(accumulated_cost[test_i-1, ref_j-1], accumulated_cost[test_i-1, ref_j], accumulated_cost[test_i, ref_j-1]):
                ref_j = ref_j-1
            else:
                test_i = test_i - 1
                ref_j= ref_j- 1
        path.append([test_i, ref_j])
        
    path.append([0,0])
    
    for [test_i, ref_j] in path:
        cost = cost + distances[test_i, ref_j]
      
    #cost according to Theodoridis implementation
    matching_cost = (cost - distances[0,0]) / (len(path) -1)
    return path, cost, matching_cost

    
def back_tracking_path_cost(ref,test,accumulated_cost, distances):
    """
    Performs backtracking on a matrix of node predecessors and returns the
    extracted best path starting from node .
    """
    
    cost = 0
    path = [[len(test)-1,len(ref)-1]]
    test_i = len(test)-1 # test
    ref_j = len(ref)-1 #j ref
    while test_i>0 and ref_j>0:
        if test_i==0:
            ref_j = ref_j - 1
        elif ref_j==0:
            test_i = test_i - 1
        else:
            if accumulated_cost[test_i-1, ref_j] == min(accumulated_cost[test_i-1, ref_j-1], accumulated_cost[test_i-1, ref_j], accumulated_cost[test_i, ref_j-1]):
                test_i = test_i - 1
            elif accumulated_cost[test_i, ref_j-1] == min(accumulated_cost[test_i-1, ref_j-1], accumulated_cost[test_i-1, ref_j], accumulated_cost[test_i, ref_j-1]):
                ref_j = ref_j-1
            else:
                test_i = test_i - 1
                ref_j= ref_j- 1
        path.append([test_i, ref_j])
        
    path.append([0,0])
    
    for [test_i, ref_j] in path:
        cost = cost + distances[test_i, ref_j]
     
    #cost according to Theodoridis implementation
    matching_cost = (cost - distances[0,0]) / (len(path) -1)
    return path, cost, matching_cost
        

#plot cost functions
def distance_cost_plot(distances,path):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar();
    path_y = [point[0] for point in path]
    path_x = [point[1] for point in path]
    plt.plot(path_x,path_y)





"""
#Working on DTW

#p = np.array([-8,-4,0,4,0,-4])
#t = np.array([0,-8,-4,0,4,0,-4,0,0])
#p = np.array([0.9575 ,  0.9649 ,  0.1576  , 0.9706])
#t = np.array([0.5377,1.8339,-2.2588,0.8622,0.3188,-1.3077,-0.4336,0.3426])

#NodeCost, accumulated_cost, path, cost, matching_cost = dtw_Endp(p,t,dist = 'euclidean',omit_left = 0,omit_right = 0, warp = 1)

#distance_cost_plot(NodeCost)

#plt.figure()

#distance_cost_plot(accumulated_cost)
#path_y = [point[0] for point in path]
#path_x = [point[1] for point in path]
#plt.plot(path_x,path_y)
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:14:17 2018

@author: victorchomel
"""

import numpy as np
import heapq
import random
import time

def algoGlouton(n, rcapt, rcom):
    covered = np.zeros((n, n))
    nodes = []
    
    nodes.append((0, 0))
    
    covered = nodesCover((0,0), rcapt, covered)
    
#    c = 0
    
    while(not testFin(covered)):
#        if c > 10:
#            break
#        else : 
#            c += 1
        
        priorq = []
        
        for node in nodes :
            nodeNeighbours = comNeighbours(node, rcom, n)
#            print(nodeNeighbours)
            for nodeNeighbour in nodeNeighbours :
                heapq.heappush(priorq, (- evalNewNode(nodeNeighbour, rcapt, covered), nodeNeighbour))
#            print(priorq)
        listNewNodes = []
        initval, newNode = heapq.heappop(priorq)
        val = initval
        while(val == initval):
            listNewNodes.append(newNode)
            val, newNode = heapq.heappop(priorq)
            
        newNode = random.choice(listNewNodes)
        nodes.append(newNode)
        nodesCover(newNode, rcapt, covered)
        
#        print(newNode)
        
#    visu(nodes, n)
#    print(len(nodes))
    
    return (len(nodes), linksNumber(nodes, rcapt, n))
    
def testFin(covered):
    p = 1
    n = covered.shape[0]
    for i in range(n):
        for j in range(n):
            p = p * covered[i, j]
    if p == 0 :
        return False
    else :
        return True

#But trouver quel voisin ajouter
def glouton(iterations, n, rcapt, rcom):
    start = time.time()
    
    res = []
    for i in range(iterations):
        res.append(algoGlouton(n, rcapt, rcom))
    print(res)
    print(min(res))
    
    print(sorted(res))
    
    end = time.time()
    
    print(end - start)
        
    
    
    
def evalNewNode(node, rcapt, covered):
    eval = 0
    n = covered.shape[0]
    for i in range(node[0] - rcapt, node[0] + rcapt + 1):
        for j in range(node[1] - rcapt, node[1] + rcapt + 1):
            if(i >= 0 and i< n):
                if(j >= 0 and j< n):
                    if((i - node[0])**2 + (j - node[1])**2 <= rcapt**2):
                        eval += 1 - covered[i,j]
    return eval
                    
    
def comNeighbours(node, rcom, n):
    nlist = []
    for i in range(node[0] - rcom, node[0] + rcom + 1):
        for j in range(node[1] - rcom, node[1] + rcom + 1):
            if(i >= 0 and i< n):
                if(j >= 0 and j< n):
                    if((i - node[0])**2 + (j - node[1])**2 <= rcom**2):
                        nlist.append((i,j))                    
    return nlist
    
#Update covered
def nodesCover(node, rcapt, covered):
    n = covered.shape[0]
    
    for i in range(node[0] - rcapt, node[0] + rcapt + 1):
        for j in range(node[1] - rcapt, node[1] + rcapt + 1):
            if(i >= 0 and i< n):
                if(j >= 0 and j< n):
                    if((i - node[0])**2 + (j - node[1])**2 <= rcapt**2):
                        covered[i,j] = 1
                        
#    print(covered)
    return covered

def visu(nodes, n):
    matrice = np.zeros((n, n))
    for node in nodes:
        matrice[node] = 1 
    print(matrice)
    
def evalFunction(nodes, n): 
    
    return

def linksNumber(nodes, rcapt, n):
    c = 0
    for node in nodes:
        for i in range(node[0] - rcapt, node[0] + rcapt + 1):
            for j in range(node[1] - rcapt, node[1] + rcapt + 1):
                if(i >= 0 and i< n):
                    if(j >= 0 and j< n):
                        if((i - node[0])**2 + (j - node[1])**2 <= rcapt**2):
                            c += 1
                            
    return c

def algoGloutonInv(n rcapt, rcom):
    covered = np.ones((n, n))
    
def canBeRemoved(node, nodesMat, rcapt):
    n = nodesMat.shape[0]
    nodeNeighbours = comNeighbours(node, rcapt, n)
#            print(nodeNeighbours)
    testCov = 1
    for nodeNeighbour in nodeNeighbours :
        cov = 0
        nodeNeighOfNeighs = comNeighbours(nodeNeighbour, rcapt, n)
        for nodeNeighOfNeigh in nodeNeighOfNeighs :
            cov += nodesMat[nodeNeighOfNeigh]
        testCov *= cov
        
    testMst = 1
    
        
    if testCov > 0 :
        
            
        
    
        
    
#algoGlouton(10, 1, 2)
glouton(50, 20, 1, 2)
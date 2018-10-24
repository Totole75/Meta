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


    
    
def voisinage(k, nodes):
    nodes = remove(k, nodes)
    nodes = complete(nodes)
    
    return nodes

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

def captorNeighbours(node, rcom, n, nodesMat):
    nlist = []
    for i in range(node[0] - rcom, node[0] + rcom + 1):
        for j in range(node[1] - rcom, node[1] + rcom + 1):
            if(i >= 0 and i< n):
                if(j >= 0 and j< n):
                    if((i - node[0])**2 + (j - node[1])**2 <= rcom**2):
                        if(nodesMat[i, j] == 1):
                            nlist.append((i,j))                    
    return nlist

#Liste des index des capteurs à rcom du noeud
def indexCaptNeighbours(nodeIndex, rcom, n, nodes):
    nlist = []
#    print(node)
    for i in range(len(nodes)) :
        if not i == nodeIndex :
            if((nodes[i][0] - nodes[nodeIndex][0])**2 + (nodes[i][1] - nodes[nodeIndex][1])**2 <= rcom**2):
                nlist.append(i)                    
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
    
def evalFunction1(nodesMat, rcapt, rcom, n):
    nodes = []
    for i in range(n):
        for j in range(n):
            if(nodesMat[i, j] == 1):
                nodes.append((i, j))
    coverNumber = np.zeros((n,n))
    for node in nodes:
        capNeighbours = comNeighbours(node, rcapt, n)
        for nodeNeigh in capNeighbours :
            coverNumber[nodeNeigh]+=1
    m = np.mean(coverNumber)
    
    evaluation = []
    
    for node in nodes:
        val = 0
        capNeighbours = comNeighbours(node, rcapt, n)
        for nodeNeigh in capNeighbours :
            val += 1/coverNumber[nodeNeigh]
        val -= len(capNeighbours)/m
        
        heapq.heappush(evaluation, (val, node))
        
         
    return evaluation

def evalFunction2(nodesMat, rcapt, rcom, n):
    nodes = []
    for i in range(n):
        for j in range(n):
            if(nodesMat[i, j] == 1):
                nodes.append((i, j))
    evaluation = []
    for node in nodes:
        val = - len(captorNeighbours(node, rcom, n, nodesMat))

        heapq.heappush(evaluation, (val, node))
        
    return evaluation


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

#alpha : randomness
def algoGloutonInv(n, rcapt, rcom, alpha):
    nodesMat = np.ones((n, n))
    
    while(True):
        evaluation = evalFunction2(nodesMat, rcapt, rcom, n)
#        print("Evaluation", evaluation)
        cantRemove = True
        while(cantRemove and len(evaluation) > 0):
            rand = random.random()
            if rand < alpha:
                toRemove = evaluation.pop(random.randint(0, len(evaluation)-1))[1]
            else :
                toRemove = heapq.heappop(evaluation)[1]
#            print(len(evaluation))
            cantRemove = not canBeRemoved(toRemove, nodesMat,rcapt, rcom)
#            print(cantRemove)
        if len(evaluation) == 0 :
#            print(nodesMat)
            
#            print(np.sum(nodesMat))
            return np.sum(nodesMat)
#            return nodesMat
        else :
#            print(toRemove)
#            print(nodesMat)
            nodesMat[toRemove] = 0
            
    
            
            #On peut enregistrer ceux qui ne peuvent pas être sortis
            
def gloutonInv(iterations, n, rcapt, rcom, alpha):
    start = time.time()
    
    res = []
    for i in range(iterations):
        res.append(algoGloutonInv(n, rcapt, rcom, alpha))
#    print(res)
    print(min(res))
    
    print(sorted(res))
    
    end = time.time()
    
    print(end - start)
            
        
    
def canBeRemoved(node, nodesMat, rcapt, rcom):
    nodesMat[node]=0
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
                    
    nodes = []
    for i in range(n):
        for j in range(n):
            if(nodesMat[i, j] == 1):
                nodes.append((i, j))
            
    nodesMat[node]=1
    if testCov > 0 and testConnexite(nodes, rcom, n):
        return True
    return False
        
def testConnexite(nodes, rcom, n):
    colored = [0]*len(nodes)
    
    colored[0] = 1
    toExplore = [0]
    while(len(toExplore) > 0):
        nodeIndex = toExplore.pop()
        for indexCaptNeighbour in indexCaptNeighbours(nodeIndex, rcom, n ,nodes):
            if(colored[indexCaptNeighbour] == 0):
                colored[indexCaptNeighbour] = 1
                toExplore.append(indexCaptNeighbour)
    for b in colored :
        if b == 0:
            return False
    return True
    
    
#algoGlouton(10, 1, 2)
#glouton(50, 20, 1, 2)
#algoGloutonInv(5, 1, 2)
#gloutonInv(5, 10, 1, 2, 0.2)
for r in np.arange(0.05, 0.5, 0.05):
    gloutonInv(5, 10, 1, 2, r)
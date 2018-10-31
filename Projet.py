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
from sklearn.metrics.pairwise import pairwise_distances
import meta
import tool_box
import copy

##### Pour la GRILLE uniquement ######

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

def testFinGrille(covered):
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
            

        

    
    
#algoGlouton(10, 1, 2)
#glouton(150, 6, 1, 2)
#algoGloutonInv(5, 1, 2)
#gloutonInv(5, 10, 1, 2, 0.2)
#for r in np.arange(0.05, 0.5, 0.05):
#    gloutonInv(5, 10, 1, 2, r)




##### Pour le RESEAU #####

#On suppose que l'on a en entrée :
#   - nodes = array la liste des points [(xi, yi)]
#   - matCap : matrice d'adjacence des sommets pour rcapt
#   - matCom : matrice d'adjacence des sommets pour rcom

#n bord de la grille
def nodesGrille(n):
    nodes = np.zeros((n**2, 2))
    for i in range(n):
        for j in range(n):
            nodes[i + n*j]= (i, j)
    
    return nodes

def matAdjGrille(n, rayon):
    matAdj = np.zeros((n**2,n**2))
    for i0 in range(n):
        for j0 in range(n):
            for i1 in range(i0 - rayon, i0 + rayon + 1):
                for j1 in range(j0 - rayon, j0 + rayon + 1):
                    if(i1 >= 0 and i1< n):
                        if(j1 >= 0 and j1< n):
                            if((i1 - i0)**2 + (j1 - j0)**2 <= rayon**2):
                                matAdj[i0 + n*j0, i1 + n*j1] = 1
    return matAdj
            
    

def algoGloutonReseau(nodes, matCom, matCap, capteurs_init = []):
    n = nodes.shape[0]
    
    
    capteurs = copy.copy(capteurs_init)
    
    covered = np.zeros(n)
    
    if len(capteurs) > 0:
        covered = alreadyCovered(capteurs, matCap, n)
    
    if not 0 in capteurs :
        capteurs.append(0)
        
        covered = captCover(0, covered, matCap)
    
    while(not testFinReseau(covered)):

        
        priorq = []
        
        for capt in capteurs :
            
            for captNeighbour in range(n):
                if(matCom[capt, captNeighbour] == 1):
                    heapq.heappush(priorq, (- evalNewCapt(captNeighbour, covered, matCap), captNeighbour))

        listNewCapt = []
        initval, newCapt = heapq.heappop(priorq)
        val = initval
        while(val == initval):
            listNewCapt.append(newCapt)
            val, newCapt = heapq.heappop(priorq)
            
        newCapt = random.choice(listNewCapt)
        capteurs.append(newCapt)
        captCover(newCapt, covered, matCap)
        
#        print(testFinReseau(covered))
        
#        print(newNode)
        
#    visu(nodes, n)
#    print(capteurs)
    
    return capteurs

def captCover(capt, covered, matCap):
    n = covered.shape[0]
    
    for captNeighbour in range(n):
        if(matCap[capt, captNeighbour] == 1):
            covered[captNeighbour] = 1
            
#    print(covered)
    return covered

def testFinReseau(covered):
    p = 1    
    for i in covered :
        p *= i
        
    if p > 0:
        return True
    return False

def evalNewCapt(capt, covered, matCap):
    evaluation = 0
    n = covered.shape[0]
    
    for captNeighbour in range(n):
        if(matCap[capt, captNeighbour] == 1):
            evaluation += 1 - covered[captNeighbour] 
    return evaluation



def gloutonGrille(iterations, n, rcom, rcapt):
    start = time.time()
    
    res = []
    grille = nodesGrille(n)
    matAdjCom = matAdjGrille(n, rcom)
    matAdjCap = matAdjGrille(n, rcapt)
    for i in range(iterations):
        res.append(len(algoGloutonReseau(grille, matAdjCom, matAdjCap)))
    print(res)
    print(min(res))
    
    print(sorted(res))
    
    end = time.time()
    
    print(end - start)
    
#gloutonGrille(1, 40, 2, 3)


def voisinagePasConnexe(k, capteurs, coord_nodes, matCom, matCap, matDist, rcom):
#    print(capteurs)
    capteurs = removeK(k, capteurs)
#    print("Removed", len(capteurs))
    capteurs = algoGloutonReseau(coord_nodes, matCom, matCap, capteurs)

    capteurs = meta.reconstruction(coord_nodes, matDist, capteurs, matCom, rcom)
#    print("Complete", len(capteurs))

    capteurs = removeConnexe(capteurs, coord_nodes, matCom, matCap)
    
    return capteurs

def removeK(k, capteurs):
    if k < len(capteurs):
        toRemove = np.random.choice(capteurs, k, replace = False)
        
        for captRemov in toRemove:
            capteurs.remove(captRemov)
            
        
    return capteurs

    
def voisinageConnexe(k, capteurs, nodes, matCom, matCap):
#    print(capteurs)
    capteurs = addK(k, capteurs, nodes.shape[0])
#    print("Removed", len(capteurs))

    tool_box.trace(nodes, capteurs, matCom, matCap)

    capteurs = removeConnexe(capteurs, nodes, matCom, matCap)
#    print("Complete", len(capteurs))
    
    return capteurs


def addK(k, capteurs, n):
    cpt = 0
    while(cpt < k):
        newCapt = np.random.randint(0, n - 1)
#        print("Capteur", newCapt)
        
        if not (newCapt in capteurs) :
            capteurs.append(newCapt)
            cpt += 1
    
    return capteurs

def removeConnexe(capteurs, nodes, matCom, matCap):
    while True:
        toRemove = []
        capteursBis = copy.deepcopy(capteurs)
        for capt in capteurs:
            capteursBis.remove(capt)
#            print("Taillle capteurs", len(capteurs))
#            print("Taillle capteursBis", len(capteursBis))
#            print("Taillle toRemove", len(toRemove))
            if canBeRemoved(capt, capteursBis, matCom, matCap, nodes.shape[0]):
                toRemove.append(capt)
            capteursBis.append(capt)

        if(len(toRemove) > 0):
            capteurs.remove(np.random.choice(toRemove))
            
        else:
            break
    return capteurs
        
        
        
def canBeRemoved(capt, capteurs, matCom, matCap, n):
    if not testConnexite(capteurs, matCom):
        return False
    
    testCov = 1
    for nodeNeighbour in range(n) :
        
        if(matCap[nodeNeighbour, capt] == 1):
            cov = 0
            for captNeighofNeigh in range(n):
                if(matCap[nodeNeighbour, captNeighofNeigh] == 1):
                    if captNeighofNeigh in capteurs :
                        cov += 1

            testCov *= cov
                    
    
    if testCov > 0 :
        return True
    return False
    
    


def complete(capteurs, nodes, matCom, matCap, rcom, rcapt):
    n = nodes.shape[0]
#    print(capteurs)
    
    covered = alreadyCovered(capteurs, matCap, rcapt, n)
    
#    print(sum(covered))
    
    while(not testFinReseau(covered)):

        
        priorq = []
        
        for capt in capteurs :
            
            for captNeighbour in range(n):
                if(matCom[capt, captNeighbour] == 1):
                    heapq.heappush(priorq, (- evalNewCapt(captNeighbour, rcapt, covered, matCap), captNeighbour))

        listNewCapt = []
        initval, newCapt = heapq.heappop(priorq)
        val = initval
        while(val == initval):
            listNewCapt.append(newCapt)
            val, newCapt = heapq.heappop(priorq)
            
        newCapt = random.choice(listNewCapt)
        capteurs.append(newCapt)
        covered = captCover(newCapt, rcapt, covered, matCap)
        
#        print(sum(covered))
#        print(covered)
        
    return capteurs

def alreadyCovered(capteurs, matCap, n):
    covered = np.zeros(n)
    
    for capt in capteurs :
        covered = captCover(capt, covered, matCap)
        
    return covered

def gloutonReseau(iterations, path, rcom, rcapt):
    start = time.time()
    
    res = []
    nodes = meta.read_data(path)
    distance_matrix = pairwise_distances(nodes)
    matAdjCom, matAdjCap = matrices_adj(distance_matrix, rcom, rcapt)
    for i in range(iterations):
        res.append(len(algoGloutonReseau(nodes, matAdjCom, matAdjCap, rcom, rcapt)))
    print(res)
    print(min(res))
    
    print(sorted(res))
    
    end = time.time()
    
    print(end - start)
    
    
def matrices_adj(distance_matrix, rcom, rcapt):
    n = distance_matrix.shape[0]
    
    matAdjCom = np.zeros((n, n))
    
    matAdjCap = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            d = distance_matrix[i, j]
            if(d <= rcom):
                matAdjCom[i, j] = 1
            if(d <= rcapt):
                matAdjCap[i, j] = 1
                
                
    return matAdjCom, matAdjCap

def algoVoisinageGloutonReseauPasConnexe(path, k, p, rcom, rcapt):
    start = time.time()
    nodes, distance_matrix = tool_box.read_data(path)
    matAdjCom, matAdjCap = matrices_adj(distance_matrix, rcom, rcapt)
    
    capteurs = algoGloutonReseau(nodes, matAdjCom, matAdjCap)
#    print(capteurs)
    
    vals = [len(capteurs)]
    
    minValue = nodes.shape[0]
    
    
    minCapteurs = capteurs
    
    for i in range(p):
        print(vals)
        capteurs = voisinagePasConnexe(k, capteurs, nodes, matAdjCom, matAdjCap, distance_matrix, rcom)
        
        vals.append(len(capteurs))
        
        if(len(capteurs) < minValue):
            
            minValue = len(capteurs)
            minCapteurs = copy.deepcopy(capteurs)

    print(vals)
    
    end = time.time()
    
    print(end - start)
    
    print(minValue)
    print(minCapteurs)
    print(len(minCapteurs))
        
    tool_box.trace(nodes, minCapteurs, rcom, matAdjCap)
    
    return minCapteurs

def algoVoisinageGloutonReseau(path, k, p, rcom, rcapt):
    start = time.time()
    nodes = meta.read_data(path)
    distance_matrix = pairwise_distances(nodes)
    matAdjCom, matAdjCap = matrices_adj(distance_matrix, rcom, rcapt)
    
    capteurs = algoGloutonReseau(nodes, matAdjCom, matAdjCap, rcom, rcapt)
#    print(capteurs)
    
    vals = [len(capteurs)]
    
    minValue = nodes.shape[0]
    
    
    minCapteurs = capteurs
    
    for i in range(p):
        print(vals)
        capteurs = voisinageConnexe(k, capteurs, nodes, matAdjCom, matAdjCap)
        
        vals.append(len(capteurs))
        
        if(len(capteurs) < minValue):
            
            minValue = len(capteurs)
            minCapteurs = copy.deepcopy(capteurs)

    print(vals)
    
    end = time.time()
    
    print(end - start)
    
    print(minValue)
    print(minCapteurs)
    print(len(minCapteurs))
        
    tool_box.trace(nodes, minCapteurs, matAdjCom, matAdjCap)
    
    return minCapteurs

def algoVoisinageGloutonGrille(n, k, p, rcom, rcapt):
    start = time.time()
    grille = nodesGrille(n)
    matAdjCom = matAdjGrille(n, rcom)
    matAdjCap = matAdjGrille(n, rcapt)
    
    capteurs, covered = algoGloutonReseau(grille, matAdjCom, matAdjCap, rcom, rcapt)
#    print(capteurs)
    
    vals = [len(capteurs)]
    
    for i in range(p):
#        print(vals)
        capteurs = voisinageConnexe(k, capteurs, grille, matAdjCom, matAdjCap)
        
        vals.append(len(capteurs))
#        print("Capteurs", capteurs)
        
    print(vals)
    print(min(vals))
    
    end = time.time()
    
    print(end - start)
    
    return vals
    
def algoVoisinageGloutonGrillePasConnexe(n, k, p, rcom, rcapt):
    start = time.time()
    grille = nodesGrille(n)
    distance_matrix = pairwise_distances(grille)
    matAdjCom = matAdjGrille(n, rcom)
    matAdjCap = matAdjGrille(n, rcapt)
    
    capteurs = algoGloutonReseau(grille, matAdjCom, matAdjCap)
#    print(capteurs)

    minValue = n**2
    
    minCapteurs = capteurs
    
    vals = [len(capteurs)]
    
    for i in range(p):
#        print(vals)
        capteurs = voisinagePasConnexe(k, capteurs, grille, matAdjCom, matAdjCap, distance_matrix, rcom)
        
        vals.append(len(capteurs))
#        print("Capteurs", capteurs)

        if(len(capteurs) < minValue):
            
            minValue = len(capteurs)
            minCapteurs = copy.deepcopy(capteurs)
        
    print(vals)
    print(min(vals))
    
    end = time.time()
    
    print(end - start)
    
    tool_box.trace(grille, minCapteurs, rcom, matAdjCap)
    
    return vals

def testConnexite(capteurs, matCom):
    colored = [0]*len(capteurs)
    
    colored[0] = 1
    toExplore = [capteurs[0]]
    while(len(toExplore) > 0):
        node = toExplore.pop()
        for nodeNeighIndex in range(len(capteurs)):
            nodeNeigh = capteurs[nodeNeighIndex]
            if(matCom[node, nodeNeigh] == 1 and not nodeNeigh == node):
                if(colored[nodeNeighIndex] == 0):
                    colored[nodeNeighIndex] = 1
                    toExplore.append(nodeNeigh)
    for b in colored :
        if b == 0:
            return False
    return True


#algoVoisinageGloutonGrillePasConnexe(10, 20, 100, 1, 1)
    
    
#gloutonReseau(50, "/Users/victorchomel/Documents/Cours/MPRO/MH/Meta/Instances/captANOR225_9_20.dat", 1, 1)

algoVoisinageGloutonReseauPasConnexe("/Users/victorchomel/Documents/Cours/MPRO/MH/Meta/Instances/captANOR225_9_20.dat", 10, 50, 2, 3)

#algoVoisinageGloutonGrille(5, 2, 3, 2, 1)




    


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:36:30 2018

@author: anatole parre
"""
import numpy as np
import heapq
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def read_data(file_path):
    """Lit les données du file_path"""
    liste = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            words = line.split()
            liste.append((float(words[1]), float(words[2])))
    return(np.array(liste), pairwise_distances(np.array(liste)))

def write_data(distance_matrix, Rcapt, Rcom, file='data.dat'):
    """Ecrit les donnees du problemes au format .dat pour le PLNE"""
    N = distance_matrix.shape[0]
    size_Rcom = 'param Rcom :=' + str(Rcom) + ';\n'
    size_Rcapt = 'param Rcapt :=' + str(Rcapt) + ';\n'
    size_N = 'param n :=' + str(N) + ';\n'
    with open(file, 'w') as f:
        f.write(size_Rcom)
        f.write(size_Rcapt)
        f.write(size_N)
        first_line = 'param d :'
        for i in range(N):
            first_line += ' ' + str(i+1)
        first_line += ':='
        f.write(first_line + '\n')
        for i in range(N):
            if i==N-1:
                endcar=';'
            else:
                endcar='\n'
            line =  ' '.join(map(str, distance_matrix[i,:])) + endcar
            f.write(str(i+1) + ' '+ line)
        
        
def compute_square_grid(n):
    """Pour une grille de taille n :
    Renvoie un np.array avec les coordonnees des points dans [1,n]^2
    et la matrice de distance associee"""
    coords_pts = [(i,j) for i in range(n) for j in range(n)]
    return np.array(coords_pts), pairwise_distances(coords_pts)

def reconstruction(coords_pts, dist, capteurs, matAdjCom, Rcom):
    matCsr = csr_matrix(matrice_csr(dist[capteurs,:][:,capteurs], Rcom))
    Tcsr = minimum_spanning_tree(matCsr).toarray().astype(int)
    Tcsr_compl = Tcsr + Tcsr.transpose()
    
    X, Y = np.where(Tcsr > 100)
    
    if type(capteurs) != type([]):
        capteurs_fixed = capteurs.tolist()
    else:
        capteurs_fixed = capteurs
        
    for i in range(len(X)):
        X0 = capteurs[X[i]]
        Y0 = capteurs[Y[i]]
        connexite = False
        k=0
        new_capteurs = []
        while connexite == False and k<3:
            
            # indice des points qui sont dans le cercle de Rcom de X[i]
            indexes_X = np.where(matAdjCom[X0,:] == 1)[0]
            
            indexes_Y = np.where(matAdjCom[Y0,:] == 1)[0]
            
            commonalities = set(indexes_X) - (set(indexes_X) - set(indexes_Y))
            if len(commonalities) != 0:
                if len(new_capteurs) != 0:
                    for i in new_capteurs:
                        capteurs_fixed.append(i)
                capteurs_fixed.append(commonalities.pop())
                connexite = True
                break
            
            X0 = indexes_X[np.argmin(dist[indexes_X,Y0])]
            Y0 = indexes_Y[np.argmin(dist[indexes_Y,X0])]
            
            new_capteurs.append(X0)
            new_capteurs.append(Y0)
            
            k+=1
    
    #tool_box.trace(coords_pts[capteurs,:], range(len(capteurs)), Rcom, Tcsr)
    
    #matCsr = csr_matrix(matrice_csr(dist[capteurs_fixed,:][:,capteurs_fixed], Rcom))
    #Tcsr = minimum_spanning_tree(matCsr).toarray().astype(int)
    
    #Tcsr_compl = Tcsr + Tcsr.transpose()
    
    #tool_box.trace(coords_pts[capteurs_fixed,:], range(len(capteurs_fixed)), Rcom, Tcsr)
    
    return(capteurs_fixed)
    
def matrice_csr(distance_matrix, rcom):
    n = distance_matrix.shape[0]
    
    matAdjCom_seuil = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1,n):
            d = distance_matrix[i, j]
            if(d <= rcom):
                matAdjCom_seuil[i, j] = d 
            else:
                matAdjCom_seuil[i, j] = 100 + d
                
    return matAdjCom_seuil

def trace(coords_pts, capteurs, Rcom, matAdjCap):
    """Trace sur une grid les points en bleu et les capteurs en rouge"""
    
    n2 = matAdjCap.shape[0]
    
    for i in range(n2):
        if i in capteurs:
            indexes_voisins = np.where(matAdjCap[i,:]==1)[0]
            for j in indexes_voisins:
                X = (coords_pts[i,0],coords_pts[j,0])
                Y = (coords_pts[i,1],coords_pts[j,1])
                plt.plot(X, Y, 'g-')
            
    plt.plot(coords_pts[:,0], coords_pts[:,1], 'b.')
    plt.plot(coords_pts[capteurs,0], coords_pts[capteurs,1], 'r.', markersize=12)
    plt.axis('equal')
    plt.show()
    
    #contraintes de connexe
    c = []
    for i in range(len(capteurs)):
        #plt.Circle((coords_pts[capteurs[i],0]+1, coords_pts[capteurs[i],1]+1), 10, color='b')
        c.append(plt.Circle((coords_pts[capteurs[i],0], coords_pts[capteurs[i],1]), Rcom, color='b'))
    fig, ax = plt.subplots()
    for i in c:
        ax.add_artist(i)
    plt.plot(coords_pts[capteurs,0], coords_pts[capteurs,1], 'r.', markersize=12)
    
    plt.axis('equal')
    plt.show()
    
    return 0

def nodesGrille(n):
    nodes = np.arange(0, n**2)
    
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
            
    

def algoGloutonReseau(nodes, matCom, matCap, rcom, rcapt):
    n = nodes.shape[0]
    
    covered = np.zeros(n)
    capteurs = []
    
    capteurs.append(0)
    
    covered = captCover(0, rcapt, covered, matCap)
    
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
        captCover(newCapt, rcapt, covered, matCap)
        
#        print(testFinReseau(covered))
        
#        print(newNode)
        
#    visu(nodes, n)
#    print(capteurs)
    
    return (capteurs)

def captCover(capt, rcapt, covered, matCap):
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

def evalNewCapt(capt, rcapt, covered, matCap):
    evaluation = 0
    n = covered.shape[0]
    
    for captNeighbour in range(n):
        if(matCap[capt, captNeighbour] == 1):
            evaluation += 1 - covered[captNeighbour] 
    return evaluation



def gloutonGrille(iterations, n, rcapt, rcom):
    start = time.time()
    
    res = []
    grille = nodesGrille(n)
    matAdjCom = matAdjGrille(n, rcom)
    matAdjCap = matAdjGrille(n, rcapt)
    for i in range(iterations):
        res.append(len(algoGloutonReseau(grille, matAdjCom, matAdjCap, rcom, rcapt)))
    print(res)
    print(min(res))
    
    print(sorted(res))
    
    end = time.time()
    
    print(end - start)
    
#gloutonGrille(1, 40, 2, 3)


def voisinage(k, capteurs, nodes, matCom, matCap, rcapt, rcom):
    capteurs = removeK(k, capteurs)
    capteurs = complete(capteurs, nodes, matCom, matCap, rcapt, rcom)
    
    return capteurs
    
    
def removeK(k, capteurs):
    if k < len(capteurs):
        toRemove = np.random.choice(capteurs, k, replace = False)
        
        for captRemov in toRemove:
            capteurs = capteurs.remove(captRemov)
        
    return capteurs

def complete(capteurs, nodes, matCom, matCap, rcapt, rcom):
    n = nodes.shape[0]
    print(capteurs)
    
    covered = alreadyCovered(capteurs, matCap, rcapt, n)
    
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
        captCover(newCapt, rcapt, covered, matCap)
        
    return capteurs

def alreadyCovered(capteurs, matCap, rcapt, n):
    covered = np.zeros(n)
    
    for capt in capteurs :
        covered = captCover(capt, rcapt, covered, matCap)
        
    return covered

def gloutonReseau(iterations, path, rcapt, rcom):
    start = time.time()
    
    res = []
    nodes = read_data(path)
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

def algoVoisinageGloutonReseau(path, k, p, rcom, rcapt):
    nodes = read_data(path)
    distance_matrix = pairwise_distances(nodes)
    matAdjCom, matAdjCap = matrices_adj(distance_matrix, rcom, rcapt)
    
    capteurs = algoGloutonReseau(nodes, matAdjCom, matAdjCap, rcom, rcapt)
    print(capteurs)
    
    vals = [len(capteurs)]
    
    for i in range(p):
        print(vals)
        capteurs = voisinage(k, capteurs, nodes, matAdjCom, matAdjCap, rcom, rcapt)
        
        vals.append(len(capteurs))
        
    return vals
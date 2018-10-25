# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:31:13 2018

@author: anatole parre
"""

from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import projet

def read_data(file_path):
    liste = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            words = line.split()
            liste.append((float(words[1]), float(words[2])))
    return(np.array(liste))

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


def trace(coords_pts, capteurs, matAdjCom, matAdjCap):
    """Trace sur une grid les points en bleu et les capteurs en rouge"""
    
    n2 = matAdjCap.shape[0]
    
    for i in range(n2):
        if i in capteurs:
            indexes_voisins = np.where(matAdjCap[i,:]==1)[0]
            for j in indexes_voisins:
                X = (coords_pts[i,0]+1,coords_pts[j,0]+1)
                Y = (coords_pts[i,1]+1,coords_pts[j,1]+1)
                plt.plot(X, Y, 'g-')
            
    plt.plot(coords_pts[:,0]+1, coords_pts[:,1]+1, 'b.')
    plt.plot(coords_pts[capteurs,0]+1, coords_pts[capteurs,1]+1, 'r.', markersize=12)
    plt.show()
    return 0

if False:
    file_path = 'Instances\captANOR1500_21_500.dat'
    coords_pts = read_data(file_path)
    trace(coords_pts, [])
    
if True:
    n = 5
    Rcom = 2
    Rcapt = 1
    num_to_select = 5
    coords_pts, distance_matrix = compute_square_grid(n)
    
    #calcul des matrices d'adjacence
    matAdjCom, matAdjCap = matrices_adj(distance_matrix, Rcom, Rcapt)
    
    #generation aleatoire de capteurs
    capteurs = np.array(rd.sample(range(n*n), num_to_select))
    trace(coords_pts, capteurs, matAdjCom, matAdjCap)

if False:
    N = 5
    Rcapt = 1
    Rcom = 2
    coords_pts, dist = compute_square_grid(N)
    write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
    
if False:
    N = 5
    Rcapt = 1
    Rcom = 1
    file_path = 'Instances\captANOR1500_21_500.dat'
    coords_pts = read_data(file_path)
    dist = pairwise_distances(coords_pts)
    write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
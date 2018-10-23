# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:31:13 2018

@author: anatole parre
"""

from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import random as rd 

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
            line =  ' '.join(map(str, distance_matrix[i,:].astype(int))) + endcar
            f.write(str(i+1) + ' '+ line)
        
        
def compute_square_grid(n):
    """Pour une grille de taille n :
    Renvoie un np.array avec les coordonnees des points dans [1,n]^2
    et la matrice de distance associee"""
    coords_pts = [(i,j) for i in range(n) for j in range(n)]
    return np.array(coords_pts), pairwise_distances(coords_pts)



def trace(coords_pts, capteurs):
    """Trace sur une grid les points en bleu et les capteurs en rouge"""
    plt.plot(coords_pts[:,0]+1, coords_pts[:,1]+1, 'b.')
    plt.plot(capteurs[:,0]+1, capteurs[:,1]+1, 'r.')
    return 0



if True:
    n = 10
    num_to_select = 5
    coords_pts, _ = compute_square_grid(n)
    
    #generation aleatoire de capteurs
    capteurs = np.array(rd.sample(compute_square_grid(n)[0].tolist(), num_to_select))
    trace(coords_pts, capteurs)

if False:
    N = 6
    Rcapt = 3
    Rcom = 4
    coords_pts, dist = compute_square_grid(N)
    write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:31:13 2018

@author: anatole parre
"""

import random as rd
import numpy as np
import tool_box
import os
    
def read_sol(sol_path):
    with open(sol_path, 'r') as f:
        data = f.readlines()
        for line in data:
            if (data.find('x[') != -1):
                
                return(0)

if False:
    file_path = 'Instances\captANOR1500_21_500.dat'
    coords_pts = tool_box.read_data(file_path)
    tool_box.trace(coords_pts, [])
    
if False:
    n = 5
    Rcom = 2
    Rcapt = 1
    num_to_select = 5
    coords_pts, distance_matrix = tool_box.compute_square_grid(n)
    
    #calcul des matrices d'adjacence
    matAdjCom, matAdjCap = tool_box.matrices_adj(distance_matrix, Rcom, Rcapt)
    
    #generation aleatoire de capteurs
    capteurs = np.array(rd.sample(range(n*n), num_to_select))
    trace(coords_pts, capteurs, Rcom, matAdjCap)

if False:
    Rcom = 2
    Rcapt = 1
    num_to_select = 50
    
    file_path = 'Instances\captANOR1500_21_500.dat'
    coords_pts, distance_matrix = tool_box.read_data(file_path)
    n = coords_pts.shape[0]
    dist = tool_box.pairwise_distances(coords_pts)
    
    #calcul des matrices d'adjacence
    matAdjCom, matAdjCap = tool_box.matrices_adj(distance_matrix, Rcom, Rcapt)
    
    #generation aleatoire de capteurs
    capteurs = np.array(rd.sample(range(n), num_to_select))
    tool_box.trace(coords_pts, capteurs, matAdjCom, matAdjCap)

if False:
    N = 5
    Rcapt = 1
    Rcom = 2
    coords_pts, dist = tool_box.compute_square_grid(N)
    tool_box.write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
    
if False:
    Rcapt = 2
    Rcom = 3
    file_path = 'Instances\captANOR225_9_20.dat'
    coords_pts, dist = tool_box.read_data(file_path)
    tool_box.write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
        

if False:
    
    
    N = 10
    Rcapt = 1
    Rcom = 1
    #file_path = 'Instances\captANOR225_9_20.dat'

    #coords_pts, dist = tool_box.read_data(file_path)

    coords_pts, dist = tool_box.compute_square_grid(N)
    n = coords_pts.shape[0]
    num_to_select = int(n/4)
    capteurs = np.array(rd.sample(range(n), num_to_select))
    
    matAdjCom, matAdjCap = tool_box.matrices_adj(dist, Rcom, Rcapt)
  
    new_capteurs = reconstruction(coords_pts, dist, capteurs, matAdjCom)
    print(capteurs)
    print(new_capteurs)
    
    
        
if False:
    Rcapt = 2
    Rcom = 3
    n = 6
    file_path = 'Instances\captANOR225_9_20.dat'
    sol_path = 'sol.txt'

    coords_pts, dist_matrix = tool_box.read_data(file_path)

    nb_pts = coords_pts.shape[0]
    L = max(coords_pts[:,1]) - min(coords_pts[:,1])    
    cote_carre = 2.5*L/(np.sqrt(nb_pts))    

    for i in range(2):
        for j in range(2):
            inds = np.where((coords_pts[:,0] >= i*cote_carre)*(coords_pts[:,0] <= (i+1)*cote_carre))[0]
            dist = dist_matrix[inds,:][:,inds]
            tool_box.write_data(distance_matrix=dist, Rcapt=Rcapt, Rcom=Rcom, file='Meta.dat')
            os.system('glpsol -m Meta.mod -d Meta.dat -o sol.txt')
            solution = read_sol(sol_path)
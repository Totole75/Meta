# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:31:13 2018

@author: anatole parre
"""

import random as rd
import tool_box


if False:
    file_path = 'Instances\captANOR1500_21_500.dat'
    coords_pts = tool_box.read_data(file_path)
    tool_box.trace(coords_pts, [])
    
if True:
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
    trace(coords_pts, capteurs, matAdjCom, matAdjCap)

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
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:08:45 2018

@author: anatole parre
"""

import numpy as np
import random as rd
import tool_box
import time

def generation_voisin(grille):
    return(grille)
    
def generation_sol_initiale(input_reseau, Rcom, Rcapt):
    ## Parameters
    tp_lim_sec = 1*60 #en secondes
    
    if type(input_reseau) == int:
        #alors on veut une grille
        coords_pts, mat_dist = tool_box.compute_square_grid(input_reseau)
    else:
        coords_pts, mat_dist = tool_box.read_data(input_reseau)
        
    matAdjCom, matAdjCap = tool_box.matrices_adj(mat_dist, Rcom, Rcapt)
    capteurs_init = []
    taille = []
    timeout = time.time() + tp_lim_sec
    
    count = 0
    while True:
        capteurs_iteration = tool_box.algoGloutonReseau(coords_pts, matAdjCom, matAdjCap, Rcom, Rcapt)
        capteurs_init.append(capteurs_iteration)
        taille.append(len(capteurs_iteration))
        if time.time() > timeout:
            break
        count += 1
    print("Heuristique initiale : " + str(count) + " iterations.")
    
    capteurs_choisis = capteurs_init[np.argmin(taille)]
    return(coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs_choisis)
    
def temperature(k, param_temperature, T0):
    T = T0*param_temperature**k
    return(T)
    
def energy(sol, matAdjCom, matAdjCap, poids_com):
    """score associe a la solution sol"""
    capteurs = sol[1]
    penalite_com = np.sum(matAdjCom[np.array(capteurs)])
    E = len(capteurs) - poids_com*penalite_com
    return(E)
    
def P(E1, E2, T):
    return(np.exp((E2-E1)/T))

def recuit_simule(input_reseau, Rcom, Rcapt):
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs_choisis = generation_sol_initiale(input_reseau, Rcom, Rcapt)
    tool_box.trace(coords_pts, capteurs_choisis, Rcom, matAdjCap)

    """
    
    energie_courante = energy(sol_courante, matAdjCom, matAdjCap, poids_com)
    meilleure_solution = sol_courante
    
    for k in range(Kmax):
        T = temperature(k, param_temperature, T0)
        sol_voisin = generation_voisin(sol_courante)
        energie_voisin = energy(sol_voisin, matAdjCom, matAdjCap, poids_com)
        if energie_voisin < energie_courante:
            sol_courante = sol_voisin
            meilleure_solution = sol_courante
        else:
            if P(energie_courante, energie_voisin, T) > rd.random(1):
                sol_courante = sol_voisin
        
        if k%nb_affichage == 0:
            trace(meilleure_solution[0], meilleure_solution[1], matAdjCom, matAdjCap)
    
    return meilleure_solution
    """
if True:
    
    input_reseau = 'Instances\captANOR225_9_20.dat'
    #input_reseau = 6

    Rcom = 2
    Rcapt = 1
    
    recuit_simule(input_reseau, Rcom, Rcapt)
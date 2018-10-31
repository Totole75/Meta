# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:08:45 2018

@author: anatole parre
"""

import numpy as np
import random as rd
import tool_box
import time
import projet
import meta

def generation_voisin(coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs, Rcom, Rcapt, k):
    capteurs_voisin = projet.removeK(k, capteurs)
    
    capteurs_voisin = projet.algoGloutonReseau(coords_pts, matAdjCom, matAdjCap, capteurs_voisin)

    capteurs_connexe = meta.reconstruction(coords_pts, mat_dist, capteurs_voisin, matAdjCom, Rcom)
    
    capteurs_connexe = projet.removeConnexe(capteurs_connexe, coords_pts, matAdjCom, matAdjCap)
    
    return(capteurs_voisin, capteurs_connexe)
    
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
    
def energy(coords_pts, matAdjCom, matAdjCap, capteurs_total):
    """score associe a la solution sol"""
    #capteurs = sol[1]
    #penalite_com = np.sum(matAdjCom[np.array(capteurs)])
    #E = len(capteurs) - poids_com*penalite_com
    return(len(capteurs_total))
    
def P(E1, E2, T):
    return(np.exp((E2-E1)/T))

def recuit_simule(input_reseau, Rcom, Rcapt):
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs = generation_sol_initiale(input_reseau, Rcom, Rcapt)
    #tool_box.trace(coords_pts, capteurs, Rcom, matAdjCap)

    energie_courante = energy(coords_pts, matAdjCom, matAdjCap, capteurs)
    meilleurs_capteurs = capteurs
    capteurs_courant = capteurs
    meilleurs_energie = energie_courante
    print("Longueur initiale " + str(len(capteurs)))
    for k in range(Kmax):
        T = temperature(k, param_temperature, T0)
        capteurs_voisin, capteurs_connexe = generation_voisin(coords_pts, mat_dist, matAdjCom, matAdjCap, capteurs_courant, Rcom, Rcapt, k)
        energie_voisin = energy(coords_pts, matAdjCom, matAdjCap, capteurs_connexe)
        print(len(capteurs_connexe))
        # Gestion de la solution courante
        if energie_voisin <= energie_courante:
            capteurs_courant = capteurs_voisin
            energie_courante = energie_voisin

        else:
            if P(energie_courante, energie_voisin, T) > rd.random():
                capteurs_courant = capteurs_voisin
                energie_courante = energie_voisin
        
        # Gestion de la meilleure solution
        if energie_courante < meilleurs_energie:
            meilleurs_capteurs = capteurs_courant
            meilleurs_energie = energie_courante
        #if k%nb_affichage == 0:
        #    trace(meilleure_solution[0], meilleure_solution[1], matAdjCom, matAdjCap)
    
    return len(meilleurs_capteurs)
    
if True:
    
    input_reseau = 'Instances\captANOR225_9_20.dat'
    #input_reseau = 10

    Rcom = 1
    Rcapt = 1
    
    print("Longueur optimale ",recuit_simule(input_reseau, Rcom, Rcapt))
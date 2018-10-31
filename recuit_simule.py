# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:08:45 2018

@author: anatole parre
"""

import numpy as np
import random as rd
import tool_box

def generation_voisin(grille):
    return(grille)
    
def generation_sol_initiale():
    return(1)
    
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

def recuit_simule():
    #parametres
    Kmax = 100
    param_temperature = 0.99
    T0 = 100
    nb_affichage = 100
    
    sol_courante = generation_sol_initiale()
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
    
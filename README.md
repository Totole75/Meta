# Meta
Metaheuristique sur le placement de capteurs en milieu hostile

# Fichier recuit_simule.py
Code d'execution du recuit simule, qui fait appel à tool_box.py et à projet.py

L'execution de ce script permet de générer une solution initiale et d'exécuter le recuit simulé.

# Fichier tool_box.py
Ensemble de fonctions qui sont utilisées dans les autres fichiers pour faire différentes tâches élémentaires :
afficher une solution, calculer un arbre couvrant, lire les données, écrire un fichier de données, faire appel au PLNE.

# Fichier projet.py
Ensemble des fonctions relatives aux voisinages et aux heuristiques. Contient des fonctions élémentaires qui sont appelées par la fonction recuit_simule.py

# Fichier meta.py
Fichier qui permet d'exécuter des petites briques du code simplement.

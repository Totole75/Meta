param n; #taille de la grille
param Rcapt; #rayon de captation
param Rcom; #rayon de communication
param d{i in 1..n,j in 1..n}; #matrice de distance entre i et j

var x{i in 1..n}binary; #1 si un capteur est en position i
var y{i in 1..n,j in 1..n, k in 1..n}binary; #flot au depart de i entre  j et k a destination de 1


minimize obj: sum{i in 1..n}(x[i]);
subject to
captation {i in 1..n} : sum{j in 1..n : d[i,j] <= Rcapt}(x[j]) >= 1;

kirshof {i in 1..n, j in 1..n : i<>j and i<>1 and j<>1} : sum{k in 1..n}y[i,j,k] - sum{k in 1..n}y[i,k,j] = 0;

source {i in 1..n : i<>1} : sum{k in 1..n}y[i,i,k] = 1;
source2 {i in 1..n : i<>1} : sum{k in 1..n}y[i,k,i] = 0;

puits {i in 1..n : i<>1} : sum{k in 1..n}y[i,k,1] = 1;
puits2 {i in 1..n : i<>1} : sum{k in 1..n}y[i,1,k] = 0;

activation {i in 1..n, j in 1..n, k in 1..n : i <> 1 and k<>1} : y[i,j,k] <= x[k];

diagonulle {i in 1..n, k in 1..n} : y[i,k,k] = 0;

communication1 {i in 1..n, j in 1..n, k in 1..n : d[j,k] > Rcom} : y[i,j,k] = 0;

solve;
display x;
display y;
data;

#param n:=5;
#param Rcapt := 2;
#param Rcom := 5;
#param d : 1 2 3 4 5:=
#1 1 0 1 2 3
#2 1 3 1 4 1
#3 1 2 1 1 3
#4 1 2 3 4 5
#5 3 1 2 1 2;
# LeNet-5

*Last update: 2024.11.29*

## Get started
These tutorials courses are done using Python notebooks.

### Install prerequesities

```
python3 -m venv .venv

# Activate the environment
# Linux/MacOs
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

### Leave the environment after usage
```
deactivate
```

## **TP : Implémentation d'un CNN - LeNet-5 sur GPU**

### **Objectifs & Méthodes de travail:**

Les objectif de ces 4 séances de TP de HSP sont :
- Apprendre à utiliser CUDA,
- Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU,
- Observer les limites de l'utilisation d'un GPU,
- Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement,
- Exporter des données depuis un notebook python et les réimporter dans un projet cuda,
- Faire un suivi de votre projet et du versionning à l'outil git

### **Implémentation d'un CNN** 
L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très claissque : LeNet-5 proposé par Yann LeCun et al. en 1998 pour la reconnaissance de chiffres manuscrits.

La lecture de cet article vous apportera les informations nécessaires pour comprendre ce réseau de neurone.

![alt text](image.png)

**Layer 3 Attention**: Contraitement à ce qui est décrit dans l'article, la 3eme couche du CNN prendra en compte tous les features pour chaque sortie.

## **Partie 1 -** Prise en main de CUDA: Multiplication de matrices

**Multiplication de matrices**

**Paramètres:**
- n : nombre de lignes de la matrice,
- p : nombre de colonnes de la matrice si n différent de p,
- M : pointeur de la matrice

**Allocation de mémoire** <br>
L'allocation de la mémoire (malloc) se fera dans votre fonction principale main.

***

### **1.1 Création d'une matrice sur CPU** <br>
Cette fonction initialise une matrice de taille n x p Initialisez les valeurs de la matrice de façon aléatoire entre -1 et 1.

```
void MatrixInit(float *M, int n, int p)
```

**Résultat:**<br>
Execute 
`./matrix`
```
-0.46 -0.61 0.67 
-0.17 -0.96 -0.69 
0.38 0.61 -0.35
```

***

### **1.2 Affichage d'une matrice sur CPU**<br>
Cette fonction affiche une matrice de taille n x p.

```
void MatrixPrint(float *M, int n, int p)
```

**Résultat:**: 
Execute 
`./matrix-print`
```
-0.20 0.83 0.16 0.98 -0.02 
0.68 -0.20 0.45 0.77 -0.09 
0.75 -0.13 0.37 0.22 -0.24
```

***
### **1.3 Addition de deux matrices sur CPU** <br>
Cette fonction additionne deux matrices M1 et M2 de même taille n x p.
```
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

**Résultat:**<br>
Execute 
`./matrix-add`
```
Matrice M1 :
-0.54 0.39 -0.73 -0.78 
-0.09 -0.42 -0.84 0.70 
0.70 -0.04 -0.18 -0.80 

Matrice M2 :
-0.54 0.39 -0.73 -0.78 
-0.09 -0.42 -0.84 0.70 
0.70 -0.04 -0.18 -0.80 

M1 + M2:
-1.08 0.77 -1.47 -1.56 
-0.18 -0.84 -1.69 1.40 
1.41 -0.08 -0.37 -1.61
```

***
### **1.4 Addition de deux matrices sur GPU** <br>
Cette fonction additionne deux matrices M1 et M2 de même taille n x p Vous pouvez considérer les
dimensions des matrices comme les paramètres gridDim et blockDim : les lignes correspondent aux blocks,
les colonnes correspondent aux threads.
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p)
```

**Résultat pour (n,p = 1000,1000):** <br>
Execute 
`time ./matrix-add-gpu-cpu GPU 1000 1000`
```
On GPU:

real  0m0.286s
user  0m0.113s
sys   0m0.153s
```

`time ./matrix-add-gpu-cpu CPU 1000 1000`
```
On CPU:

real  0m9.085s
user  0m9.073s
sys   0m0.012s
```

**Résultat pour (n,p = 10,10):** <br>
Execute 
`time ./matrix-add-gpu-cpu-10 GPU`
```
On CPU:

real  0m0.002s
user  0m0.000s
sys   0m0.002s
```

Execute 
`time ./matrix-add-gpu-cpu-10 GPU`
```
On GPU:

real  0m0.180s
user  0m0.024s
sys   0m0.132s
```

**Addition de deux matrices:**
| **Device**     | **n=10, p=10**      | **n=1000, p=1000**      |
|----------------|----------------|----------------|
| CPU | 2 ms  | 9.085s |
| GPU| 180 ms | 286ms |

&rarr; On en conclu que pour les calculs complexes (dès qu'une a une matrice de dimension élevée), la parallélisation du GPU permet des calculs beaucoup plus rapides qu'avec le CPU.

***

### **1.5 Multiplication de deux matrices NxN sur CPU** <br>
Cette fonction multiplie 2 matrices M1 et M2 de taillle n x n.
```
void MatrixMult(float *M1, float *M2, float *Mout, int n)
```

**Résultat pour une matrice 3x3:**: <br>
Execute 
`time ./matrix-product-cpu`
```
On CPU:
Matrice M1 :
0.06 0.60 0.00 
-0.20 0.88 -0.46 
-0.65 -0.48 -0.07 

Matrice M2 :
0.06 0.60 0.00 
-0.20 0.88 -0.46 
-0.65 -0.48 -0.07 

M1 * M2:
-0.11 0.56 -0.27 
0.11 0.89 -0.37 
0.10 -0.78 0.23 
```

Execute: <br>
`time ./matrix-product-cpu-gpu GPU 1000 1000`
```
On GPU:

real  0m0.297s
user  0m0.119s
sys   0m0.154s
```
`./matrix-product-cpu-gpu CPU 1000 1000`
```

On CPU:

real  0m9.083s
user  0m9.074s
sys   0m0.009s
```

| **Device**     | **n=3, p=3**      | **n=1000, p=1000**      |
|----------------|----------------|----------------|
| CPU | 0.005s  | 9.083s |
| GPU| 0.185s | 0.297s |

&rarr; On en conclu que pour les calculs complexes (dès qu'une a une matrice de dimension élevée), la parallélisation du GPU permet des calculs beaucoup plus rapides qu'avec le CPU.


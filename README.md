# LeNet-5

*Last update: 2024.12.09*

## Get started

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

## **Partie 2.** Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

L'architecture du réseau LeNet-5 est composé de plusieurs couches:
- **Layer 1**: Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST.
- **Layer 2**: Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
- **Layer 3:** Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

### **Layer 1 - Génération des données de test**

On génère tout d'abord les matrices dont on a besoin :
- Une matrice float **raw_data** de taille 32x32 initialisé avec des valeurs comprises entre 0 et 1, correspondant à nos données d'entrée.
- Une matrice float **C1_data** de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première convolution.
- Une matrice float **S1_data** de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du souséchantillonnage. S1 correspond aux données après le premier Sous-échantillonnage.
- Une matrice float **C1_kernel** de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.

On crée des tableaux à 1 dimension N=32x32, 6x28x28, 6x14x14 et 6x5x5 respectivement pour ces vecteurs.

### **Layer 2 - Convolution 2D** <br> 
Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

### **Layer 3- Sous-échantillonnage** <br> 
Sous-échantillonnage d'un facteur 2. La taille résultantes des données
est donc de 6x14x14.

### **Tests**
```
Canal 0 :
6.29 7.56 6.57 8.09 7.90 8.28 6.47 7.86 7.17 6.51 6.58 8.51 8.02 8.28 
9.20 8.80 6.26 7.06 6.64 6.90 5.46 7.32 7.72 7.15 7.97 7.68 6.61 6.41 
6.11 7.20 6.87 7.57 8.15 7.78 6.87 9.15 7.94 6.47 6.65 9.01 7.88 8.04 
8.04 8.14 6.99 5.99 7.12 5.95 5.80 6.45 6.58 6.00 8.16 7.92 6.95 7.19 
7.34 6.69 5.89 7.18 7.88 6.66 7.39 8.55 8.48 5.54 7.48 7.71 7.69 8.18 
8.49 8.24 6.35 6.19 5.96 5.78 6.02 7.43 6.22 6.39 6.75 6.90 5.43 7.92 
7.92 8.31 7.38 7.32 6.85 6.98 8.57 8.16 7.13 5.95 7.33 7.10 6.03 7.85 
8.65 8.87 6.08 8.52 6.00 5.96 6.30 8.11 5.08 7.13 7.81 7.61 5.85 9.14 
7.12 7.50 8.29 7.77 7.43 7.54 8.61 7.50 6.15 6.53 5.79 6.97 5.82 6.81 
7.36 7.94 7.37 7.36 6.67 6.74 5.98 7.16 6.14 7.87 8.21 7.39 7.22 9.29 
6.61 7.72 7.67 7.93 8.32 8.99 8.44 7.90 8.29 7.44 7.03 6.90 6.78 6.57 
7.17 7.22 7.44 6.94 7.38 6.25 6.41 6.88 7.73 7.78 8.13 7.33 7.54 8.76 
7.46 8.24 7.81 8.87 8.59 9.06 8.59 8.04 8.80 8.16 7.35 6.73 6.15 6.94 
5.69 6.75 4.98 6.37 5.40 6.02 6.60 7.89 6.78 8.89 7.87 6.95 7.88 8.54 
Canal 1 :
6.68 9.17 6.98 8.92 9.00 10.57 7.76 9.09 7.77 8.99 7.53 7.86 7.56 8.04 
7.37 7.74 5.70 7.41 6.46 6.43 6.72 8.27 6.92 8.73 7.27 7.33 9.19 8.48 
5.73 7.04 6.17 7.61 8.05 9.14 8.38 9.48 8.21 8.19 6.75 8.23 6.94 6.31 
6.86 6.44 5.06 6.72 6.22 6.13 5.71 7.30 7.64 8.45 8.30 7.83 8.25 7.98 
6.81 7.54 6.43 7.26 7.17 8.27 8.04 8.64 8.47 8.47 7.73 8.32 6.94 6.00 
5.39 6.44 4.81 6.22 6.04 5.62 6.99 6.20 7.81 8.17 9.13 7.93 8.04 6.98 
6.61 7.88 6.93 7.35 8.61 8.05 7.36 8.08 8.12 8.17 6.68 8.46 7.10 6.94 
5.73 6.40 5.89 6.83 5.57 6.60 5.57 6.95 7.64 7.87 7.50 7.66 7.76 7.15 
6.76 7.65 5.67 7.84 8.86 7.71 6.70 7.33 6.44 7.24 5.91 7.67 7.58 7.95 
8.40 7.58 8.03 8.19 7.38 7.51 7.52 8.80 7.40 7.79 7.46 8.37 8.05 8.76 
7.97 6.05 5.98 6.50 7.10 5.93 7.53 7.41 6.10 6.88 6.31 7.28 6.51 7.16 
7.61 6.53 5.42 6.98 7.33 6.88 8.37 8.36 9.14 9.35 8.90 8.46 8.76 8.93 
8.74 6.23 6.96 8.94 8.00 7.31 7.09 7.39 6.76 5.80 6.67 7.76 8.00 8.74 
9.10 8.59 6.99 7.74 8.01 8.47 10.14 10.70 9.59 9.18 8.62 8.45 7.85 7.01 
Canal 2 :
7.39 6.26 7.65 8.60 9.03 8.27 8.25 7.16 6.24 5.88 6.29 5.52 7.00 7.62 
8.04 7.13 6.58 7.15 6.53 8.63 9.43 11.10 9.56 10.42 8.93 8.13 6.06 6.51 
6.76 7.34 8.25 8.95 7.76 7.45 7.37 7.51 6.38 6.28 6.29 6.86 7.20 7.60 
8.90 7.74 8.12 6.50 7.27 7.26 8.34 10.19 9.78 9.74 7.29 7.84 7.45 6.81 
8.06 6.21 7.77 9.31 8.27 7.91 8.16 7.57 6.98 6.30 7.30 7.01 7.70 7.34 
8.09 6.90 7.04 7.38 7.84 6.61 7.39 8.33 9.74 8.67 7.46 7.81 6.91 6.63 
8.59 6.64 8.51 8.97 8.87 7.19 9.34 7.19 6.52 4.57 6.59 6.33 7.79 7.82 
9.70 8.62 8.57 9.29 9.21 8.88 8.14 8.33 7.64 7.05 5.98 6.36 6.78 6.87 
7.03 7.39 8.04 8.80 8.41 7.18 9.02 7.09 7.15 5.86 6.65 5.48 7.42 6.52 
8.46 7.78 7.24 6.52 7.42 7.80 6.47 7.39 6.72 6.41 5.05 7.04 6.29 6.75 
7.24 6.72 7.37 7.85 7.62 7.50 7.74 7.25 6.75 6.06 6.29 6.47 7.52 8.04 
9.52 8.34 9.74 8.48 8.83 7.32 7.60 6.78 7.62 5.60 5.70 7.75 6.82 6.90 
7.98 5.99 8.28 7.79 7.83 7.62 8.26 7.13 7.93 5.96 6.04 6.36 6.88 7.58 
8.39 8.88 8.67 8.28 8.98 7.70 7.94 7.50 8.46 6.33 6.95 8.16 7.75 7.74 
Canal 3 :
7.81 5.59 7.84 7.15 8.02 7.38 8.94 7.98 9.36 6.54 7.30 7.36 7.95 6.14 
7.22 7.62 8.22 8.15 9.12 8.22 7.72 8.86 8.96 7.40 7.77 7.60 8.00 8.34 
7.08 6.89 8.09 7.71 7.29 6.88 7.05 8.27 7.77 7.07 7.95 6.46 6.71 6.55 
7.34 6.97 7.30 6.40 7.80 6.33 7.26 7.09 8.51 8.10 9.62 8.70 7.95 7.54 
8.50 8.19 8.84 7.22 6.88 7.14 7.56 8.02 6.32 6.64 7.80 7.26 6.35 6.60 
7.40 6.66 7.38 6.93 8.74 7.51 9.10 7.59 9.21 8.76 9.89 9.21 9.30 8.25 
7.41 9.15 9.66 8.83 7.46 7.08 6.80 7.63 7.74 8.00 8.20 7.07 6.59 6.10 
7.47 6.24 6.64 6.65 7.04 7.07 7.84 7.45 8.06 7.98 9.81 8.77 9.30 8.04 
7.90 8.16 8.00 7.86 7.15 7.16 7.30 7.68 7.66 8.25 8.88 8.22 8.33 6.90 
6.67 6.40 6.57 6.22 8.06 7.28 7.59 7.69 8.47 6.33 7.77 8.51 8.20 8.57 
8.99 8.79 8.19 7.82 6.23 7.90 7.38 7.38 7.37 7.58 8.16 8.14 7.91 6.96 
7.54 7.52 6.53 7.06 8.43 7.37 6.72 7.96 8.49 7.50 7.51 7.67 8.67 8.43 
8.63 8.54 7.37 7.66 5.73 6.69 7.12 7.78 7.84 7.73 8.77 9.06 8.60 7.72 
8.12 7.52 5.46 7.34 7.74 6.98 7.24 8.20 7.67 7.57 7.73 7.11 8.47 9.08 
Canal 4 :
5.49 6.69 5.63 6.40 7.63 7.05 5.41 6.59 6.17 5.43 5.76 7.77 6.86 7.13 
6.95 7.44 5.51 4.74 5.47 5.40 4.75 5.60 6.10 6.79 6.89 6.24 6.38 5.45 
5.84 5.15 5.67 6.60 6.38 6.92 5.63 6.55 6.67 6.51 6.22 7.68 7.05 6.87 
7.23 6.91 5.87 5.08 5.31 4.83 5.24 5.41 6.02 4.78 6.58 6.53 5.06 6.06 
5.77 6.62 5.24 5.81 6.68 5.82 5.69 7.24 7.61 5.33 6.40 6.95 6.90 6.44 
7.11 7.15 5.70 5.67 5.15 5.14 5.11 5.85 4.98 5.83 5.22 6.59 4.91 5.76 
5.81 5.93 6.25 6.31 6.68 6.04 6.35 7.03 6.37 5.25 4.65 5.85 6.03 6.22 
6.58 7.36 5.42 6.20 5.59 5.95 4.59 7.01 5.66 5.20 6.25 6.56 5.19 6.97 
5.87 6.33 6.19 6.16 6.64 6.43 6.86 6.83 5.64 5.19 5.29 5.44 4.72 6.36 
6.83 5.82 6.18 6.51 5.40 5.09 6.03 6.11 4.59 6.08 6.97 6.23 6.10 7.15 
5.58 6.38 7.48 6.62 6.13 7.34 7.03 6.74 6.88 5.86 6.14 5.25 5.96 5.41 
6.47 6.89 5.34 5.42 5.58 4.85 5.24 5.32 6.36 6.45 7.36 6.28 6.14 7.21 
6.75 7.26 6.37 7.17 7.59 7.12 7.12 6.85 6.86 5.89 5.87 5.72 5.19 6.64 
5.41 6.10 4.58 5.36 5.27 4.55 5.21 6.93 5.69 7.17 6.45 6.07 6.32 7.57 
Canal 5 :
5.22 6.77 7.07 7.45 6.69 8.89 7.14 7.66 6.42 7.11 5.98 6.83 6.05 6.09 
6.86 6.34 5.07 5.78 6.39 6.00 5.75 6.94 5.20 7.09 7.11 6.53 7.20 7.36 
5.86 6.02 5.65 6.77 6.82 7.43 7.48 7.14 7.65 7.39 5.76 6.94 6.32 4.98 
5.18 5.39 3.91 5.38 4.94 5.48 5.46 6.01 6.80 7.26 7.11 6.99 7.30 6.14 
5.56 6.33 5.74 6.32 7.11 7.68 6.32 8.21 6.94 6.90 6.93 6.53 6.63 5.74 
4.04 5.61 3.84 4.16 5.62 4.77 5.70 6.15 6.71 6.83 7.03 6.35 6.71 5.97 
5.66 5.93 5.19 6.66 7.14 7.18 7.37 6.51 7.15 7.31 5.34 6.98 6.07 5.40 
5.85 5.13 4.95 5.94 4.80 5.81 5.08 5.38 5.78 7.42 6.74 6.10 6.58 6.46 
5.52 7.16 4.94 5.70 6.97 5.90 5.66 6.61 5.22 6.33 5.24 5.56 6.50 6.20 
5.79 6.86 5.80 6.59 6.70 4.84 6.10 6.63 6.21 7.20 6.05 7.00 7.03 6.60 
6.69 5.19 5.32 5.42 6.28 5.88 5.18 5.62 5.92 5.40 5.72 6.31 5.76 5.92 
5.86 5.47 5.50 5.56 5.49 5.75 6.56 6.93 6.78 7.49 8.34 6.78 7.04 7.38 
5.98 6.34 5.84 6.63 7.04 6.68 6.78 5.75 5.09 5.51 5.65 5.63 6.43 7.15 
7.09 7.41 6.24 6.70 6.96 6.75 7.41 8.96 8.40 8.34 7.45 6.41 6.24 6.44 
```


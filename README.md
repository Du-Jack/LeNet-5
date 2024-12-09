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

Dans cette partie nous allons mettre en avant l'intéret des calculs sur GPU pour la convolution. En effet les calculs de la convolution sont paraléllisables. Nous allons donc implémenter chaque étapes en CUDA.

L'architecture du réseau LeNet-5 est composé de plusieurs couches:
- **Layer 1**: Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST.
- **Layer 2**: Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
- **Layer 3:** Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

### **Layer 1 - Génération des données de test**

On génère tout d'abord les matrices dont nous avonss besoin :
- Une matrice float **raw_data** de taille 32x32 initialisé avec des valeurs comprises entre 0 et 1, correspondant à nos données d'entrée.
- Une matrice float **C1_data** de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première convolution.
- Une matrice float **S1_data** de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du souséchantillonnage. S1 correspond aux données après le premier Sous-échantillonnage.
- Une matrice float **C1_kernel** de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.

On crée des tableaux à 1 dimension N=32x32, 6x28x28, 6x14x14 et 6x5x5 respectivement pour ces vecteurs.

Nous initialisation le Kernel avec des valeurs aléatoire entre 0 et 1. Et les valeurs de la matrice initiale est initialisé avec des valeur aléatoire entre -10 et 10. (Nous avons pris l'initative de modifier cette valeur d'initialisation pour mettre en lumière la fonction d'activation. De ce fait nous obtiendrons après activation des valeurs positive ou négative entre -1 et 1)

### **Layer 2 - Convolution 2D** <br> 
Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

### **Layer 3- Sous-échantillonnage** <br> 
Sous-échantillonnage d'un facteur 2. La taille résultantes des données
est donc de 6x14x14.

### **Tests**
Affichage du premier terme de la convolution : 

Execute `./convolve `
```
# Convolution 

C1_data[0][0][0] = 7.014266
```
Initialisation de la matrice Raw data puis la convolution C1 data puis le sous échantiollonnage S1 data :

Execute `./part2-init`
```
Canal 0 :
-5.81 3.11 2.78 7.31 -5.42 3.19 -8.04 -3.40 7.03 7.85 0.91 -7.98 7.62 9.00 
-9.43 8.68 -7.89 6.90 2.14 7.05 5.71 6.62 -4.92 8.60 5.71 7.74 -7.45 -4.78 
-8.82 -3.40 -2.45 -4.63 9.71 -9.68 -7.32 -5.71 3.51 -5.36 0.89 0.54 -7.51 -8.20 
2.56 -9.89 -9.20 3.13 8.79 -7.09 0.03 0.93 9.95 -4.26 -2.45 -4.97 -5.66 -6.74 
-7.23 -3.12 -1.52 -6.05 3.49 6.03 -0.68 3.20 6.36 2.00 7.49 -0.13 6.64 -1.62 
-9.59 9.14 0.19 2.97 9.25 0.99 -3.91 8.04 3.89 6.12 -1.03 3.85 -8.14 6.52 
8.88 -3.81 9.78 -8.35 3.08 -1.74 -4.40 -3.44 -5.70 4.91 9.76 -9.35 -3.09 7.25 
0.52 -6.45 -4.37 0.93 -7.31 5.82 -6.10 -8.07 -3.19 -0.01 9.97 -9.30 -3.89 -1.06 
4.55 -2.03 -4.55 3.42 4.16 -4.77 5.07 -2.76 3.49 -9.33 3.80 7.79 5.58 3.57 
8.44 -7.51 0.82 -1.03 -3.96 6.45 9.90 -1.27 2.27 -6.20 0.66 9.07 3.79 0.63 
9.77 9.90 9.57 4.32 -2.13 -4.98 -2.26 -7.96 0.25 -7.18 -0.72 -6.26 -6.51 -6.92 
-8.47 9.07 6.64 9.97 -8.44 -2.54 -1.06 -2.40 -6.09 -1.16 6.33 6.18 2.64 -3.01 
5.25 -3.57 7.62 5.03 -3.66 7.19 -0.65 4.21 -7.79 7.09 6.25 2.46 9.91 -4.48 
6.20 -6.60 -1.40 7.73 -7.53 -4.75 7.70 -5.97 2.71 -3.36 1.63 6.62 5.48 -2.04

# And 4 other canal 
________________________________________________________________
C1 data  

Canal 0 :
144.48 -145.04 -305.38 105.30 -235.26 -9.84 110.02 -308.99 -130.64 -112.66 35.76 -156.02 -13.15 18.62 
-94.31 67.66 -10.24 -181.22 205.11 -106.69 -233.56 27.18 -216.66 155.64 -83.44 0.78 53.67 154.40 
347.05 -96.68 -22.20 70.32 191.76 104.49 -27.53 70.42 275.64 101.25 253.89 305.36 -3.36 251.13 
213.72 24.62 -145.44 -174.17 142.62 -133.44 87.90 -89.70 -83.29 -205.41 -59.62 44.23 33.11 17.30 
-39.48 -38.33 215.18 -54.41 -97.53 277.11 -338.76 113.40 295.90 -43.64 136.35 32.66 -45.85 -17.32 
-39.20 -53.76 -3.48 149.42 -53.92 94.97 220.95 -225.00 69.92 -150.06 281.59 -32.44 -95.79 219.67 
-58.95 194.97 286.86 -24.49 173.13 195.04 -90.94 167.74 33.74 40.18 -113.75 -179.93 308.80 188.28 
181.92 -216.98 142.99 -78.18 -48.59 -151.43 -53.19 -121.80 -9.98 20.97 171.19 -87.75 -4.09 197.72 
-112.20 204.83 93.08 -121.62 15.61 -114.79 -73.73 -206.87 -221.16 -204.24 -31.58 187.63 153.64 236.96 
-170.15 103.88 118.77 6.19 289.93 -252.10 181.78 8.80 18.51 56.46 -253.99 -176.89 -128.08 -217.96 
37.23 153.20 90.20 -132.27 156.07 -26.60 12.72 -131.70 179.44 126.34 110.23 92.84 19.47 -125.39 
-249.28 52.36 -317.13 -209.43 22.81 -141.30 347.44 -167.14 190.26 68.17 -88.96 46.83 29.66 -62.32 
-92.90 8.30 -91.49 -78.13 -125.09 -231.20 -154.34 -8.57 23.46 -167.49 -169.18 -153.04 -74.87 -93.18 
106.90 30.71 -39.05 282.52 278.11 -70.62 189.00 -131.91 35.50 -224.19 -161.44 37.39 -125.61 -180.99
________________________________________________________________
S1 data. 

Canal 0 :
62.45 -37.99 12.79 -39.02 33.39 109.75 63.31 52.92 -127.77 26.90 -52.04 -87.43 -24.51 64.62 
14.55 105.78 136.94 -37.14 81.55 -31.17 108.48 -32.00 52.69 -39.74 -44.76 -17.29 83.15 79.37 
70.77 -17.65 7.57 -99.89 -29.90 89.78 71.17 -65.80 -100.40 -20.17 92.72 83.35 -118.25 -94.67 
-126.49 26.54 -157.40 6.01 -120.81 -75.53 -38.72 -21.59 4.26 -39.31 77.06 7.13 -75.33 -73.43 
-198.76 82.91 -49.13 -28.27 -30.20 57.02 -50.35 -73.94 -126.75 9.17 94.68 -83.94 -19.81 -29.57 
121.30 -136.98 45.25 -114.56 14.67 77.73 -41.37 21.49 48.87 -41.31 -33.09 -135.58 -39.65 155.67 
86.30 15.14 55.47 -1.06 -49.33 -161.51 22.80 7.99 76.27 84.20 -75.07 132.17 161.07 -128.78 
-51.81 27.04 -37.60 66.29 5.35 54.13 -98.67 -66.34 -100.27 40.72 83.60 -55.07 42.22 75.83 
-12.08 20.20 0.05 40.16 28.57 -45.95 49.56 83.37 46.43 -95.69 -55.77 -204.65 38.43 32.60 
-33.93 -63.34 143.68 55.96 -24.42 81.59 143.70 73.78 -7.33 -1.91 -39.58 -167.31 -206.52 -43.29 
-47.46 158.17 64.49 -41.40 116.79 -98.45 1.02 -49.48 -37.30 -161.07 -120.00 -0.20 37.57 -105.40 
86.68 138.36 112.61 144.31 -26.65 -78.15 -66.08 -108.79 -221.48 -72.94 64.95 -109.60 -10.29 40.42 
-47.04 76.33 -66.96 -36.21 -116.65 48.66 -104.29 -52.52 28.31 -43.14 -79.00 -62.76 -72.27 -116.42 
52.52 -8.74 35.00 10.88 143.13 1.72 30.16 17.68 -97.64 31.94 -71.15 -12.39 47.45 23.30
```
Affichage de la matrice après activation :

Execute `./part2-activation`
```
S1 data. 
Canal 0 :
-1.00 1.00 1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 1.00 1.00 -1.00 1.00 1.00 
-1.00 1.00 -1.00 -1.00 1.00 -1.00 1.00 -1.00 -1.00 1.00 1.00 -0.86 0.99 -1.00 
1.00 1.00 1.00 1.00 -1.00 -1.00 -1.00 1.00 -1.00 1.00 1.00 1.00 -0.90 1.00 
-1.00 -1.00 -1.00 -1.00 -1.00 -1.00 1.00 -1.00 -1.00 -1.00 1.00 -1.00 -1.00 -1.00 
1.00 -1.00 1.00 1.00 1.00 -1.00 1.00 -1.00 1.00 1.00 -1.00 -1.00 1.00 -1.00 
1.00 1.00 -1.00 -1.00 -1.00 -1.00 1.00 -1.00 -1.00 -1.00 1.00 1.00 -1.00 1.00 
-1.00 1.00 1.00 0.94 1.00 -1.00 -1.00 1.00 1.00 -1.00 -1.00 -1.00 1.00 1.00 
1.00 -1.00 -1.00 -1.00 -1.00 1.00 -0.23 -1.00 1.00 1.00 1.00 1.00 1.00 -1.00 
1.00 1.00 1.00 1.00 -1.00 -1.00 -1.00 0.22 1.00 -1.00 -1.00 1.00 -1.00 1.00 
1.00 1.00 -1.00 -1.00 1.00 1.00 1.00 1.00 -1.00 1.00 -1.00 -1.00 -1.00 -1.00 
-1.00 1.00 -1.00 1.00 1.00 -1.00 1.00 -1.00 1.00 -1.00 -1.00 1.00 -1.00 1.00 
1.00 1.00 1.00 1.00 -1.00 -0.99 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 
-1.00 1.00 -1.00 1.00 -1.00 1.00 1.00 1.00 -1.00 -1.00 -1.00 1.00 -1.00 1.00 
1.00 1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00 1.00 1.00 -1.00 1.00 1.00
```
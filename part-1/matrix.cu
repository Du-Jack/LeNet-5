#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void MatrixInit(float *M, int n, int p) {
    // init random generator
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n * p; i++) {
        // rand() % (max - min + 1) give a number between 0 et max - min (include)
        M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void MatrixPrint(float *M, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", M[i * p + j]); 
        }
        printf("\n"); 
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

int main() {
    int n = 3, p = 4;  

    float M1[n * p], M2[n * p], Mout[n * p];

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    printf("Matrice M1 :\n");
    MatrixPrint(M1, n, p);
    
    printf("\nMatrice M2 :\n");
    MatrixPrint(M2, n, p);

    MatrixAdd(M1, M2, Mout, n, p);

    // Affichage du résultat de l'addition
    printf("\nM1 + M2:\n");
    MatrixPrint(Mout, n, p);
    return 0;
}

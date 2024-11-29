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

int main() {
    int n = 3, p = 5; 
    float matrix[n * p]; 

    MatrixInit(matrix, n, p);

    return 0;
}

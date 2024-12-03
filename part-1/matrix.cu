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

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p)
{
   // M1 rows
   for (int i = 0; i < n; i++) {
       // M2 col
       for (int j = 0; j < p; j++) {
           // Initialiser Mout[i * p + j] à zéro avant l'addition
           Mout[i * p + j] = 0;
           for (int k = 0; k < p; k++) {
               Mout[i * p + j] += M1[i * p + k] * M2[k * p + j];
           }
       }
   }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < p) {
        int index = idx * p + idy;  
        Mout[index] = M1[index] + M2[index];  
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n)

int main(int argc, char *argv[]) {
    
    const char *hardware = "CPU";
    if (argc > 1) {
        hardware = argv[1];  
    }

    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    // Declare CPU memory variables
    float M1[n * p], M2[n * p];
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    if (strcmp(hardware, "CPU") == 0) {
        float *Mout = (float *)malloc(n * p * sizeof(float));
        printf("\nOn CPU:\n");
        // Add matrices on CPU
        // MatrixAdd(M1, M2, Mout, n, p);

        MatrixMult(M1, M2, Mout, n, p);
        // MatrixPrint(Mout, n, p);

        printf("Matrice M1 :\n");
        MatrixPrint(M1, n, p);
    
        printf("\nMatrice M2 :\n");
        MatrixPrint(M2, n, p);

        printf("\nM1 * M2:\n");
        MatrixPrint(Mout, n, p);
    }
    else if (strcmp(hardware, "GPU") == 0) {
        printf("\nOn GPU:\n");
        // Allocate and initialize host matrices
        float *M1 = (float *)malloc(n * p * sizeof(float));
        float *M2 = (float *)malloc(n * p * sizeof(float));
        float *Mout = (float *)malloc(n * p * sizeof(float));

        MatrixInit(M1, n, p);
        MatrixInit(M2, n, p);

        // Allocate memory on the GPU
        float *d_M1, *d_M2, *d_Mout;
        cudaMalloc((void **)&d_M1, n * p * sizeof(float));
        cudaMalloc((void **)&d_M2, n * p * sizeof(float));
        cudaMalloc((void **)&d_Mout, n * p * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

        // Define block and grid sizes
        dim3 blockDim(16,16);  
        dim3 gridSize((p+blockDim.x-1)/blockDim.x, (n + blockDim.y-1)/blockDim.y);

        // Launch kernel
        cudaMatrixAdd<<<gridSize, blockDim>>>(d_M1, d_M2, d_Mout, n, p);
        cudaDeviceSynchronize();  

        // Copy result back from device to host
        cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

        // printf("Matrice M1 :\n");
        // MatrixPrint(h_M1, n, p);
    
        // printf("\nMatrice M2 :\n");
        // MatrixPrint(h_M2, n, p);

        // printf("\nM1 + M2:\n");
        // MatrixPrint(h_Mout, n, p);

        // Free device memory
        free(M1);
        free(M2);
        free(Mout);
        cudaFree(d_M1);
        cudaFree(d_M2);
        cudaFree(d_Mout);
    }

    return 0;
}


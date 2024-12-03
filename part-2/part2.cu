#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define WIDTH 32
#define HEIGHT 32
#define KERNEL_SIZE 5
#define NUM_CHANNELS 6
#define OUTPUT_HEIGHT 28
#define OUTPUT_WIDTH 28

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

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p){
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

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n) {
        float value = 0;
        for (int k = 0; k < n; k++) {
            value += M1[idx * n + k] * M2[k * n + idy];
        }
        Mout[idx * n + idy] = value;
    }
}

__global__ void convolve_2D(float *raw_data, float *C1_kernel, float *C1_data) {
    int c = blockIdx.z;  
    int h = blockIdx.y * blockDim.y + threadIdx.y; // ligne dans la matrice C1_data
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // colonne dans la matrice C1_data

    if (h < OUTPUT_HEIGHT && w < OUTPUT_WIDTH) {
        float sum = 0.0f;

        for (int kh = 0; kh < KERNEL_SIZE; kh++) { // parcours le kernel
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int x = w + kw;  // x coordinate of input 
                int y = h + kh;  // y coordinate of input 
                int input_index = y * WIDTH + x;  // Index in raw_data
                int kernel_index = c * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw;  // Index in C1_kernel
                sum += raw_data[input_index] * C1_kernel[kernel_index];
            }
        }
        // output
        int output_index = c * OUTPUT_HEIGHT * OUTPUT_WIDTH + h * OUTPUT_WIDTH + w;
        C1_data[output_index] = sum;
    }
}

int main(int argc, char *argv[]) {
    int N1 = 32 * 32;  
    int N2 = 6 * 28 * 28;  
    int N3 = 6 * 14 * 14; 
    int N4 = 6 * 5 * 5;    

    // Allocation des tableaux à 1 dimension
    float *raw_data = (float *)malloc(N1 * sizeof(float));  
    float *C1_data = (float *)malloc(N2 * sizeof(float));  
    float *S1_data = (float *)malloc(N3 * sizeof(float));  
    float *C1_kernel = (float *)malloc(N4 * sizeof(float));  
    // init raw_data between 0 and 1
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N1; i++) {
        raw_data[i] = ((float)rand() / RAND_MAX);
    }
    // Initialisation des matrices à 0
    for (int i = 0; i < N2; i++) {
        C1_data[i] = 0.0f;
    }
    for (int i = 0; i < N3; i++) {
        S1_data[i] = 0.0f;
    }
    for (int i = 0; i < N4; i++) {
        C1_kernel[i] = ((float)rand() / RAND_MAX);
    }

    float *d_raw_data, *d_C1_data, *d_C1_kernel;

    cudaMalloc((void **)&d_raw_data, N1 * sizeof(float));
    cudaMalloc((void **)&d_C1_data, N2 * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, N4 * sizeof(float));

    cudaMemcpy(d_raw_data, raw_data, N1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, N4 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); 
    dim3 gridDim((28 + 16 - 1) / 16,  // Calcul de la grille
                 (28 + 16 - 1) / 16, 
                 6);  // 6 output channel

    convolve_2D<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data);

    cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, N2 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C1_data[0][0][0] = %f\n", C1_data[0]);

    // Libération de la mémoire
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}


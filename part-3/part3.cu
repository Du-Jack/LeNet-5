#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#define WIDTH 32
#define HEIGHT 32
#define KERNEL_SIZE 5
#define NUM_CHANNELS 6
#define OUTPUT_HEIGHT 28
#define OUTPUT_WIDTH 28
#define CHANNELS 6

void MatrixPrint(float *M, int channels, int height, int width) {
    for (int c = 0; c < channels; c++) {
        printf("Canal %d :\n", c);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                printf("%.2f ", M[c * height * width + h * width + w]);
            }
            printf("\n");
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

__global__ void subsample2D(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int c = blockIdx.z;  
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int w = blockIdx.x * blockDim.x + threadIdx.x;  

    if (h < outputHeight && w < outputWidth) {
        // Calcul de la position de la fenêtre 2x2 dans l'entrée
        int input_y = h * 2;
        int input_x = w * 2;
        
        // Moyenne des 4 pixels du bloc 2x2 dans l'entrée
        float sum = 0.0f;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int input_index = c * inputHeight * inputWidth + (input_y + dy) * inputWidth + (input_x + dx);
                sum += input[input_index];
            }
        }

        int output_index = c * outputHeight * outputWidth + h * outputWidth + w;
        output[output_index] = sum / 4.0f;  // avg of 4 pixels
    }
}

__device__ float activation_tanh(float M) {
    return tanh(M);
}

__global__ void apply_activation_tanh(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = activation_tanh(data[idx]);
    }
}

__device__ float activation_softmax(float *input, int idx, int length) {
    // Calcul du maximum pour la stabilité numérique
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Calcul exp + sum
    float exp_sum = 0.0f;
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - max_val); // Décalage pour la stabilité
        exp_sum += input[i];
    }

    // Normalisation
    return input[idx] / exp_sum;
}

__global__ void apply_activation_softmax(float *data, int size, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = activation_softmax(data, idx, length);
    }
}

__global__ void flatten(float *input, float *output, int width, int height, int channels) {
    int c = blockIdx.z;  
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < channels && h < height && w < width) {
        int input_idx = c * width * height + h * width + w;
        int output_idx = c * width * height + h * width + w;
        output[output_idx] = input[input_idx];
    }
}


__global__ void dense(float *input, float *weights, float *biases, float *output, int inputSize, int outputSize) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // index de sortie

    if (o < outputSize) {
        float sum = biases[o];  // Initialise avec le biais
        for (int i = 0; i < inputSize; i++) {
            int weight_index = o * inputSize + i;  // Accès aux poids
            sum += input[i] * weights[weight_index];
        }
        output[o] = sum;  // Résultat final
    }
}

void PrintFlattenedOutput(float *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
        if ((i + 1) % 14 == 0) {  // Exemple : saut de ligne après 14 éléments
            printf("\n");
        }
    }
    printf("\n");
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
    float *flattened_output;
    float *flattened_host = (float *)malloc(14 * 14 * 6 * sizeof(float));
    
    int inputSize = WIDTH * HEIGHT * CHANNELS;  // Input size
    int outputSize = 120;                       // Output size of Dense

    float *weights = (float *)malloc(inputSize * outputSize * sizeof(float));
    float *biases = (float *)malloc(outputSize * sizeof(float));
    float *dense_output = (float *)malloc(outputSize * sizeof(float));


    // init raw_data between 0 and 1
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N1; i++) {
        raw_data[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // Entre -10 et 10
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

    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;

    cudaMalloc((void **)&d_raw_data, N1 * sizeof(float));
    cudaMalloc((void **)&d_C1_data, N2 * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, N4 * sizeof(float));
    cudaMalloc((void **)&d_S1_data, N3 * sizeof(float));
    cudaMalloc((void **)&flattened_output, 14 * 14 * 6 * sizeof(float)); 


    cudaMemcpy(d_raw_data, raw_data, N1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, N4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, N3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C1_data, d_C1_data, N2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, N3 * sizeof(float), cudaMemcpyDeviceToHost);
    

    dim3 blockDim(16, 16); 
    dim3 gridDim((28 + 16 - 1) / 16,  
                 (28 + 16 - 1) / 16, 
                 6);  // 6 output channel

    convolve_2D<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data);

    subsample2D<<<gridDim, blockDim>>>(d_C1_data, d_S1_data, 28, 28, 14, 14);

    //apply_activation_tanh<<<(N3 + 255) / 256, 256>>>(d_S1_data, N3);
    apply_activation_softmax<<<(N3 + 255) / 256, 256>>>(d_S1_data, N3, 6);

    //cudaDeviceSynchronize();

    
    
    //printf("raw data. \n");
    //MatrixPrint(raw_data, 6, 14, 14);
    //printf("C1 data. \n");
    //MatrixPrint(C1_data, 6, 14, 14);
    //printf("S1 data. \n");
    //printf("Activation output. \n");
    //printf("Softmax output. \n");
    //MatrixPrint(S1_data, 6, 14, 14);


    // Test for Dense and Flatten
    dim3 blockDimFlatten(16, 16, 1);
    dim3 gridDimFlatten((WIDTH + 15) / 16, (HEIGHT + 15) / 16, CHANNELS);


    // Call Flatten
    flatten<<<gridDimFlatten, blockDimFlatten>>>(d_S1_data, flattened_output, WIDTH, HEIGHT, CHANNELS);
    cudaMemcpy(flattened_host, flattened_output, 14 * 14 * 6 * sizeof(float), cudaMemcpyDeviceToHost);
    //printf("Flatten output:\n");
    //PrintFlattenedOutput(flattened_host, 14 * 14 * 6);
    

    // Call Dense
    for (int i = 0; i < inputSize * outputSize; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Poids entre -1 et 1
    }
    for (int i = 0; i < outputSize; i++) {
        biases[i] = 0.0f;  // Biais initialisés à 0
    }

    float *d_weights, *d_biases, *d_dense_output;
    cudaMalloc((void **)&d_weights, inputSize * outputSize * sizeof(float));
    cudaMalloc((void **)&d_biases, outputSize * sizeof(float));
    cudaMalloc((void **)&d_dense_output, outputSize * sizeof(float));

    cudaMemcpy(d_weights, weights, inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, outputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Call of Dense
    dim3 blockDimDense(256);
    dim3 gridDimDense((outputSize + 255) / 256);
    dense<<<gridDimDense, blockDimDense>>>(flattened_output, d_weights, d_biases, d_dense_output, inputSize, outputSize);
    cudaDeviceSynchronize();
    
    // Retrieve output
    cudaMemcpy(dense_output, d_dense_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Display Dense output
    printf("Dense layer output:\n");
    for (int i = 0; i < outputSize; i++) {
        printf("%f ", dense_output[i]);
    }
    printf("\n");


    // Free memory
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(flattened_host);
    free(weights);
    free(biases);
    free(dense_output);
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_S1_data);
    cudaFree(flattened_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_dense_output);

    return 0;
}
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

__global__ void convolve_multichannel(float *input, float *kernels, float *output,
                                      int input_height, int input_width,
                                      int kernel_size, int input_channels,
                                      int output_channels, int output_height,
                                      int output_width) {
    int o_c = blockIdx.z;  // Output channel index
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int w = blockIdx.x * blockDim.x + threadIdx.x;  

    if (o_c < output_channels && h < output_height && w < output_width) {
        float sum = 0.0f;
        for (int i_c = 0; i_c < input_channels; i_c++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int y = h + kh;  
                    int x = w + kw;
                    int input_idx = i_c * input_height * input_width + y * input_width + x;
                    int kernel_idx = o_c * input_channels * kernel_size * kernel_size +
                                     i_c * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input[input_idx] * kernels[kernel_idx];
                }
            }
        }
        int output_idx = o_c * output_height * output_width + h * output_width + w;
        output[output_idx] = sum;
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

void PrintFlattenedOutput(float *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
        if ((i + 1) % 14 == 0) {  // Saut de ligne après 14 éléments
            printf("\n");
        }
    }
    printf("\n");
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


// PRINT IMG
void charBckgrndPrint(char *str, int rgb[3]){
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
    int row, col;
    char *str="  ";
    for(row=0; row<height; row++){
        for(col=0; col<width; col++){
            charBckgrndPrint(str,img[row][col]);
        }
        printf("\n");
    }
}

void loadMNIST(float *data, const char *filename, int imgIdx, int rows, int cols) {
    FILE *fptr;
    if ((fptr = fopen(filename, "rb")) == NULL) {
        printf("Can't open file %s\n", filename);
        exit(1);
    }

    fseek(fptr, 16 + imgIdx * rows * cols, SEEK_SET);  // Offset pour aller à l'image imgIdx
    unsigned char pixel;
    for (int i = 0; i < rows * cols; i++) {
        fread(&pixel, sizeof(unsigned char), 1, fptr);
        data[i] = (float)pixel / 255.0f;  // Normalise les valeurs entre 0 et 1
    }

    fclose(fptr);
}


int main(int argc, char *argv[]) {
    // ---- Dimensions ----
    const int INPUT_SIZE = 28 * 28;
    const int C1_OUTPUT_SIZE = 28 * 28 * 6; // Conv2D -> (28, 28, 6)
    const int S2_OUTPUT_SIZE = 14 * 14 * 6; // Pooling -> (14, 14, 6)
    const int C3_OUTPUT_SIZE = 10 * 10 * 16; // Conv2D -> (10, 10, 16)
    const int S4_OUTPUT_SIZE = 5 * 5 * 16; // Pooling -> (5, 5, 16)
    const int FLATTEN_SIZE = 5 * 5 * 16; // Après flatten (1D)
    const int F5_OUTPUT_SIZE = 120; // Dense vers 120
    const int F6_OUTPUT_SIZE = 84; // Dense vers 84
    const int FINAL_OUTPUT_SIZE = 10; // Dense vers 10 (sortie)

    // ---- Allocate memory ----
    float *raw_data = (float *)malloc(INPUT_SIZE * sizeof(float));
    float *C1_kernel = (float *)malloc(6 * 5 * 5 * sizeof(float));
    float *C1_output = (float *)malloc(C1_OUTPUT_SIZE * sizeof(float));
    float *S2_output = (float *)malloc(S2_OUTPUT_SIZE * sizeof(float));
    float *C3_kernel = (float *)malloc(16 * 6 * 5 * 5 * sizeof(float));
    float *C3_output = (float *)malloc(C3_OUTPUT_SIZE * sizeof(float));
    float *S4_output = (float *)malloc(S4_OUTPUT_SIZE * sizeof(float));
    float *flattened = (float *)malloc(FLATTEN_SIZE * sizeof(float));
    float *F5_weights = (float *)malloc(F5_OUTPUT_SIZE * FLATTEN_SIZE * sizeof(float));
    float *F5_biases = (float *)malloc(F5_OUTPUT_SIZE * sizeof(float));
    float *F5_output = (float *)malloc(F5_OUTPUT_SIZE * sizeof(float));
    float *F6_weights = (float *)malloc(F6_OUTPUT_SIZE * F5_OUTPUT_SIZE * sizeof(float));
    float *F6_biases = (float *)malloc(F6_OUTPUT_SIZE * sizeof(float));
    float *F6_output = (float *)malloc(F6_OUTPUT_SIZE * sizeof(float));
    float *FINAL_weights = (float *)malloc(F6_OUTPUT_SIZE * FINAL_OUTPUT_SIZE * sizeof(float));
    float *FINAL_biases = (float *)malloc(FINAL_OUTPUT_SIZE * sizeof(float));
    float *FINAL_output = (float *)malloc(FINAL_OUTPUT_SIZE * sizeof(float));

    // ---- Initialisation ----
    srand(time(NULL));
    
    // ---- Load MNIST Data ----
    const char *filename = "train-images.idx3-ubyte";
    int imgIdx = 0; // Index of the image to be loaded
    float *mnistImage = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
    loadMNIST(mnistImage, filename, imgIdx, HEIGHT, WIDTH);

    for (int i = 0; i < 6 * 5 * 5; i++) C1_kernel[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < 16 * 6 * 5 * 5; i++) C3_kernel[i] = (float)rand() / RAND_MAX;

    

    printf("Input Image:\n");
    for (int h = 0; h < HEIGHT; h++) {
        for (int w = 0; w < WIDTH; w++) {
            printf("%.1f ", mnistImage[h * WIDTH + w]);
        }
        printf("\n");
    }
    printf("\n");

    // Init for Dense
    for (int i = 0; i < F5_OUTPUT_SIZE * FLATTEN_SIZE; i++) F5_weights[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < F5_OUTPUT_SIZE; i++) F5_biases[i] = 0.0f;
    for (int i = 0; i < F6_OUTPUT_SIZE * F5_OUTPUT_SIZE; i++) F6_weights[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < F6_OUTPUT_SIZE; i++) F6_biases[i] = 0.0f;
    for (int i = 0; i < FINAL_OUTPUT_SIZE * F6_OUTPUT_SIZE; i++) FINAL_weights[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) FINAL_biases[i] = 0.0f;

    // Memory allocation 
    float *d_raw_data, *d_C1_output, *d_S2_output, *d_C3_output, *d_S4_output;
    cudaMalloc(&d_raw_data, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_C1_output, C1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_S2_output, S2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_C3_output, C3_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_S4_output, S4_OUTPUT_SIZE * sizeof(float));

    // Convolution 1 (C1)
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid_C1(28 / 16, 28 / 16, 6), block(16, 16);
    convolve_2D<<<grid_C1, block>>>(d_raw_data, C1_kernel, d_C1_output);

    // Pooling 1 (S2)
    subsample2D<<<grid_C1, block>>>(d_C1_output, d_S2_output, 28, 28, 14, 14);

    // Convolution 2 (C3)
    dim3 grid_C3(10 / 16, 10 / 16, 16);
    convolve_multichannel<<<grid_C3, block>>>(d_S2_output, C3_kernel, d_C3_output, 14, 14, 5, 6, 16, 10, 10);

    // Pooling 2 (S4)
    subsample2D<<<grid_C3, block>>>(d_C3_output, d_S4_output, 10, 10, 5, 5);

    // Flatten
    cudaMemcpy(S4_output, d_S4_output, S4_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(flattened, S4_output, FLATTEN_SIZE * sizeof(float));

    // Fully Connected Layers
    dense<<<1, 120>>>(flattened, F5_weights, F5_biases, F5_output, FLATTEN_SIZE, F5_OUTPUT_SIZE);
    dense<<<1, 84>>>(F5_output, F6_weights, F6_biases, F6_output, F5_OUTPUT_SIZE, F6_OUTPUT_SIZE);
    dense<<<1, 10>>>(F6_output, FINAL_weights, FINAL_biases, FINAL_output, F6_OUTPUT_SIZE, FINAL_OUTPUT_SIZE);

    // Output
    printf("Output : \n");
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) printf("%f ", FINAL_output[i]);
    printf("\n");

    // Free memory
    free(raw_data);
    free(C1_kernel);
    free(C1_output);
    free(S2_output);
    free(C3_kernel);
    free(C3_output);
    free(S4_output);
    free(flattened);
    free(F5_weights);
    free(F5_biases);
    free(F5_output);
    free(F6_weights);
    free(F6_biases);
    free(F6_output);
    free(FINAL_weights);
    free(FINAL_biases);
    free(FINAL_output);
    cudaFree(d_raw_data);
    cudaFree(d_C1_output);
    cudaFree(d_S2_output);
    cudaFree(d_C3_output);
    cudaFree(d_S4_output);
    return 0;
}
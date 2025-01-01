#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

#define WIDTH 28
#define HEIGHT 28
#define KERNEL_SIZE 5
#define NUM_CHANNELS 6
#define OUTPUT_HEIGHT 28
#define OUTPUT_WIDTH 28
#define CHANNELS 6


// Structure pour stocker les dimensions de chaque couche
typedef struct {
    int input_height;
    int input_width;
    int kernel_size;
    int num_channels;
    int output_height;
    int output_width;
} LayerDims;

LayerDims layers[4]; // Un tableau de structures pour stocker les dimensions de chaque couche

void init_layers() {
    // Initialisation de la couche C1 (Convolution 1)
    layers[0].input_height = 28;
    layers[0].input_width = 28;
    layers[0].kernel_size = 5;
    layers[0].num_channels = 6;
    layers[0].output_height = layers[0].input_height - layers[0].kernel_size + 1;
    layers[0].output_width = layers[0].input_width - layers[0].kernel_size + 1;

    // Initialisation de la couche S2 (Pooling 1)
    layers[1].input_height = layers[0].output_height;
    layers[1].input_width = layers[0].output_width;
    layers[1].kernel_size = 2;  // Supposons que le pooling utilise une fenêtre 2x2
    layers[1].num_channels = layers[0].num_channels;
    layers[1].output_height = layers[1].input_height / 2;
    layers[1].output_width = layers[1].input_width / 2;

    // Initialisation de la couche C3 (Convolution 2)
    layers[2].input_height = layers[1].output_height;
    layers[2].input_width = layers[1].output_width;
    layers[2].kernel_size = 5;
    layers[2].num_channels = 16; // Changement du nombre de canaux
    layers[2].output_height = layers[2].input_height - layers[2].kernel_size + 1;
    layers[2].output_width = layers[2].input_width - layers[2].kernel_size + 1;

    // Initialisation de la couche S4 (Pooling 2)
    layers[3].input_height = layers[2].output_height;
    layers[3].input_width = layers[2].output_width;
    layers[3].kernel_size = 2;  // Supposons que le pooling utilise une fenêtre 2x2
    layers[3].num_channels = layers[2].num_channels;
    layers[3].output_height = layers[3].input_height / 2;
    layers[3].output_width = layers[3].input_width / 2;
}

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

__global__ void convolve_2D(float *raw_data, float *C1_kernel, float *C1_data,
                             int input_height, int input_width,
                             int output_height, int output_width, 
                             int kernel_size, int num_channels) {
    int c = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int w = blockIdx.x * blockDim.x + threadIdx.x;  

    if (h < output_height && w < output_width) {
        float sum = 0.0f;

        for (int kh = 0; kh < kernel_size; kh++) { // Loop through kernel
            for (int kw = 0; kw < kernel_size; kw++) {
                int x = w + kw;  // x coordinate of input 
                int y = h + kh;  // y coordinate of input 
                int input_index = y * input_width + x;  // Index in raw_data
                int kernel_index = c * kernel_size * kernel_size + kh * kernel_size + kw;  // Index in C1_kernel
                sum += raw_data[input_index] * C1_kernel[kernel_index];
            }
        }

        // Output index
        int output_index = c * output_height * output_width + h * output_width + w;
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

__global__ void apply_activation_tanh(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanh(data[idx]);
    }
}

/*
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
*/

void softmax(float *input, int size, float *output) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum_exp;
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

void load_weights(const char *filename, float *weights, size_t size) {
    FILE *fptr = fopen(filename, "rb");
    if (fptr == NULL) {
        printf("Error: Cannot open file %s.\n", filename);
        exit(1);
    }
    fread(weights, sizeof(float), size, fptr);
    fclose(fptr);
    printf("Successfully loaded weights from %s\n", filename);
}

void load_biases(const char *filename, float *biases, size_t size) {
    FILE *fptr = fopen(filename, "rb");
    if (fptr == NULL) {
        printf("Error: Cannot open file %s.\n", filename);
        exit(1);
    }
    fread(biases, sizeof(float), size, fptr);
    fclose(fptr);
    printf("Successfully loaded biases from %s\n", filename);
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
    init_layers();

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

    // Load weights and biases
    load_weights("conv1_weights.dat", C1_kernel, 6 * 5 * 5);  // Poids de C1
    load_biases("conv1_biases.dat", F5_biases, 6);  // Biais de C1

    load_weights("conv3_weights.dat", C3_kernel, 16 * 6 * 5 * 5);  // Poids de C3
    load_biases("conv3_biases.dat", F5_biases, 16);  // Biais de C3

    load_weights("dense_c5_weights.dat", F5_weights, F5_OUTPUT_SIZE * FLATTEN_SIZE);  // Poids de C5
    load_biases("dense_c5_biases.dat", F5_biases, F5_OUTPUT_SIZE);  // Biais de C5

    load_weights("dense_f6_weights.dat", F6_weights, F6_OUTPUT_SIZE * F5_OUTPUT_SIZE);  // Poids de F6
    load_biases("dense_f6_biases.dat", F6_biases, F6_OUTPUT_SIZE);  // Biais de F6

    load_weights("dense_output_weights.dat", FINAL_weights, FINAL_OUTPUT_SIZE * F6_OUTPUT_SIZE);  // Poids de la sortie
    load_biases("dense_output_biases.dat", FINAL_biases, FINAL_OUTPUT_SIZE);  // Biais de la sortie
    for (int i = 0; i < 10; i++) {
        printf("%f ", C1_kernel[i]);
    }
    printf("\n");
    
    // ---- Load MNIST Data ----
    const char *filename = "train-images.idx3-ubyte";
    int imgIdx = 0; // Index of the image to be loaded
    loadMNIST(raw_data, filename, imgIdx, 28, 28);

    float *mnistImage = (float *)malloc(WIDTH * HEIGHT * sizeof(float));

    FILE *fptr;
    int indice_img = 0; // Par défaut, première image
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;

    // ---- Vérifiez les arguments ----
    if (argc > 1) {
        indice_img = atoi(argv[1]); // Charger l'image d'indice donné en argument
    }

    // ---- Allocation dynamique pour img ----
    int ***img = (int ***)malloc(HEIGHT * sizeof(int **));
    for (int i = 0; i < HEIGHT; i++) {
        img[i] = (int **)malloc(WIDTH * sizeof(int *));
        for (int j = 0; j < WIDTH; j++) {
            img[i][j] = (int *)malloc(3 * sizeof(int));
        }
    }

    // ---- Ouvrir le fichier MNIST ----
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Error: Cannot open file.\n");
        exit(1);
    }

    // ---- Lire l'en-tête MNIST ----
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    // ---- Positionner le curseur pour lire l'image spécifiée ----
    fseek(fptr, 16 + indice_img * WIDTH * HEIGHT, SEEK_SET);

    // ---- Lire l'image en niveaux de gris et remplir img ----
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr); // Lire la valeur d'un pixel
            img[i][j][0] = img[i][j][1] = img[i][j][2] = (int)val; // Valeurs RGB égales
        }
    }

    // ---- Afficher l'image ----
    printf("Input Image:\n");
    imgColorPrint(HEIGHT, WIDTH, img);


    


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
    // Copier les données d'image depuis le CPU vers le GPU
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_C1((layers[0].output_width + 15) / 16, (layers[0].output_height + 15) / 16, layers[0].num_channels);
    dim3 block(16, 16);

    float *d_C1_kernel;
    //cudaMalloc(&d_C1_kernel, 6 * 5 * 5 * sizeof(float));  // Allocations pour 6 filtres de taille 5x5
    cudaMalloc(&d_C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float));


    // Copier les données du noyau (C1_kernel) depuis le CPU vers le GPU
    //cudaMemcpy(d_C1_kernel, C1_kernel, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Appel de la fonction de convolution
    convolve_2D<<<grid_C1, block>>>(d_raw_data, d_C1_kernel, d_C1_output,
                                layers[0].input_height, layers[0].input_width, 
                                layers[0].output_height, layers[0].output_width, 
                                layers[0].kernel_size, layers[0].num_channels);

    
    // Synchronisation pour détecter toute erreur CUDA
    cudaDeviceSynchronize();

    // Appliquer la fonction d'activation tanh sur C1_output
    apply_activation_tanh<<<(layers[0].output_height * layers[0].output_width * layers[0].num_channels + 255) / 256, 256>>>(d_C1_output, layers[0].output_height * layers[0].output_width * layers[0].num_channels);

    // Synchronisation pour s'assurer que le calcul est terminé
    cudaDeviceSynchronize();
    
    /*
    // Copier les résultats du GPU vers la mémoire du CPU
    cudaMemcpy(C1_output, d_C1_output, C1_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(d_C1_kernel, C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Imprimer les résultats après la convolution
    for (int i = 0; i < 100; i++) {
        printf("C1_output[%d]: %f\n", i, C1_output[i]);
    }
    */

    // Pooling 1 (S2)
    subsample2D<<<grid_C1, block>>>(d_C1_output, d_S2_output, 
                                layers[0].output_height, layers[0].output_width, 
                                layers[1].output_height, layers[1].output_width);

    
    // Synchronisation pour s'assurer que le calcul sur le GPU est terminé
    cudaDeviceSynchronize();

    
    // Copier les résultats de la sortie de pooling (S2) depuis le GPU vers le CPU
    cudaMemcpy(S2_output, d_S2_output, layers[1].output_height * layers[1].output_width * layers[1].num_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Imprimer les résultats après le pooling (S2)
    for (int i = 0; i < layers[1].output_height * layers[1].output_width * layers[1].num_channels; i++) {
        printf("S2_output[%d]: %f\n", i, S2_output[i]);
    }

    // Convolution 2 (C3)
    dim3 grid_C3((layers[2].output_width + 15) / 16, (layers[2].output_height + 15) / 16, layers[2].num_channels);
    convolve_multichannel<<<grid_C3, block>>>(d_S2_output, C3_kernel, d_C3_output, 
                                                layers[1].output_height, layers[1].output_width, 
                                                layers[2].output_height, layers[2].output_width, 
                                                layers[1].kernel_size, layers[1].num_channels, 
                                                layers[2].num_channels);

    // Synchronisation pour détecter toute erreur CUDA
    cudaDeviceSynchronize();
    
    // Pooling 2 (S4)
    subsample2D<<<grid_C3, block>>>(d_C3_output, d_S4_output, 10, 10, 5, 5);

    // Flatten
    cudaMemcpy(S4_output, d_S4_output, S4_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(flattened, S4_output, FLATTEN_SIZE * sizeof(float));

    // Fully Connected Layers
    dense<<<1, 120>>>(flattened, F5_weights, F5_biases, F5_output, FLATTEN_SIZE, F5_OUTPUT_SIZE);
    dense<<<1, 84>>>(F5_output, F6_weights, F6_biases, F6_output, F5_OUTPUT_SIZE, F6_OUTPUT_SIZE);
    dense<<<1, 10>>>(F6_output, FINAL_weights, FINAL_biases, FINAL_output, F6_OUTPUT_SIZE, FINAL_OUTPUT_SIZE);

    // activation softmax
    float *softmax_output = (float *)malloc(FINAL_OUTPUT_SIZE * sizeof(float));
    softmax(FINAL_output, FINAL_OUTPUT_SIZE, softmax_output);

    printf("Output (Probabilities):\n");
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) {
        printf("%f ", softmax_output[i]);
    }
    printf("\n");
    

    // Output
    printf("Output : \n");
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) printf("%f ", FINAL_output[i]);
    printf("\n");

    // Free memory
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);
    free(mnistImage);
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
    free(softmax_output);

    return 0;
}
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


// Structure to store the dimensions of each layer
typedef struct {
    int input_height;
    int input_width;
    int kernel_size;
    int num_channels;
    int output_height;
    int output_width;
    int num_neurons;
} LayerDims;

LayerDims layers[4]; // Array of structures to store the dimensions of each layer

void init_layers() {
    // Initialize layer C1 (Convolution 1)
    layers[0].input_height = 28;
    layers[0].input_width = 28;
    layers[0].kernel_size = 5;
    layers[0].num_channels = 6;
    layers[0].output_height = layers[0].input_height - layers[0].kernel_size + 1;
    layers[0].output_width = layers[0].input_width - layers[0].kernel_size + 1;

    // Initialize layer S2 (Pooling 1)
    layers[1].input_height = layers[0].output_height;
    layers[1].input_width = layers[0].output_width;
    layers[1].kernel_size = 2;  // Assume pooling uses a 2x2 window
    layers[1].num_channels = layers[0].num_channels;
    layers[1].output_height = layers[1].input_height / 2;
    layers[1].output_width = layers[1].input_width / 2;

    // Initialize layer C3 (Convolution 2)
    layers[2].input_height = layers[1].output_height;
    layers[2].input_width = layers[1].output_width;
    layers[2].kernel_size = 5;
    layers[2].num_channels = 16; // Change number of channels
    layers[2].output_height = layers[2].input_height - layers[2].kernel_size + 1;
    layers[2].output_width = layers[2].input_width - layers[2].kernel_size + 1;

    // Initialize layer S4 (Pooling 2)
    layers[3].input_height = layers[2].output_height;
    layers[3].input_width = layers[2].output_width;
    layers[3].kernel_size = 2;  // Assume pooling uses a 2x2 window
    layers[3].num_channels = layers[2].num_channels;
    layers[3].output_height = layers[3].input_height / 2;
    layers[3].output_width = layers[3].input_width / 2;

    layers[4].num_neurons = 120; // C5
    layers[5].num_neurons = 84;  // F6
    layers[6].num_neurons = 10;  // Final layer

}

void MatrixPrint(float *M, int channels, int height, int width) {
    for (int c = 0; c < channels; c++) {
        printf("Channel %d :\n", c);
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
       // M2 columns
       for (int j = 0; j < p; j++) {
           // Initialize Mout[i * p + j] to zero before addition
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

__global__ void subsample2D(float *input, float *output,
                            int input_height, int input_width,
                            int output_height, int output_width,
                            int kernel_size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z;

    if (x < output_width && y < output_height) {
        float sum = 0.0f;

        // Average pooling
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_x = x * kernel_size + i;
                int input_y = y * kernel_size + j;

                int input_idx = c * input_height * input_width + input_y * input_width + input_x;
                sum += input[input_idx];
            }
        }

        int output_idx = c * output_height * output_width + y * output_width + x;
        output[output_idx] = sum / (kernel_size * kernel_size);  // Average
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
                    if (input_idx < input_channels * input_height * input_width && kernel_idx < output_channels * input_channels * kernel_size * kernel_size) {
                        sum += input[input_idx] * kernels[kernel_idx];
                    }
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
        if ((i + 1) % 14 == 0) {  // Line break after 14 elements
            printf("\n");
        }
    }
    printf("\n");
}

__global__ void dense(float *input, float *weights, float *biases, float *output, int inputSize, int outputSize) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // output index

    if (o < outputSize) {
        float sum = biases[o];  // Initialize with bias
        for (int i = 0; i < inputSize; i++) {
            int weight_index = o * inputSize + i;  // Access weights
            sum += input[i] * weights[weight_index];
        }
        output[o] = sum;  // Final result
    }
}

__global__ void apply_activation_tanh(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanh(data[idx]);
    }
}

__global__ void apply_activation_softmax(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Step 1: Find the maximum value
        float max_val = data[0];
        for (int i = 1; i < size; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
            }
        }

        // Step 2: Subtract the maximum value to avoid overflows
        float exp_sum = 0.0f;
        float exp_val = expf(data[idx] - max_val);
        
        // Calculate the sum of exponentials
        for (int i = 0; i < size; i++) {
            exp_sum += expf(data[i] - max_val);
        }

        // Step 3: Apply normalization
        data[idx] = exp_val / exp_sum;
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

    fseek(fptr, 16 + imgIdx * rows * cols, SEEK_SET);  // Offset to the imgIdx image
    unsigned char pixel;
    for (int i = 0; i < rows * cols; i++) {
        fread(&pixel, sizeof(unsigned char), 1, fptr);
        data[i] = (float)pixel / 255.0f;  // Normalize values between 0 and 1
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
    const int FLATTEN_SIZE = 5 * 5 * 16; // After flattening (1D)
    const int F5_OUTPUT_SIZE = 120; // Dense layer to 120
    const int F6_OUTPUT_SIZE = 84; // Dense layer to 84
    const int FINAL_OUTPUT_SIZE = 10; // Dense layer to 10 (output)
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
    load_weights("conv1_weights.dat", C1_kernel, 6 * 5 * 5);
    load_biases("conv1_biases.dat", F5_biases, 6);

    load_weights("conv3_weights.dat", C3_kernel, 16 * 6 * 5 * 5);
    load_biases("conv3_biases.dat", F5_biases, 16);

    load_weights("dense_c5_weights.dat", F5_weights, F5_OUTPUT_SIZE * FLATTEN_SIZE);
    load_biases("dense_c5_biases.dat", F5_biases, F5_OUTPUT_SIZE);

    load_weights("dense_f6_weights.dat", F6_weights, F6_OUTPUT_SIZE * F5_OUTPUT_SIZE);
    load_biases("dense_f6_biases.dat", F6_biases, F6_OUTPUT_SIZE);

    load_weights("dense_output_weights.dat", FINAL_weights, FINAL_OUTPUT_SIZE * F6_OUTPUT_SIZE);
    load_biases("dense_output_biases.dat", FINAL_biases, FINAL_OUTPUT_SIZE);

    // Print the first 10 elements of C1_kernel as an example
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
    int indice_img = 0; // By default, first picture
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;

    // ---- Load image from MNIST dataset ----
    if (argc > 1) {
        indice_img = atoi(argv[1]); // Get the image index from the command line
    }

    // ---- Memory allocation for the image ----
    int ***img = (int ***)malloc(HEIGHT * sizeof(int **));
    for (int i = 0; i < HEIGHT; i++) {
        img[i] = (int **)malloc(WIDTH * sizeof(int *));
        for (int j = 0; j < WIDTH; j++) {
            img[i][j] = (int *)malloc(3 * sizeof(int));
        }
    }

    // ---- Open the MNIST file ----
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Error: Cannot open file.\n");
        exit(1);
    }

    // ---- Read the MNIST file ----
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    // ---- Position the cursor to read the specified image ----
    fseek(fptr, 16 + indice_img * WIDTH * HEIGHT, SEEK_SET);

    // ---- Read the grayscale image and fill img ----
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr); // Read the value of 1 pixel
            img[i][j][0] = img[i][j][1] = img[i][j][2] = (int)val; // Equal RGB values
        }
    }

    // ---- Print the image ----
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
    // Copy image data from CPU to GPU
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_C1((layers[0].output_width + 15) / 16, (layers[0].output_height + 15) / 16, layers[0].num_channels);
    dim3 block(16, 16);

    float *d_C1_kernel;
    cudaMalloc(&d_C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float));


    // Copy kernel data (C1_kernel) from CPU to GPU
    cudaMemcpy(d_C1_kernel, C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    
    
    // ------ Convolution 1 (C1) -------
    convolve_2D<<<grid_C1, block>>>(d_raw_data, d_C1_kernel, d_C1_output, layers[0].input_height, layers[0].input_width, layers[0].output_height, layers[0].output_width, layers[0].kernel_size, layers[0].num_channels);
    cudaDeviceSynchronize();

    // Apply tanh activation function on C1_output
    apply_activation_tanh<<<(layers[0].output_height * layers[0].output_width * layers[0].num_channels + 255) / 256, 256>>>(d_C1_output, layers[0].output_height * layers[0].output_width * layers[0].num_channels);
    cudaDeviceSynchronize();
    
    /* To debug the output of the convolution C1
    // Copy results from GPU to CPU memory
    cudaMemcpy(d_C1_kernel, C1_kernel, layers[0].num_channels * layers[0].kernel_size * layers[0].kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Print the results after convolution 1 (C1)
    for (int i = 0; i < 100; i++) {
        printf("C1_output[%d]: %f\n", i, C1_output[i]);
    }
    */


    // ------------- Pooling 1 (S2) -------------
    subsample2D<<<grid_C1, block>>>(d_C1_output, d_S2_output, layers[0].output_height, layers[0].output_width, layers[1].output_height, layers[1].output_width, layers[1].kernel_size);
    cudaDeviceSynchronize();

    /*
    // Copy the results from GPU to CPU memory
    cudaMemcpy(S2_output, d_S2_output, layers[1].output_height * layers[1].output_width * layers[1].num_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results after pooling 1 (S2)
    for (int i = 0; i < layers[1].output_height * layers[1].output_width * layers[1].num_channels; i++) {
        printf("S2_output[%d]: %f\n", i, S2_output[i]);
    }
    */


    // -------- Convolution 2 (C3) --------
    dim3 grid_C3((layers[2].output_width + 15) / 16, (layers[2].output_height + 15) / 16, layers[2].num_channels);
       
    float *d_C3_kernel;
    cudaMalloc(&d_C3_kernel, layers[2].num_channels * layers[1].num_channels * layers[1].kernel_size * layers[1].kernel_size * sizeof(float));
    cudaMemcpy(d_C3_kernel, C3_kernel, layers[2].num_channels * layers[1].num_channels * layers[1].kernel_size * layers[1].kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    convolve_multichannel<<<grid_C3, block>>>(d_S2_output, d_C3_kernel, d_C3_output, layers[1].output_height, layers[1].output_width, layers[2].kernel_size, layers[1].num_channels, layers[2].num_channels, layers[2].output_height, layers[2].output_width);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    /*
    cudaMemcpy(C3_output, d_C3_output, layers[2].output_height * layers[2].output_width * layers[2].num_channels * sizeof(float), cudaMemcpyDeviceToHost);
                                                
    // Print the results after convolution 2 (C3)
    for (int i = 0; i < layers[2].output_height * layers[2].output_width * layers[2].num_channels; i++) {
        printf("C3_output[%d]: %f\n", i, C3_output[i]);
    }
    */                                               
    
    cudaDeviceSynchronize();

    // Apply tanh activation function on C3_output
    apply_activation_tanh<<<(layers[0].output_height * layers[2].output_width * layers[2].num_channels + 255) / 256, 256>>>(d_C3_output, layers[2].output_height * layers[2].output_width * layers[2].num_channels);
    cudaDeviceSynchronize();
    

    // ------- Pooling 2 (S4) -------
    dim3 grid_S4((layers[3].output_width + 15) / 16, (layers[3].output_height + 15) / 16, layers[3].num_channels);
    subsample2D<<<grid_S4, block>>>(d_C3_output, d_S4_output, layers[2].output_height, layers[2].output_width, layers[3].output_height, layers[3].output_width, layers[3].kernel_size);

    cudaMemcpy(S4_output, d_S4_output, layers[3].output_height * layers[3].output_width * layers[3].num_channels * sizeof(float), cudaMemcpyDeviceToHost);

    // Fully Connected Layers
    int flatten_size = layers[3].output_height * layers[3].output_width * layers[3].num_channels;
    cudaMemcpy(flattened, S4_output, flatten_size * sizeof(float), cudaMemcpyDeviceToHost);

    // ------- Flatten F5 -------
    dense<<<1, layers[4].num_neurons>>>(flattened, F5_weights, F5_biases, F5_output, flatten_size, layers[4].num_neurons);
    apply_activation_tanh<<<(layers[4].num_neurons + 255) / 256, 256>>>(F5_output, layers[4].num_neurons);

    // ------- Flatten F6 -------
    dense<<<1, layers[5].num_neurons>>>(F5_output, F6_weights, F6_biases, F6_output, layers[4].num_neurons, layers[5].num_neurons);
    apply_activation_tanh<<<(layers[5].num_neurons + 255) / 256, 256>>>(F6_output, layers[5].num_neurons);

    // ------- Flatten FINAL -------
    dense<<<1, layers[6].num_neurons>>>(F6_output, FINAL_weights, FINAL_biases, FINAL_output, layers[5].num_neurons, layers[6].num_neurons);

    int num_threads = 256; // Number of threads per bloc
    int num_blocks = (FINAL_OUTPUT_SIZE + num_threads - 1) / num_threads; // Number of blocks

    apply_activation_softmax<<<num_blocks, num_threads>>>(FINAL_output, FINAL_OUTPUT_SIZE);
    cudaDeviceSynchronize();

    // Copy the results from the GPU to the CPU
    float *softmax_output = (float *)malloc(FINAL_OUTPUT_SIZE * sizeof(float));
    cudaMemcpy(softmax_output, FINAL_output, FINAL_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Output (Softmax Probabilities):\n");
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) {
        printf("%f ", softmax_output[i]);
    }
    printf("\n");

    // Check if the sum of the softmax output is close to 1.0
    float sum = 0.0f;
    for (int i = 0; i < FINAL_OUTPUT_SIZE; i++) {
        sum += softmax_output[i];
    }
    printf("Sum of softmax output: %f\n", sum);

    // ---- Free memory ----
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
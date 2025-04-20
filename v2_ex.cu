#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 1 // Naive single image processing
#define NUM_CLASSES 10
#define THREADS_PER_BLOCK 256

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// === V1: CPU Implementation ===

// Allocate matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free matrix
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// V1 Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetworkV1;


// V1 Activation functions
void relu_v1(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax_v1(double* x, int size) {
    double max_x = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_x) max_x = x[i];
    }
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_x);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// V1 Initialize network
NeuralNetworkV1* createNetworkV1() {
    NeuralNetworkV1* net = (NeuralNetworkV1*)malloc(sizeof(NeuralNetworkV1));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(42);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.02;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.02;

    return net;
}

// V1 Forward pass
void forward_v1(NeuralNetworkV1* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu_v1(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax_v1(output, OUTPUT_SIZE);
}

// V1 Backpropagation
void backward_v1(NeuralNetworkV1* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

// V1 Train
void train_v1(NeuralNetworkV1* net, double** images, double** labels, int numImages, double* total_time, double* test_accuracy) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward_v1(net, images[i], hidden, output);
            backward_v1(net, images[i], hidden, output, labels[i]);

            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k] + 1e-10);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("V1 Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    *total_time = get_time(total_start);
    printf("V1 Total training time: %.3fs\n", *total_time);
}

// V1 Evaluate
void evaluate_v1(NeuralNetworkV1* net, double** images, double** labels, int numImages, double* test_accuracy) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward_v1(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    *test_accuracy = (correct / (double)numImages) * 100;
    printf("V1 Test Accuracy: %.2f%%\n", *test_accuracy);
}

// V1 Free
void freeNetworkV1(NeuralNetworkV1* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// === V2: Naive GPU Implementation ===

// CUDA kernels
__global__ void matrixMulKernel(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void addBiasKernel(double* A, double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        A[idx] += bias[row];
    }
}

__global__ void reluKernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void softmaxKernel(double* x, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        double max_x = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_x) max_x = x[i];
        }
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            output[i] = exp(x[i] - max_x);
            sum += output[i];
        }
        for (int i = 0; i < size; i++) {
            output[i] = (sum > 0) ? output[i] / sum : 0.0;
        }
    }
}

__global__ void computeOutputGradKernel(double* output, double* target, double* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void computeHiddenGradKernel(double* W2, double* d_output, double* d_hidden, double* hidden, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        double sum = 0.0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + idx] * d_output[j];
        }
        d_hidden[idx] = sum * (hidden[idx] > 0 ? 1.0 : 0.0);
    }
}

__global__ void updateWeightsKernel(double* W, double* grad, double* input, double lr, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        W[row * cols + col] -= lr * grad[row] * input[col];
    }
}

__global__ void updateBiasKernel(double* b, double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] -= lr * grad[idx];
    }
}

// V2 Neural network structure
typedef struct {
    double *W1, *W2, *b1, *b2;
} NeuralNetworkV2;

// V2 Initialize
NeuralNetworkV2* createNetworkV2() {
    NeuralNetworkV2* net = (NeuralNetworkV2*)malloc(sizeof(NeuralNetworkV2));
    
    CUDA_CHECK(cudaMalloc(&net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b2, OUTPUT_SIZE * sizeof(double)));
    
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    srand(42);
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        h_W1[i] = ((double)rand() / RAND_MAX - 0.5) * 0.02;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        h_W2[i] = ((double)rand() / RAND_MAX - 0.5) * 0.02;
    
    CUDA_CHECK(cudaMemcpy(net->W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(net->b1, 0, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(net->b2, 0, OUTPUT_SIZE * sizeof(double)));
    
    free(h_W1);
    free(h_W2);
    return net;
}

// V2 Forward pass
void forward_v2(NeuralNetworkV2* net, double* d_input, double* d_hidden, double* d_output) {
    dim3 threads(16, 16);
    dim3 blocks((HIDDEN_SIZE + threads.x - 1) / threads.x, (1 + threads.y - 1) / threads.y);
    
    matrixMulKernel<<<blocks, threads>>>(net->W1, d_input, d_hidden, HIDDEN_SIZE, INPUT_SIZE, 1);
    CUDA_CHECK(cudaGetLastError());
    addBiasKernel<<<(HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_hidden, net->b1, 1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    reluKernel<<<(HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_hidden, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    blocks = dim3((OUTPUT_SIZE + threads.x - 1) / threads.x, (1 + threads.y - 1) / threads.y);
    matrixMulKernel<<<blocks, threads>>>(net->W2, d_hidden, d_output, OUTPUT_SIZE, HIDDEN_SIZE, 1);
    CUDA_CHECK(cudaGetLastError());
    addBiasKernel<<<(OUTPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_output, net->b2, 1, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    softmaxKernel<<<1, THREADS_PER_BLOCK>>>(d_output, d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// V2 Backward pass
void backward_v2(NeuralNetworkV2* net, double* d_input, double* d_hidden, double* d_output, double* d_target, double* d_d_output, double* d_d_hidden) {
    dim3 threads(16, 16);
    dim3 blocks;
    
    computeOutputGradKernel<<<(OUTPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    computeHiddenGradKernel<<<(HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(net->W2, d_d_output, d_d_hidden, d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    blocks = dim3((HIDDEN_SIZE + threads.x - 1) / threads.x, (OUTPUT_SIZE + threads.y - 1) / threads.y);
    updateWeightsKernel<<<blocks, threads>>>(net->W2, d_d_output, d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    blocks = dim3((INPUT_SIZE + threads.x - 1) / threads.x, (HIDDEN_SIZE + threads.y - 1) / threads.y);
    updateWeightsKernel<<<blocks, threads>>>(net->W1, d_d_hidden, d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    
    updateBiasKernel<<<(OUTPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(net->b2, d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());
    updateBiasKernel<<<(HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(net->b1, d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// V2 Train
void train_v2(NeuralNetworkV2* net, double** images, double** labels, int numImages, double* total_time, double* test_accuracy) {
    clock_t total_start = clock();
    double *d_input, *d_hidden, *d_output, *d_target, *d_d_output, *d_d_hidden;
    
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d_output, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i++) {
            CUDA_CHECK(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_target, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            
            forward_v2(net, d_input, d_hidden, d_output);
            backward_v2(net, d_input, d_hidden, d_output, d_target, d_d_output, d_d_hidden);
            
            CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                if (!isnan(h_output[k]) && h_output[k] > 1e-10) {
                    loss -= labels[i][k] * log(h_output[k]);
                }
            }
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        printf("V2 Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    
    *total_time = get_time(total_start);
    printf("V2 Total training time: %.3fs\n", *total_time);
    
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_d_output));
    CUDA_CHECK(cudaFree(d_d_hidden));
}

// V2 Evaluate
void evaluate_v2(NeuralNetworkV2* net, double** images, double** labels, int numImages, double* test_accuracy) {
    int correct = 0;
    double *d_input, *d_hidden, *d_output;
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double)));
    
    for (int i = 0; i < numImages; i++) {
        CUDA_CHECK(cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        forward_v2(net, d_input, d_hidden, d_output);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    *test_accuracy = (correct / (double)numImages) * 100;
    printf("V2 Test Accuracy: %.2f%%\n", *test_accuracy);
    
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
}

// V2 Free
void freeNetworkV2(NeuralNetworkV2* net) {
    CUDA_CHECK(cudaFree(net->W1));
    CUDA_CHECK(cudaFree(net->W2));
    CUDA_CHECK(cudaFree(net->b1));
    CUDA_CHECK(cudaFree(net->b2));
    free(net);
}

// === Dataset Loading ===
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// === Main ===
int main() {
    printf("MNIST Neural Network Comparison (V1: CPU vs V2: Naive GPU)\n\n");
    
    // Dataset paths
    const char* train_images_path = "/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-images-idx3-ubyte/train-images-idx3-ubyte";
    const char* train_labels_path = "/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    const char* test_images_path = "/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    const char* test_labels_path = "/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    
    printf("Loading datasets...\n");
    double** train_images = loadMNISTImages(train_images_path, 60000);
    double** train_labels = loadMNISTLabels(train_labels_path, 60000);
    double** test_images = loadMNISTImages(test_images_path, 10000);
    double** test_labels = loadMNISTLabels(test_labels_path, 10000);
    
    double v1_time, v2_time, v1_accuracy, v2_accuracy;
    
    printf("\n=== Running V1 (CPU) ===\n");
    NeuralNetworkV1* net_v1 = createNetworkV1();
    train_v1(net_v1, train_images, train_labels, 60000, &v1_time, &v1_accuracy);
    evaluate_v1(net_v1, test_images, test_labels, 10000, &v1_accuracy);
    freeNetworkV1(net_v1);
    
    CUDA_CHECK(cudaDeviceReset());
    
    printf("\n=== Running V2 (Naive GPU) ===\n");
    NeuralNetworkV2* net_v2 = createNetworkV2();
    train_v2(net_v2, train_images, train_labels, 60000, &v2_time, &v2_accuracy);
    evaluate_v2(net_v2, test_images, test_labels, 10000, &v2_accuracy);
    freeNetworkV2(net_v2);
    
    printf("\n=== Performance Comparison ===\n");
    printf("V1 (CPU) Total Time: %.3f seconds\n", v1_time);
    printf("V2 (GPU) Total Time: %.3f seconds\n", v2_time);
    printf("Speedup (V1/V2): %.2fx\n", v1_time / v2_time);
    printf("\n=== Accuracy Comparison ===\n");
    printf("V1 (CPU) Test Accuracy: %.2f%%\n", v1_accuracy);
    printf("V2 (GPU) Test Accuracy: %.2f%%\n", v2_accuracy);
    printf("Accuracy Difference (V1 - V2): %.2f%%\n", v1_accuracy - v2_accuracy);
    
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}
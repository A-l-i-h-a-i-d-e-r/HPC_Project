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
#define BATCH_SIZE 64
#define NUM_CLASSES 10

// Check CUDA errors
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Timer function for CPU
double get_cpu_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Timer function for GPU
float get_gpu_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds / 1000.0; // Convert to seconds
}

// Allocate memory for a matrix on host
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory on host
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions (CPU)
void relu_cpu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax_cpu(double* x, int size) {
    double max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// CUDA kernels
__global__ void matrixMulKernel(double* A, double* B, double* C, double* bias, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        double sum = bias[row];
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void reluKernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void softmaxKernel(double* x, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double max = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max) max = x[i];
        }
        double sum = 0;
        for (int i = 0; i < size; i++) {
            x[i] = exp(x[i] - max);
            sum += x[i];
        }
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }
}

__global__ void outputGradientKernel(double* output, double* target, double* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void hiddenGradientKernel(double* W2, double* d_output, double* hidden, double* d_hidden, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        double sum = 0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0);
    }
}

__global__ void updateWeightsKernel(double* W, double* grad, double* input, int rows, int cols, double lr) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        W[i * cols + j] -= lr * grad[i] * input[j];
    }
}

__global__ void updateBiasKernel(double* b, double* grad, int size, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        b[i] -= lr * grad[i];
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
    double *d_W1, *d_W2, *d_b1, *d_b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(42);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    CUDA_CHECK(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));

    double* temp_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* temp_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            temp_W1[i * INPUT_SIZE + j] = net->W1[i][j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            temp_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

    CUDA_CHECK(cudaMemcpy(net->d_W1, temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    free(temp_W1);
    free(temp_W2);

    return net;
}

// Forward pass (CPU)
void forward_cpu(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu_cpu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax_cpu(output, OUTPUT_SIZE);
}

// Forward pass (GPU)
void forward_gpu(NeuralNetwork* net, double* input, double* hidden, double* output) {
    double *d_input, *d_hidden, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((1 + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    matrixMulKernel<<<gridDim, blockDim>>>(net->d_W1, d_input, d_hidden, net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 256;
    int blocks = (HIDDEN_SIZE + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_hidden, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    gridDim = dim3((1 + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    matrixMulKernel<<<gridDim, blockDim>>>(net->d_W2, d_hidden, d_output, net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    softmaxKernel<<<1, 32>>>(d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
}

// Backward pass (CPU)
void backward_cpu(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
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

// Backward pass (GPU)
void backward_gpu(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double *d_input, *d_hidden, *d_output, *d_target, *d_d_output, *d_d_hidden;
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d_output, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_d_hidden, HIDDEN_SIZE * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (OUTPUT_SIZE + threads - 1) / threads;
    outputGradientKernel<<<blocks, threads>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks = (HIDDEN_SIZE + threads - 1) / threads;
    hiddenGradientKernel<<<blocks, threads>>>(net->d_W2, d_d_output, d_hidden, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 blockDim(16, 16);
    dim3 gridDim((HIDDEN_SIZE + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    updateWeightsKernel<<<gridDim, blockDim>>>(net->d_W2, d_d_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    CUDA_CHECK(cudaDeviceSynchronize());

    gridDim = dim3((INPUT_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    updateWeightsKernel<<<gridDim, blockDim>>>(net->d_W1, d_d_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks = (OUTPUT_SIZE + threads - 1) / threads;
    updateBiasKernel<<<blocks, threads>>>(net->d_b2, d_d_output, OUTPUT_SIZE, LEARNING_RATE);
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks = (HIDDEN_SIZE + threads - 1) / threads;
    updateBiasKernel<<<blocks, threads>>>(net->d_b1, d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
    CUDA_CHECK(cudaDeviceSynchronize());

    double* temp_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* temp_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    CUDA_CHECK(cudaMemcpy(temp_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(temp_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = temp_W1[i * INPUT_SIZE + j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = temp_W2[i * HIDDEN_SIZE + j];

    free(temp_W1);
    free(temp_W2);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_d_output));
    CUDA_CHECK(cudaFree(d_d_hidden));
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages, bool use_gpu, double* total_time, double* loss_out, double* train_acc_out) {
    double loss = 0.0;
    int correct = 0;

    if (use_gpu) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            loss = 0.0;
            correct = 0;

            for (int i = 0; i < numImages; i++) {
                double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
                forward_gpu(net, images[i], hidden, output);
                backward_gpu(net, images[i], hidden, output, labels[i]);

                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    loss -= labels[i][k] * log(output[k] + 1e-10);
                }
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                    if (labels[i][j] > labels[i][actual]) actual = j;
                }
                if (pred == actual) correct++;
            }

            printf("GPU Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
                   epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
            if (epoch == EPOCHS - 1) {
                *loss_out = loss / numImages;
                *train_acc_out = (correct / (double)numImages) * 100;
            }
        }

        CUDA_CHECK(cudaEventRecord(stop));
        *total_time = get_gpu_time(start, stop);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        printf("GPU Total training time: %.3fs\n", *total_time);
    } else {
        clock_t total_start = clock();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            loss = 0.0;
            correct = 0;

            for (int i = 0; i < numImages; i++) {
                double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
                forward_cpu(net, images[i], hidden, output);
                backward_cpu(net, images[i], hidden, output, labels[i]);

                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    loss -= labels[i][k] * log(output[k] + 1e-10);
                }
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[j] > output[pred]) pred = j;
                    if (labels[i][j] > labels[i][actual]) actual = j;
                }
                if (pred == actual) correct++;
            }

            printf("CPU Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
                   epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
            if (epoch == EPOCHS - 1) {
                *loss_out = loss / numImages;
                *train_acc_out = (correct / (double)numImages) * 100;
            }
        }

        *total_time = get_cpu_time(total_start);
        printf("CPU Total training time: %.3fs\n", *total_time);
    }
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages, bool use_gpu, double* test_acc_out) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        if (use_gpu)
            forward_gpu(net, images[i], hidden, output);
        else
            forward_cpu(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    *test_acc_out = (correct / (double)numImages) * 100;
    printf("%s Test Accuracy: %.2f%%\n", use_gpu ? "GPU" : "CPU", *test_acc_out);
}

// Read MNIST dataset
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

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    CUDA_CHECK(cudaFree(net->d_W1));
    CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1));
    CUDA_CHECK(cudaFree(net->d_b2));
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network - V2 (Naive GPU with Fixed Timing)\n\n");

    // Load datasets
    double** train_images = loadMNISTImages("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-images-idx3-ubyte/train-images-idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", 10000);

    // CPU execution
    NeuralNetwork* net_cpu = createNetwork();
    double cpu_total_time, cpu_loss, cpu_train_acc, cpu_test_acc;
    printf("Running CPU implementation...\n");
    train(net_cpu, train_images, train_labels, 60000, false, &cpu_total_time, &cpu_loss, &cpu_train_acc);
    evaluate(net_cpu, test_images, test_labels, 10000, false, &cpu_test_acc);

    // GPU execution
    NeuralNetwork* net_gpu = createNetwork();
    double gpu_total_time, gpu_loss, gpu_train_acc, gpu_test_acc;
    printf("\nRunning GPU implementation...\n");
    train(net_gpu, train_images, train_labels, 60000, true, &gpu_total_time, &gpu_loss, &gpu_train_acc);
    evaluate(net_gpu, test_images, test_labels, 10000, true, &gpu_test_acc);

    // Result comparison
    printf("\n=== Result Comparison ===\n");
    printf("CPU Total Time: %.3fs\n", cpu_total_time);
    printf("GPU Total Time: %.3fs\n", gpu_total_time);
    printf("Speedup: %.2fx\n", cpu_total_time / gpu_total_time);
    printf("CPU Loss: %.4f\n", cpu_loss);
    printf("GPU Loss: %.4f\n", gpu_loss);
    printf("Loss Difference: %.6f\n", fabs(cpu_loss - gpu_loss));
    printf("CPU Train Accuracy: %.2f%%\n", cpu_train_acc);
    printf("GPU Train Accuracy: %.2f%%\n", gpu_train_acc);
    printf("Train Accuracy Difference: %.2f%%\n", fabs(cpu_train_acc - gpu_train_acc));
    printf("CPU Test Accuracy: %.2f%%\n", cpu_test_acc);
    printf("GPU Test Accuracy: %.2f%%\n", gpu_test_acc);
    printf("Test Accuracy Difference: %.2f%%\n", fabs(cpu_test_acc - gpu_test_acc));

    // Free memory
    freeNetwork(net_cpu);
    freeNetwork(net_gpu);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);

    return 0;
}
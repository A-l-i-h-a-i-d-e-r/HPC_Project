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

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
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
__global__ void matrixMulKernel(double* A, double* B, double* C, double* bias, int rowsA, int colsA, int colsB, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    if (row < rowsA && col < colsB && batch < batch_size) {
        double sum = bias[row];
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[batch * colsA + k];
        }
        C[batch * rowsA + row] = sum;
    }
}

__global__ void reluKernel(double* x, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (idx < size && batch < batch_size) {
        int offset = batch * size + idx;
        x[offset] = (x[offset] > 0) ? x[offset] : 0;
    }
}

__global__ void softmaxKernel(double* x, int size, int batch_size) {
    int batch = blockIdx.x;
    if (batch < batch_size) {
        int offset = batch * size;
        double max = x[offset];
        for (int i = 1; i < size; i++) {
            if (x[offset + i] > max) max = x[offset + i];
        }
        double sum = 0;
        for (int i = 0; i < size; i++) {
            x[offset + i] = exp(x[offset + i] - max);
            sum += x[offset + i];
        }
        for (int i = 0; i < size; i++) {
            x[offset + i] /= sum;
        }
    }
}

__global__ void outputGradientKernel(double* output, double* target, double* d_output, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (idx < size && batch < batch_size) {
        int offset = batch * size + idx;
        d_output[offset] = output[offset] - target[offset];
    }
}

__global__ void hiddenGradientKernel(double* W2, double* d_output, double* hidden, double* d_hidden, int hidden_size, int output_size, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (i < hidden_size && batch < batch_size) {
        double sum = 0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + i] * d_output[batch * output_size + j];
        }
        d_hidden[batch * hidden_size + i] = sum * (hidden[batch * hidden_size + i] > 0);
    }
}

__global__ void updateWeightsKernel(double* W, double* grad, double* input, int rows, int cols, double lr, int batch_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        double delta = 0;
        for (int b = 0; b < batch_size; b++) {
            delta += grad[b * rows + i] * input[b * cols + j];
        }
        W[i * cols + j] -= lr * delta / batch_size;
    }
}

__global__ void updateBiasKernel(double* b, double* grad, int size, double lr, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double delta = 0;
        for (int b = 0; b < batch_size; b++) {
            delta += grad[b * size + i];
        }
        b[i] -= lr * delta / batch_size;
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
    // GPU pointers
    double *d_W1, *d_W2, *d_b1, *d_b2;
    double *d_input, *d_hidden, *d_output, *d_target, *d_d_output, *d_d_hidden;
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

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_input, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));

    // Copy weights and biases to GPU
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
            net->W1[i][j] * input[j];
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
void forward_gpu(NeuralNetwork* net, double** inputs, double** hiddens, double** outputs, int batch_size) {
    // Copy batch inputs to GPU
    for (int b = 0; b < batch_size; b++) {
        CUDA_CHECK(cudaMemcpy(net->d_input + b * INPUT_SIZE, inputs[b], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((1 + 15) / 16, (HIDDEN_SIZE + 15) / 16, batch_size);
    matrixMulKernel<<<gridDim, blockDim>>>(net->d_W1, net->d_input, net->d_hidden, net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1, batch_size);

    int threads = 256;
    dim3 gridDimRelu((HIDDEN_SIZE + threads - 1) / threads, batch_size);
    reluKernel<<<gridDimRelu, threads>>>(net->d_hidden, HIDDEN_SIZE, batch_size);

    gridDim = dim3((1 + 15) / 16, (OUTPUT_SIZE + 15) / 16, batch_size);
    matrixMulKernel<<<gridDim, blockDim>>>(net->d_W2, net->d_hidden, net->d_output, net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1, batch_size);

    gridDim = dim3(batch_size, 1);
    softmaxKernel<<<gridDim, 1>>>(net->d_output, OUTPUT_SIZE, batch_size);

    // Copy results back
    for (int b = 0; b < batch_size; b++) {
        CUDA_CHECK(cudaMemcpy(hiddens[b], net->d_hidden + b * HIDDEN_SIZE, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outputs[b], net->d_output + b * OUTPUT_SIZE, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    }
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
void backward_gpu(NeuralNetwork* net, double** inputs, double** hiddens, double** outputs, double** targets, int batch_size) {
    // Copy batch targets to GPU
    for (int b = 0; b < batch_size; b++) {
        CUDA_CHECK(cudaMemcpy(net->d_target + b * OUTPUT_SIZE, targets[b], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->d_output + b * OUTPUT_SIZE, outputs[b], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->d_hidden + b * HIDDEN_SIZE, hiddens[b], HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(net->d_input + b * INPUT_SIZE, inputs[b], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    }

    int threads = 256;
    dim3 gridDimOutput((OUTPUT_SIZE + threads - 1) / threads, batch_size);
    outputGradientKernel<<<gridDimOutput, threads>>>(net->d_output, net->d_target, net->d_d_output, OUTPUT_SIZE, batch_size);

    dim3 gridDimHidden((HIDDEN_SIZE + threads - 1) / threads, batch_size);
    hiddenGradientKernel<<<gridDimHidden, threads>>>(net->d_W2, net->d_d_output, net->d_hidden, net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);

    dim3 blockDim(16, 16);
    dim3 gridDimW2((HIDDEN_SIZE + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    updateWeightsKernel<<<gridDimW2, blockDim>>>(net->d_W2, net->d_d_output, net->d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE, batch_size);

    dim3 gridDimW1((INPUT_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    updateWeightsKernel<<<gridDimW1, blockDim>>>(net->d_W1, net->d_d_hidden, net->d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE, batch_size);

    dim3 gridDimB2((OUTPUT_SIZE + threads - 1) / threads, 1);
    updateBiasKernel<<<gridDimB2, threads>>>(net->d_b2, net->d_d_output, OUTPUT_SIZE, LEARNING_RATE, batch_size);

    dim3 gridDimB1((HIDDEN_SIZE + threads - 1) / threads, 1);
    updateBiasKernel<<<gridDimB1, threads>>>(net->d_b1, net->d_d_hidden, HIDDEN_SIZE, LEARNING_RATE, batch_size);

    // Copy updated weights and biases back to host
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
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages, bool use_gpu, double* total_time, double* loss_out, double* train_acc_out) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int current_batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - i);
            double* hidden[BATCH_SIZE];
            double* output[BATCH_SIZE];
            for (int b = 0; b < current_batch_size; b++) {
                hidden[b] = (double*)malloc(HIDDEN_SIZE * sizeof(double));
                output[b] = (double*)malloc(OUTPUT_SIZE * sizeof(double));
            }

            if (use_gpu) {
                forward_gpu(net, &images[i], hidden, output, current_batch_size);
                backward_gpu(net, &images[i], hidden, output, &labels[i], current_batch_size);
            } else {
                for (int b = 0; b < current_batch_size; b++) {
                    forward_cpu(net, images[i + b], hidden[b], output[b]);
                    backward_cpu(net, images[i + b], hidden[b], output[b], labels[i + b]);
                }
            }

            for (int b = 0; b < current_batch_size; b++) {
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    loss -= labels[i + b][k] * log(output[b][k] + 1e-10);
                }
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (output[b][j] > output[b][pred]) pred = j;
                    if (labels[i + b][j] > labels[i + b][actual]) actual = j;
                }
                if (pred == actual) correct++;
                free(hidden[b]);
                free(output[b]);
            }
        }

        printf("%s Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               use_gpu ? "GPU" : "CPU", epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
        if (epoch == EPOCHS - 1) {
            *loss_out = loss / numImages;
            *train_acc_out = (correct / (double)numImages) * 100;
        }
    }
    *total_time = get_time(total_start);
    printf("%s Total training time: %.3fs\n", use_gpu ? "GPU" : "CPU", *total_time);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages, bool use_gpu, double* test_acc_out) {
    int correct = 0;
    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        int current_batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - i);
        double* hidden[BATCH_SIZE];
        double* output[BATCH_SIZE];
        for (int b = 0; b < current_batch_size; b++) {
            hidden[b] = (double*)malloc(HIDDEN_SIZE * sizeof(double));
            output[b] = (double*)malloc(OUTPUT_SIZE * sizeof(double));
        }

        if (use_gpu) {
            forward_gpu(net, &images[i], hidden, output, current_batch_size);
        } else {
            for (int b = 0; b < current_batch_size; b++) {
                forward_cpu(net, images[i + b], hidden[b], output[b]);
            }
        }

        for (int b = 0; b < current_batch_size; b++) {
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[b][j] > output[b][pred]) pred = j;
                if (labels[i + b][j] > labels[i + b][actual]) actual = j;
            }
            if (pred == actual) correct++;
            free(hidden[b]);
            free(output[b]);
        }
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
    CUDA_CHECK(cudaFree(net->d_input));
    CUDA_CHECK(cudaFree(net->d_hidden));
    CUDA_CHECK(cudaFree(net->d_output));
    CUDA_CHECK(cudaFree(net->d_target));
    CUDA_CHECK(cudaFree(net->d_d_output));
    CUDA_CHECK(cudaFree(net->d_d_hidden));
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network - V2 (Naive GPU)\n\n");

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
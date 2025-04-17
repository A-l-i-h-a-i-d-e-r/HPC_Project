#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3
#define NUM_TRAIN 60000
#define NUM_TEST 10000

// CUDA error check
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Utility: Load MNIST images
float* load_images(const char* filename, int num_images) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    fseek(f, 16, SEEK_SET);  // skip header
    float* data = (float*)malloc(num_images * INPUT_SIZE * sizeof(float));
    for (int i = 0; i < num_images * INPUT_SIZE; i++) {
        unsigned char pixel;
        fread(&pixel, sizeof(unsigned char), 1, f);
        data[i] = pixel / 255.0f;
    }
    fclose(f);
    return data;
}

// Utility: Load MNIST labels
float* load_labels(const char* filename, int num_labels) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    fseek(f, 8, SEEK_SET);  // skip header
    float* data = (float*)calloc(num_labels * OUTPUT_SIZE, sizeof(float));
    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, f);
        data[i * OUTPUT_SIZE + label] = 1.0f;
    }
    fclose(f);
    return data;
}

// CUDA Kernel: ReLU Activation
__global__ void relu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) x[idx] = fmaxf(0.0f, x[idx]);
}

// CUDA Kernel: Softmax (per sample)
__device__ void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// CUDA Kernel: Forward pass (1 sample)
__global__ void forward_kernel(
    float* input, float* W1, float* b1, float* W2, float* b2,
    float* hidden, float* output
) {
    int sample = blockIdx.x;
    float* x = &input[sample * INPUT_SIZE];
    float* h = &hidden[sample * HIDDEN_SIZE];
    float* o = &output[sample * OUTPUT_SIZE];

    // Input to hidden
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            h[i] += W1[i * INPUT_SIZE + j] * x[j];
        h[i] = fmaxf(0.0f, h[i]);
    }

    // Hidden to output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        o[i] = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            o[i] += W2[i * HIDDEN_SIZE + j] * h[j];
    }
    softmax(o, OUTPUT_SIZE);
}

// CUDA Kernel: Accuracy evaluation
__global__ void accuracy_kernel(float* output, float* labels, int* correct) {
    int i = blockIdx.x;
    float* out = &output[i * OUTPUT_SIZE];
    float* lbl = &labels[i * OUTPUT_SIZE];

    int pred = 0, actual = 0;
    for (int j = 1; j < OUTPUT_SIZE; j++) {
        if (out[j] > out[pred]) pred = j;
        if (lbl[j] > lbl[actual]) actual = j;
    }
    if (pred == actual) atomicAdd(correct, 1);
}

int main() {
    printf("MNIST Neural Network (Naive CUDA GPU Version)\n");

    // Load data
    float* h_train_images = load_images("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-images-idx3-ubyte/train-images-idx3-ubyte", NUM_TRAIN);
    float* h_train_labels = load_labels("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte", NUM_TRAIN);
    float* h_test_images  = load_images("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", NUM_TEST);
    float* h_test_labels  = load_labels("/home/bscs-22i-1210/snap/snapd-desktop-integration/current/Desktop/project-root_HPC/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", NUM_TEST);

    // Allocate device memory
    float *d_train_images, *d_train_labels, *d_test_images, *d_test_labels;
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_hidden, *d_output;
    int* d_correct;

    size_t train_input_bytes = NUM_TEST * INPUT_SIZE * sizeof(float);
    size_t hidden_bytes = NUM_TEST * HIDDEN_SIZE * sizeof(float);
    size_t output_bytes = NUM_TEST * OUTPUT_SIZE * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_test_images, train_input_bytes));
    CUDA_CHECK(cudaMalloc(&d_test_labels, NUM_TEST * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));

    // Init weights
    float* h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1 = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    float* h_b2 = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) h_W1[i] = ((float)rand() / RAND_MAX) * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) h_W2[i] = ((float)rand() / RAND_MAX) * 0.01f;

    // Copy weights and data to GPU
    CUDA_CHECK(cudaMemcpy(d_test_images, h_test_images, train_input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_labels, h_test_labels, NUM_TEST * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Forward pass
    printf("Running forward pass on GPU for test data...\n");
    forward_kernel<<<NUM_TEST, 1>>>(d_test_images, d_W1, d_b1, d_W2, d_b2, d_hidden, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Accuracy evaluation
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
    accuracy_kernel<<<NUM_TEST, 1>>>(d_output, d_test_labels, d_correct);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_correct = 0;
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Test Accuracy: %.2f%%\n", (h_correct / (float)NUM_TEST) * 100.0f);

    // Cleanup
    cudaFree(d_train_images); cudaFree(d_train_labels);
    cudaFree(d_test_images); cudaFree(d_test_labels);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_hidden); cudaFree(d_output); cudaFree(d_correct);
    free(h_train_images); free(h_train_labels); free(h_test_images); free(h_test_labels);
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);

    return 0;
}

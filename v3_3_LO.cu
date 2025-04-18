// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <time.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>

// #define INPUT_SIZE 784
// #define HIDDEN_SIZE 128
// #define OUTPUT_SIZE 10
// #define LEARNING_RATE 0.01
// #define EPOCHS 3
// #define BATCH_SIZE 64
// #define NUM_CLASSES 10

// #define CUDA_CHECK(err) do { \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
//         exit(EXIT_FAILURE); \
//     } \
// } while(0)

// // Define precision (double, float)
// #define DATA_TYPE float
// typedef DATA_TYPE Real;

// // Configuration struct for launch parameters
// typedef struct {
//     int block_size_1d;     // Threads per block for 1D kernels
//     int block_size_2d_x;   // Block dimension X for 2D kernels
//     int block_size_2d_y;   // Block dimension Y for 2D kernels
//     int num_streams;       // Number of CUDA streams
// } Config;

// // Default configuration
// Config config = {
//     .block_size_1d = 256,
//     .block_size_2d_x = 16,
//     .block_size_2d_y = 16,
//     .num_streams = 3
// };

// double get_cpu_time(clock_t start) {
//     return (double)(clock() - start) / CLOCKS_PER_SEC;
// }

// float get_gpu_time(cudaEvent_t start, cudaEvent_t stop) {
//     float milliseconds = 0;
//     CUDA_CHECK(cudaEventSynchronize(stop));
//     CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
//     return milliseconds / 1000.0;
// }

// Real** allocateMatrix(int rows, int cols) {
//     Real** mat = (Real**)malloc(rows * sizeof(Real*));
//     for (int i = 0; i < rows; i++) {
//         mat[i] = (Real*)malloc(cols * sizeof(Real));
//     }
//     return mat;
// }

// void freeMatrix(Real** mat, int rows) {
//     for (int i = 0; i < rows; i++) {
//         free(mat[i]);
//     }
//     free(mat);
// }

// void relu_cpu(Real* x, int size) {
//     for (int i = 0; i < size; i++) {
//         x[i] = (x[i] > 0) ? x[i] : 0;
//     }
// }

// void softmax_cpu(Real* x, int size) {
//     Real max = x[0];
//     for (int i = 1; i < size; i++) {
//         if (x[i] > max) max = x[i];
//     }
//     Real sum = 0;
//     for (int i = 0; i < size; i++) {
//         x[i] = expf(x[i] - max);
//         sum += x[i];
//     }
//     for (int i = 0; i < size; i++) {
//         x[i] /= sum;
//     }
// }

// // Compute grid and block dimensions for 2D kernels
// void computeGridBlockDim(int rows, int cols, dim3* gridDim, dim3* blockDim, Config* config) {
//     blockDim->x = config->block_size_2d_x;
//     blockDim->y = config->block_size_2d_y;
//     gridDim->x = (cols + config->block_size_2d_x - 1) / config->block_size_2d_x;
//     gridDim->y = (rows + config->block_size_2d_y - 1) / config->block_size_2d_y;
// }

// // Compute blocks and threads for 1D kernels
// void compute1DGridBlockDim(int size, int* blocks, int* threads, Config* config) {
//     *threads = config->block_size_1d;
//     *blocks = (size + config->block_size_1d - 1) / config->block_size_1d;
// }

// __global__ void matrixMulKernel(Real* A, Real* B, Real* C, Real* bias, int rowsA, int colsA, int colsB) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < rowsA && col < colsB) {
//         Real sum = bias[row];
//         for (int k = 0; k < colsA; k++) {
//             sum += A[row * colsA + k] * B[k * colsB + col];
//         }
//         C[row * colsB + col] = sum;
//     }
// }

// __global__ void reluKernel(Real* x, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         x[idx] = (x[idx] > 0) ? x[idx] : 0;
//     }
// }

// __global__ void softmaxKernel(Real* x, int size) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         Real max = x[0];
//         for (int i = 1; i < size; i++) {
//             if (x[i] > max) max = x[i];
//         }
//         Real sum = 0;
//         for (int i = 0; i < size; i++) {
//             x[i] = expf(x[i] - max);
//             sum += x[i];
//         }
//         for (int i = 0; i < size; i++) {
//             x[i] /= sum;
//         }
//     }
// }

// __global__ void outputGradientKernel(Real* output, Real* target, Real* d_output, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         d_output[idx] = output[idx] - target[idx];
//     }
// }

// __global__ void hiddenGradientKernel(Real* W2, Real* d_output, Real* hidden, Real* d_hidden, int hidden_size, int output_size) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < hidden_size) {
//         Real sum = 0;
//         for (int j = 0; j < output_size; j++) {
//             sum += W2[j * hidden_size + i] * d_output[j];
//         }
//         d_hidden[i] = sum * (hidden[i] > 0);
//     }
// }

// __global__ void updateWeightsKernel(Real* W, Real* grad, Real* input, int rows, int cols, Real lr) {
//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < rows && j < cols) {
//         W[i * cols + j] -= lr * grad[i] * input[j];
//     }
// }

// __global__ void updateBiasKernel(Real* b, Real* grad, int size, Real lr) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         b[i] -= lr * grad[i];
//     }
// }

// typedef struct {
//     Real** W1;
//     Real** W2;
//     Real* b1;
//     Real* b2;
//     Real *d_W1, *d_W2, *d_b1, *d_b2;
// } NeuralNetwork;

// NeuralNetwork* createNetwork() {
//     NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
//     net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
//     net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
//     net->b1 = (Real*)calloc(HIDDEN_SIZE, sizeof(Real));
//     net->b2 = (Real*)calloc(OUTPUT_SIZE, sizeof(Real));

//     srand(42);
//     for (int i = 0; i < HIDDEN_SIZE; i++)
//         for (int j = 0; j < INPUT_SIZE; j++)
//             net->W1[i][j] = ((Real)rand() / RAND_MAX) * 0.01;

//     for (int i = 0; i < OUTPUT_SIZE; i++)
//         for (int j = 0; j < HIDDEN_SIZE; j++)
//             net->W2[i][j] = ((Real)rand() / RAND_MAX) * 0.01;

//     CUDA_CHECK(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
//     CUDA_CHECK(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));
//     CUDA_CHECK(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(Real)));
//     CUDA_CHECK(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(Real)));

//     Real* temp_W1;
//     Real* temp_W2;
//     CUDA_CHECK(cudaMallocHost(&temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
//     CUDA_CHECK(cudaMallocHost(&temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));
//     for (int i = 0; i < HIDDEN_SIZE; i++)
//         for (int j = 0; j < INPUT_SIZE; j++)
//             temp_W1[i * INPUT_SIZE + j] = net->W1[i][j];
//     for (int i = 0; i < OUTPUT_SIZE; i++)
//         for (int j = 0; j < HIDDEN_SIZE; j++)
//             temp_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

//     CUDA_CHECK(cudaMemcpy(net->d_W1, temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(net->d_W2, temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice));

//     CUDA_CHECK(cudaFreeHost(temp_W1));
//     CUDA_CHECK(cudaFreeHost(temp_W2));

//     return net;
// }

// void forward_cpu(NeuralNetwork* net, Real* input, Real* hidden, Real* output) {
//     for (int i = 0; i < HIDDEN_SIZE; i++) {
//         hidden[i] = net->b1[i];
//         for (int j = 0; j < INPUT_SIZE; j++)
//             hidden[i] += net->W1[i][j] * input[j];
//     }
//     relu_cpu(hidden, HIDDEN_SIZE);

//     for (int i = 0; i < OUTPUT_SIZE; i++) {
//         output[i] = net->b2[i];
//         for (int j = 0; j < HIDDEN_SIZE; j++)
//             output[i] += net->W2[i][j] * hidden[j];
//     }
//     softmax_cpu(output, OUTPUT_SIZE);
// }

// void forward_gpu(NeuralNetwork* net, Real* d_input, Real* d_hidden, Real* d_output, cudaStream_t stream, Config* config) {
//     dim3 blockDim, gridDim;
//     computeGridBlockDim(HIDDEN_SIZE, 1, &gridDim, &blockDim, config);
//     matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W1, d_input, d_hidden, net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1);

//     int blocks, threads;
//     compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
//     reluKernel<<<blocks, threads, 0, stream>>>(d_hidden, HIDDEN_SIZE);

//     computeGridBlockDim(OUTPUT_SIZE, 1, &gridDim, &blockDim, config);
//     matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W2, d_hidden, d_output, net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1);

//     compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
//     softmaxKernel<<<1, 32, 0, stream>>>(d_output, OUTPUT_SIZE);
// }

// void backward_cpu(NeuralNetwork* net, Real* input, Real* hidden, Real* output, Real* target) {
//     Real d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

//     for (int i = 0; i < OUTPUT_SIZE; i++)
//         d_output[i] = output[i] - target[i];

//     for (int i = 0; i < HIDDEN_SIZE; i++) {
//         d_hidden[i] = 0;
//         for (int j = 0; j < OUTPUT_SIZE; j++)
//             d_hidden[i] += net->W2[j][i] * d_output[j];
//         d_hidden[i] *= (hidden[i] > 0);
//     }

//     for (int i = 0; i < OUTPUT_SIZE; i++)
//         for (int j = 0; j < HIDDEN_SIZE; j++)
//             net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

//     for (int i = 0; i < HIDDEN_SIZE; i++)
//         for (int j = 0; j < INPUT_SIZE; j++)
//             net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

//     for (int i = 0; i < OUTPUT_SIZE; i++)
//         net->b2[i] -= LEARNING_RATE * d_output[i];

//     for (int i = 0; i < HIDDEN_SIZE; i++)
//         net->b1[i] -= LEARNING_RATE * d_hidden[i];
// }

// void backward_gpu(NeuralNetwork* net, Real* d_input, Real* d_hidden, Real* d_output, Real* d_target, Real* d_d_output, Real* d_d_hidden, cudaStream_t stream, Config* config) {
//     int blocks, threads;
//     compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
//     outputGradientKernel<<<blocks, threads, 0, stream>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);

//     compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
//     hiddenGradientKernel<<<blocks, threads, 0, stream>>>(net->d_W2, d_d_output, d_hidden, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);

//     dim3 blockDim, gridDim;
//     computeGridBlockDim(OUTPUT_SIZE, HIDDEN_SIZE, &gridDim, &blockDim, config);
//     updateWeightsKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W2, d_d_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);

//     computeGridBlockDim(HIDDEN_SIZE, INPUT_SIZE, &gridDim, &blockDim, config);
//     updateWeightsKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W1, d_d_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);

//     compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
//     updateBiasKernel<<<blocks, threads, 0, stream>>>(net->d_b2, d_d_output, OUTPUT_SIZE, LEARNING_RATE);

//     compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
//     updateBiasKernel<<<blocks, threads, 0, stream>>>(net->d_b1, d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
// }

// void train(NeuralNetwork* net, Real** images, Real** labels, int numImages, bool use_gpu, double* total_time, double* loss_out, double* train_acc_out, Config* config) {
//     double loss = 0.0;
//     int correct = 0;

//     if (use_gpu) {
//         Real *d_hidden, *d_output, *d_d_output, *d_d_hidden;
//         CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(Real)));
//         CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(Real)));
//         CUDA_CHECK(cudaMalloc(&d_d_output, OUTPUT_SIZE * sizeof(Real)));
//         CUDA_CHECK(cudaMalloc(&d_d_hidden, HIDDEN_SIZE * sizeof(Real)));

//         Real *d_batch_images[config->num_streams], *d_batch_labels[config->num_streams], *d_batch_outputs[config->num_streams];
//         Real *h_batch_outputs[config->num_streams];
//         cudaStream_t streams[config->num_streams];
//         for (int s = 0; s < config->num_streams; s++) {
//             CUDA_CHECK(cudaMalloc(&d_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMalloc(&d_batch_labels[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMalloc(&d_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMallocHost(&h_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaStreamCreate(&streams[s]));
//         }

//         Real* temp_W1;
//         Real* temp_W2;
//         CUDA_CHECK(cudaMallocHost(&temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
//         CUDA_CHECK(cudaMallocHost(&temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));

//         cudaEvent_t start, stop;
//         CUDA_CHECK(cudaEventCreate(&start));
//         CUDA_CHECK(cudaEventCreate(&stop));
//         CUDA_CHECK(cudaEventRecord(start));

//         for (int epoch = 0; epoch < EPOCHS; epoch++) {
//             loss = 0.0;
//             correct = 0;

//             for (int i = 0; i < numImages; i += BATCH_SIZE) {
//                 int stream_idx = (i / BATCH_SIZE) % config->num_streams;
//                 cudaStream_t stream = streams[stream_idx];
//                 int batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : numImages - i;

//                 for (int b = 0; b < batch_size; b++) {
//                     int img_idx = i + b;
//                     CUDA_CHECK(cudaMemcpyAsync(d_batch_images[stream_idx] + b * INPUT_SIZE, images[img_idx],
//                                                INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));
//                     CUDA_CHECK(cudaMemcpyAsync(d_batch_labels[stream_idx] + b * OUTPUT_SIZE, labels[img_idx],
//                                                OUTPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));
//                 }

//                 for (int b = 0; b < batch_size; b++) {
//                     Real* d_input = d_batch_images[stream_idx] + b * INPUT_SIZE;
//                     Real* d_target = d_batch_labels[stream_idx] + b * OUTPUT_SIZE;
//                     Real* d_batch_output = d_batch_outputs[stream_idx] + b * OUTPUT_SIZE;

//                     forward_gpu(net, d_input, d_hidden, d_output, stream, config);
//                     backward_gpu(net, d_input, d_hidden, d_output, d_target, d_d_output, d_d_hidden, stream, config);

//                     CUDA_CHECK(cudaMemcpyAsync(d_batch_output, d_output, OUTPUT_SIZE * sizeof(Real),
//                                                cudaMemcpyDeviceToDevice, stream));
//                 }

//                 CUDA_CHECK(cudaMemcpyAsync(h_batch_outputs[stream_idx], d_batch_outputs[stream_idx],
//                                            batch_size * OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost, stream));

//                 CUDA_CHECK(cudaStreamSynchronize(stream));

//                 for (int b = 0; b < batch_size; b++) {
//                     int img_idx = i + b;
//                     for (int k = 0; k < OUTPUT_SIZE; k++) {
//                         loss -= labels[img_idx][k] * log(h_batch_outputs[stream_idx][b * OUTPUT_SIZE + k] + 1e-10);
//                     }
//                     int pred = 0, actual = 0;
//                     for (int j = 0; j < OUTPUT_SIZE; j++) {
//                         if (h_batch_outputs[stream_idx][b * OUTPUT_SIZE + j] > h_batch_outputs[stream_idx][b * OUTPUT_SIZE + pred]) pred = j;
//                         if (labels[img_idx][j] > labels[img_idx][actual]) actual = j;
//                     }
//                     if (pred == actual) correct++;
//                 }
//             }

//             printf("GPU Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
//                    epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
//             if (epoch == EPOCHS - 1) {
//                 *loss_out = loss / numImages;
//                 *train_acc_out = (correct / (double)numImages) * 100;
//             }
//         }

//         CUDA_CHECK(cudaMemcpy(temp_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(temp_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
//         CUDA_CHECK(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));

//         CUDA_CHECK(cudaEventRecord(stop));
//         CUDA_CHECK(cudaEventSynchronize(stop));
//         *total_time = get_gpu_time(start, stop);

//         CUDA_CHECK(cudaEventDestroy(start));
//         CUDA_CHECK(cudaEventDestroy(stop));

//         for (int i = 0; i < HIDDEN_SIZE; i++)
//             for (int j = 0; j < INPUT_SIZE; j++)
//                 net->W1[i][j] = temp_W1[i * INPUT_SIZE + j];
//         for (int i = 0; i < OUTPUT_SIZE; i++)
//             for (int j = 0; j < HIDDEN_SIZE; j++)
//                 net->W2[i][j] = temp_W2[i * HIDDEN_SIZE + j];

//         CUDA_CHECK(cudaFreeHost(temp_W1));
//         CUDA_CHECK(cudaFreeHost(temp_W2));
//         for (int s = 0; s < config->num_streams; s++) {
//             CUDA_CHECK(cudaFreeHost(h_batch_outputs[s]));
//             CUDA_CHECK(cudaFree(d_batch_images[s]));
//             CUDA_CHECK(cudaFree(d_batch_labels[s]));
//             CUDA_CHECK(cudaFree(d_batch_outputs[s]));
//             CUDA_CHECK(cudaStreamDestroy(streams[s]));
//         }
//         CUDA_CHECK(cudaFree(d_hidden));
//         CUDA_CHECK(cudaFree(d_output));
//         CUDA_CHECK(cudaFree(d_d_output));
//         CUDA_CHECK(cudaFree(d_d_hidden));
//     } else {
//         clock_t total_start = clock();

//         for (int epoch = 0; epoch < EPOCHS; epoch++) {
//             loss = 0.0;
//             correct = 0;

//             for (int i = 0; i < numImages; i++) {
//                 Real hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
//                 forward_cpu(net, images[i], hidden, output);
//                 backward_cpu(net, images[i], hidden, output, labels[i]);

//                 for (int k = 0; k < OUTPUT_SIZE; k++) {
//                     loss -= labels[i][k] * log(output[k] + 1e-10);
//                 }
//                 int pred = 0, actual = 0;
//                 for (int j = 0; j < OUTPUT_SIZE; j++) {
//                     if (output[j] > output[pred]) pred = j;
//                     if (labels[i][j] > labels[i][actual]) actual = j;
//                 }
//                 if (pred == actual) correct++;
//             }

//             printf("CPU Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
//                    epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
//             if (epoch == EPOCHS - 1) {
//                 *loss_out = loss / numImages;
//                 *train_acc_out = (correct / (double)numImages) * 100;
//             }
//         }

//         *total_time = get_cpu_time(total_start);
//         printf("CPU Total training time: %.3fs\n", *total_time);
//     }
// }

// void evaluate(NeuralNetwork* net, Real** images, Real** labels, int numImages, bool use_gpu, double* test_acc_out, Config* config) {
//     int correct = 0;
//     if (use_gpu) {
//         Real *d_hidden, *d_output;
//         CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(Real)));
//         CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(Real)));

//         Real *h_batch_images[config->num_streams], *h_batch_outputs[config->num_streams];
//         Real *d_batch_images[config->num_streams], *d_batch_outputs[config->num_streams];
//         cudaStream_t streams[config->num_streams];
//         for (int s = 0; s < config->num_streams; s++) {
//             CUDA_CHECK(cudaMallocHost(&h_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMallocHost(&h_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMalloc(&d_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaMalloc(&d_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
//             CUDA_CHECK(cudaStreamCreate(&streams[s]));
//         }

//         for (int i = 0; i < numImages; i += BATCH_SIZE) {
//             int stream_idx = (i / BATCH_SIZE) % config->num_streams;
//             cudaStream_t stream = streams[stream_idx];
//             int batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : numImages - i;

//             for (int b = 0; b < batch_size; b++) {
//                 int img_idx = i + b;
//                 for (int j = 0; j < INPUT_SIZE; j++) {
//                     h_batch_images[stream_idx][b * INPUT_SIZE + j] = images[img_idx][j];
//                 }
//             }

//             CUDA_CHECK(cudaMemcpyAsync(d_batch_images[stream_idx], h_batch_images[stream_idx],
//                                        batch_size * INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));

//             for (int b = 0; b < batch_size; b++) {
//                 Real* d_input = d_batch_images[stream_idx] + b * INPUT_SIZE;
//                 Real* d_batch_output = d_batch_outputs[stream_idx] + b * OUTPUT_SIZE;

//                 forward_gpu(net, d_input, d_hidden, d_output, stream, config);

//                 CUDA_CHECK(cudaMemcpyAsync(d_batch_output, d_output, OUTPUT_SIZE * sizeof(Real),
//                                            cudaMemcpyDeviceToDevice, stream));
//             }

//             CUDA_CHECK(cudaMemcpyAsync(h_batch_outputs[stream_idx], d_batch_outputs[stream_idx],
//                                        batch_size * OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost, stream));

//             CUDA_CHECK(cudaStreamSynchronize(stream));

//             for (int b = 0; b < batch_size; b++) {
//                 int img_idx = i + b;
//                 int pred = 0, actual = 0;
//                 for (int j = 0; j < OUTPUT_SIZE; j++) {
//                     if (h_batch_outputs[stream_idx][b * OUTPUT_SIZE + j] > h_batch_outputs[stream_idx][b * OUTPUT_SIZE + pred]) pred = j;
//                     if (labels[img_idx][j] > labels[img_idx][actual]) actual = j;
//                 }
//                 if (pred == actual) correct++;
//             }
//         }

//         for (int s = 0; s < config->num_streams; s++) {
//             CUDA_CHECK(cudaFreeHost(h_batch_images[s]));
//             CUDA_CHECK(cudaFreeHost(h_batch_outputs[s]));
//             CUDA_CHECK(cudaFree(d_batch_images[s]));
//             CUDA_CHECK(cudaFree(d_batch_outputs[s]));
//             CUDA_CHECK(cudaStreamDestroy(streams[s]));
//         }
//         CUDA_CHECK(cudaFree(d_hidden));
//         CUDA_CHECK(cudaFree(d_output));
//     } else {
//         for (int i = 0; i < numImages; i++) {
//             Real hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
//             forward_cpu(net, images[i], hidden, output);
//             int pred = 0, actual = 0;
//             for (int j = 0; j < OUTPUT_SIZE; j++) {
//                 if (output[j] > output[pred]) pred = j;
//                 if (labels[i][j] > labels[i][actual]) actual = j;
//             }
//             if (pred == actual) correct++;
//         }
//     }
//     *test_acc_out = (correct / (double)numImages) * 100;
//     printf("%s Test Accuracy: %.2f%%\n", use_gpu ? "GPU" : "CPU", *test_acc_out);
// }

// Real** loadMNISTImages(const char* filename, int numImages) {
//     FILE* file = fopen(filename, "rb");
//     if (!file) {
//         printf("Error opening %s\n", filename);
//         exit(1);
//     }
//     fseek(file, 16, SEEK_SET);
//     Real** images = allocateMatrix(numImages, INPUT_SIZE);
//     for (int i = 0; i < numImages; i++) {
//         for (int j = 0; j < INPUT_SIZE; j++) {
//             unsigned char pixel;
//             if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
//                 fprintf(stderr, "Error: Failed to read pixel\n");
//                 fclose(file);
//                 exit(EXIT_FAILURE);
//             }
//             images[i][j] = pixel / 255.0f;
//         }
//     }
//     fclose(file);
//     return images;
// }

// Real** loadMNISTLabels(const char* filename, int numLabels) {
//     FILE* file = fopen(filename, "rb");
//     if (!file) {
//         printf("Error opening %s\n", filename);
//         exit(1);
//     }
//     fseek(file, 8, SEEK_SET);
//     Real** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
//     for (int i = 0; i < numLabels; i++) {
//         unsigned char label;
//         if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
//             fprintf(stderr, "Error: Failed to read label\n");
//             fclose(file);
//             exit(EXIT_FAILURE);
//         }
//         for (int j = 0; j < OUTPUT_SIZE; j++) {
//             labels[i][j] = (j == label) ? 1.0f : 0.0f;
//         }
//     }
//     fclose(file);
//     return labels;
// }

// void freeNetwork(NeuralNetwork* net) {
//     freeMatrix(net->W1, HIDDEN_SIZE);
//     freeMatrix(net->W2, OUTPUT_SIZE);
//     free(net->b1);
//     free(net->b2);
//     CUDA_CHECK(cudaFree(net->d_W1));
//     CUDA_CHECK(cudaFree(net->d_W2));
//     CUDA_CHECK(cudaFree(net->d_b1));
//     CUDA_CHECK(cudaFree(net->d_b2));
//     free(net);
// }

// int main() {
//     printf("MNIST Neural Network - Dynamic Launch and Variable Precision\n\n");

//     // Print applied optimizations
//     printf("=== Applied Optimizations ===\n");
//     printf("1. Dynamic Launch Configuration: Block and grid dimensions are computed dynamically using configurable block sizes (1D: %d threads, 2D: %dx%d) to optimize kernel launches for different input sizes and GPU architectures.\n",
//            config.block_size_1d, config.block_size_2d_x, config.block_size_2d_y);
//     printf("2. Variable Precision: Supports configurable precision via DATA_TYPE macro, allowing use of double, float to balance performance and accuracy.\n");
//     printf("3. Pinned Host Memory: Used cudaMallocHost for host-side allocations (temp_W1, temp_W2) to enable faster device to host memory transfers.\n");
//     printf("4. Reused Allocated Memory: Allocated d_hidden, d_output, d_d_output, and d_d_hidden once at the start of training and evaluation, reusing them across iterations and epochs, and freeing them only when complete.\n");
//     printf("5. Asynchronous Transfers with Variable Streams: Used %d CUDA streams to overlap data transfers (batch images, labels, and outputs) with computation, processing one batch while transferring another.\n", config.num_streams);
//     printf("\n");

//     Real** train_images = loadMNISTImages("data/train-images-idx3-ubyte/train-images-idx3-ubyte", 60000);
//     Real** train_labels = loadMNISTLabels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte", 60000);
//     Real** test_images = loadMNISTImages("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 10000);
//     Real** test_labels = loadMNISTLabels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", 10000);

//     NeuralNetwork* net_cpu = createNetwork();
//     double cpu_total_time, cpu_loss, cpu_train_acc, cpu_test_acc;
//     printf("Running CPU implementation...\n");
//     train(net_cpu, train_images, train_labels, 60000, false, &cpu_total_time, &cpu_loss, &cpu_train_acc, &config);
//     evaluate(net_cpu, test_images, test_labels, 10000, false, &cpu_test_acc, &config);

//     NeuralNetwork* net_gpu = createNetwork();
//     double gpu_total_time, gpu_loss, gpu_train_acc, gpu_test_acc;
//     printf("\nRunning GPU implementation...\n");
//     train(net_gpu, train_images, train_labels, 60000, true, &gpu_total_time, &gpu_loss, &gpu_train_acc, &config);
//     evaluate(net_gpu, test_images, test_labels, 10000, true, &gpu_test_acc, &config);

//     printf("\n=== Result Comparison ===\n");
//     printf("CPU Total Time: %.3fs\n", cpu_total_time);
//     printf("GPU Total Time: %.3fs\n", gpu_total_time);
//     printf("Speedup: %.2fx\n", cpu_total_time / gpu_total_time);
//     printf("CPU Loss: %.4f\n", cpu_loss);
//     printf("GPU Loss: %.4f\n", gpu_loss);
//     printf("Loss Difference: %.6f\n", fabs(cpu_loss - gpu_loss));
//     printf("CPU Train Accuracy: %.2f%%\n", cpu_train_acc);
//     printf("GPU Train Accuracy: %.2f%%\n", gpu_train_acc);
//     printf("Train Accuracy Difference: %.2f%%\n", fabs(cpu_train_acc - gpu_train_acc));
//     printf("CPU Test Accuracy: %.2f%%\n", cpu_test_acc);
//     printf("GPU Test Accuracy: %.2f%%\n", gpu_test_acc);
//     printf("Test Accuracy Difference: %.2f%%\n", fabs(cpu_test_acc - gpu_test_acc));

//     freeNetwork(net_cpu);
//     freeNetwork(net_gpu);
//     freeMatrix(train_images, 60000);
//     freeMatrix(train_labels, 60000);
//     freeMatrix(test_images, 10000);
//     freeMatrix(test_labels, 10000);

//     return 0;
// }

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

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Define precision
#define DATA_TYPE float
typedef DATA_TYPE Real;

// Configuration struct for launch parameters
typedef struct {
    int block_size_1d;     // Threads per block for 1D kernels
    int block_size_2d_x;   // Block dimension X for 2D kernels
    int block_size_2d_y;   // Block dimension Y for 2D kernels
    int num_streams;       // Number of CUDA streams
} Config;

// Default configuration
Config config = {
    .block_size_1d = 256,
    .block_size_2d_x = 16,
    .block_size_2d_y = 16,
    .num_streams = 3
};

double get_cpu_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

float get_gpu_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds / 1000.0;
}

Real** allocateMatrix(int rows, int cols) {
    Real** mat = (Real**)malloc(rows * sizeof(Real*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (Real*)malloc(cols * sizeof(Real));
    }
    return mat;
}

void freeMatrix(Real** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void relu_cpu(Real* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax_cpu(Real* x, int size) {
    Real max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    Real sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Compute grid and block dimensions for 2D kernels
void computeGridBlockDim(int rows, int cols, dim3* gridDim, dim3* blockDim, Config* config) {
    blockDim->x = config->block_size_2d_x;
    blockDim->y = config->block_size_2d_y;
    gridDim->x = (cols + config->block_size_2d_x - 1) / config->block_size_2d_x;
    gridDim->y = (rows + config->block_size_2d_y - 1) / config->block_size_2d_y;
}

// Compute blocks and threads for 1D kernels
void compute1DGridBlockDim(int size, int* blocks, int* threads, Config* config) {
    *threads = config->block_size_1d;
    *blocks = (size + config->block_size_1d - 1) / config->block_size_1d;
}

__global__ void matrixMulKernel(Real* A, Real* B, Real* C, Real* bias, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        Real sum = bias[row];
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void reluKernel(Real* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void softmaxKernel(Real* x, int size) {
    extern __shared__ Real shared[];
    Real* max_vals = shared;
    Real* sum_vals = &shared[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Step 1: Find max value using parallel reduction
    Real local_max = idx < size ? x[idx] : -INFINITY;
    max_vals[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx < size) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }

    Real global_max = max_vals[0];

    // Step 2: Compute exponentials and store in x
    if (idx < size) {
        x[idx] = expf(x[idx] - global_max);
    }
    sum_vals[tid] = idx < size ? x[idx] : 0;
    __syncthreads();

    // Step 3: Sum exponentials using parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx < size) {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }

    Real global_sum = sum_vals[0];

    // Step 4: Normalize
    if (idx < size) {
        x[idx] /= global_sum;
    }
}

__global__ void outputGradientKernel(Real* output, Real* target, Real* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void hiddenGradientKernel(Real* W2, Real* d_output, Real* hidden, Real* d_hidden, int hidden_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        Real sum = 0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + i] * d_output[j];
        }
        d_hidden[i] = sum * (hidden[i] > 0);
    }
}

__global__ void updateWeightsKernel(Real* W, Real* grad, Real* input, int rows, int cols, Real lr) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        W[i * cols + j] -= lr * grad[i] * input[j];
    }
}

__global__ void updateBiasKernel(Real* b, Real* grad, int size, Real lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        b[i] -= lr * grad[i];
    }
}

typedef struct {
    Real** W1;
    Real** W2;
    Real* b1;
    Real* b2;
    Real *d_W1, *d_W2, *d_b1, *d_b2;
} NeuralNetwork;

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (Real*)calloc(HIDDEN_SIZE, sizeof(Real));
    net->b2 = (Real*)calloc(OUTPUT_SIZE, sizeof(Real));

    srand(42);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((Real)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((Real)rand() / RAND_MAX) * 0.01;

    CUDA_CHECK(cudaMalloc(&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, HIDDEN_SIZE * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, OUTPUT_SIZE * sizeof(Real)));

    Real* temp_W1;
    Real* temp_W2;
    CUDA_CHECK(cudaMallocHost(&temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            temp_W1[i * INPUT_SIZE + j] = net->W1[i][j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            temp_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

    CUDA_CHECK(cudaMemcpy(net->d_W1, temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_W2, temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFreeHost(temp_W1));
    CUDA_CHECK(cudaFreeHost(temp_W2));

    return net;
}

void forward_cpu(NeuralNetwork* net, Real* input, Real* hidden, Real* output) {
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

void forward_gpu(NeuralNetwork* net, Real* d_input, Real* d_hidden, Real* d_output, cudaStream_t stream, Config* config) {
    dim3 blockDim, gridDim;
    computeGridBlockDim(HIDDEN_SIZE, 1, &gridDim, &blockDim, config);
    matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W1, d_input, d_hidden, net->d_b1, HIDDEN_SIZE, INPUT_SIZE, 1);

    int blocks, threads;
    compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
    reluKernel<<<blocks, threads, 0, stream>>>(d_hidden, HIDDEN_SIZE);

    computeGridBlockDim(OUTPUT_SIZE, 1, &gridDim, &blockDim, config);
    matrixMulKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W2, d_hidden, d_output, net->d_b2, OUTPUT_SIZE, HIDDEN_SIZE, 1);

    compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
    softmaxKernel<<<blocks, threads, 2 * threads * sizeof(Real), stream>>>(d_output, OUTPUT_SIZE);
}

void backward_cpu(NeuralNetwork* net, Real* input, Real* hidden, Real* output, Real* target) {
    Real d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

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

void backward_gpu(NeuralNetwork* net, Real* d_input, Real* d_hidden, Real* d_output, Real* d_target, Real* d_d_output, Real* d_d_hidden, cudaStream_t stream, Config* config) {
    int blocks, threads;
    compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
    outputGradientKernel<<<blocks, threads, 0, stream>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);

    compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
    hiddenGradientKernel<<<blocks, threads, 0, stream>>>(net->d_W2, d_d_output, d_hidden, d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE);

    dim3 blockDim, gridDim;
    computeGridBlockDim(OUTPUT_SIZE, HIDDEN_SIZE, &gridDim, &blockDim, config);
    updateWeightsKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W2, d_d_output, d_hidden, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);

    computeGridBlockDim(HIDDEN_SIZE, INPUT_SIZE, &gridDim, &blockDim, config);
    updateWeightsKernel<<<gridDim, blockDim, 0, stream>>>(net->d_W1, d_d_hidden, d_input, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);

    compute1DGridBlockDim(OUTPUT_SIZE, &blocks, &threads, config);
    updateBiasKernel<<<blocks, threads, 0, stream>>>(net->d_b2, d_d_output, OUTPUT_SIZE, LEARNING_RATE);

    compute1DGridBlockDim(HIDDEN_SIZE, &blocks, &threads, config);
    updateBiasKernel<<<blocks, threads, 0, stream>>>(net->d_b1, d_d_hidden, HIDDEN_SIZE, LEARNING_RATE);
}

void train(NeuralNetwork* net, Real** images, Real** labels, int numImages, bool use_gpu, double* total_time, double* loss_out, double* train_acc_out, Config* config) {
    double loss = 0.0;
    int correct = 0;

    if (use_gpu) {
        Real *d_hidden, *d_output, *d_d_output, *d_d_hidden;
        CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_d_output, OUTPUT_SIZE * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_d_hidden, HIDDEN_SIZE * sizeof(Real)));

        Real *d_batch_images[config->num_streams], *d_batch_labels[config->num_streams], *d_batch_outputs[config->num_streams];
        Real *h_batch_outputs[config->num_streams];
        cudaStream_t streams[config->num_streams];
        for (int s = 0; s < config->num_streams; s++) {
            CUDA_CHECK(cudaMalloc(&d_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&d_batch_labels[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&d_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMallocHost(&h_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
        }

        Real* temp_W1;
        Real* temp_W2;
        CUDA_CHECK(cudaMallocHost(&temp_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&temp_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            loss = 0.0;
            correct = 0;

            for (int i = 0; i < numImages; i += BATCH_SIZE) {
                int stream_idx = (i / BATCH_SIZE) % config->num_streams;
                cudaStream_t stream = streams[stream_idx];
                int batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : numImages - i;

                for (int b = 0; b < batch_size; b++) {
                    int img_idx = i + b;
                    CUDA_CHECK(cudaMemcpyAsync(d_batch_images[stream_idx] + b * INPUT_SIZE, images[img_idx],
                                               INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync(d_batch_labels[stream_idx] + b * OUTPUT_SIZE, labels[img_idx],
                                               OUTPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));
                }

                for (int b = 0; b < batch_size; b++) {
                    Real* d_input = d_batch_images[stream_idx] + b * INPUT_SIZE;
                    Real* d_target = d_batch_labels[stream_idx] + b * OUTPUT_SIZE;
                    Real* d_batch_output = d_batch_outputs[stream_idx] + b * OUTPUT_SIZE;

                    forward_gpu(net, d_input, d_hidden, d_output, stream, config);
                    backward_gpu(net, d_input, d_hidden, d_output, d_target, d_d_output, d_d_hidden, stream, config);

                    CUDA_CHECK(cudaMemcpyAsync(d_batch_output, d_output, OUTPUT_SIZE * sizeof(Real),
                                               cudaMemcpyDeviceToDevice, stream));
                }

                CUDA_CHECK(cudaMemcpyAsync(h_batch_outputs[stream_idx], d_batch_outputs[stream_idx],
                                           batch_size * OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost, stream));

                CUDA_CHECK(cudaStreamSynchronize(stream));

                for (int b = 0; b < batch_size; b++) {
                    int img_idx = i + b;
                    for (int k = 0; k < OUTPUT_SIZE; k++) {
                        loss -= labels[img_idx][k] * log(h_batch_outputs[stream_idx][b * OUTPUT_SIZE + k] + 1e-10);
                    }
                    int pred = 0, actual = 0;
                    for (int j = 0; j < OUTPUT_SIZE; j++) {
                        if (h_batch_outputs[stream_idx][b * OUTPUT_SIZE + j] > h_batch_outputs[stream_idx][b * OUTPUT_SIZE + pred]) pred = j;
                        if (labels[img_idx][j] > labels[img_idx][actual]) actual = j;
                    }
                    if (pred == actual) correct++;
                }
            }

            printf("GPU Epoch %d - Loss: %.4f - Train Accuracy: %.2f%%\n",
                   epoch + 1, loss / numImages, (correct / (double)numImages) * 100);
            if (epoch == EPOCHS - 1) {
                *loss_out = loss / numImages;
                *train_acc_out = (correct / (double)numImages) * 100;
            }
        }

        CUDA_CHECK(cudaMemcpy(temp_W1, net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(temp_W2, net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(net->b1, net->d_b1, HIDDEN_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(net->b2, net->d_b2, OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        *total_time = get_gpu_time(start, stop);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (int i = 0; i < HIDDEN_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                net->W1[i][j] = temp_W1[i * INPUT_SIZE + j];
        for (int i = 0; i < OUTPUT_SIZE; i++)
            for (int j = 0; j < HIDDEN_SIZE; j++)
                net->W2[i][j] = temp_W2[i * HIDDEN_SIZE + j];

        CUDA_CHECK(cudaFreeHost(temp_W1));
        CUDA_CHECK(cudaFreeHost(temp_W2));
        for (int s = 0; s < config->num_streams; s++) {
            CUDA_CHECK(cudaFreeHost(h_batch_outputs[s]));
            CUDA_CHECK(cudaFree(d_batch_images[s]));
            CUDA_CHECK(cudaFree(d_batch_labels[s]));
            CUDA_CHECK(cudaFree(d_batch_outputs[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_d_output));
        CUDA_CHECK(cudaFree(d_d_hidden));
    } else {
        clock_t total_start = clock();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            loss = 0.0;
            correct = 0;

            for (int i = 0; i < numImages; i++) {
                Real hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
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

void evaluate(NeuralNetwork* net, Real** images, Real** labels, int numImages, bool use_gpu, double* test_acc_out, Config* config) {
    int correct = 0;
    if (use_gpu) {
        Real *d_hidden, *d_output;
        CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(Real)));

        Real *h_batch_images[config->num_streams], *h_batch_outputs[config->num_streams];
        Real *d_batch_images[config->num_streams], *d_batch_outputs[config->num_streams];
        cudaStream_t streams[config->num_streams];
        for (int s = 0; s < config->num_streams; s++) {
            CUDA_CHECK(cudaMallocHost(&h_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMallocHost(&h_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&d_batch_images[s], BATCH_SIZE * INPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&d_batch_outputs[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(Real)));
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
        }

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int stream_idx = (i / BATCH_SIZE) % config->num_streams;
            cudaStream_t stream = streams[stream_idx];
            int batch_size = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : numImages - i;

            for (int b = 0; b < batch_size; b++) {
                int img_idx = i + b;
                for (int j = 0; j < INPUT_SIZE; j++) {
                    h_batch_images[stream_idx][b * INPUT_SIZE + j] = images[img_idx][j];
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(d_batch_images[stream_idx], h_batch_images[stream_idx],
                                       batch_size * INPUT_SIZE * sizeof(Real), cudaMemcpyHostToDevice, stream));

            for (int b = 0; b < batch_size; b++) {
                Real* d_input = d_batch_images[stream_idx] + b * INPUT_SIZE;
                Real* d_batch_output = d_batch_outputs[stream_idx] + b * OUTPUT_SIZE;

                forward_gpu(net, d_input, d_hidden, d_output, stream, config);

                CUDA_CHECK(cudaMemcpyAsync(d_batch_output, d_output, OUTPUT_SIZE * sizeof(Real),
                                           cudaMemcpyDeviceToDevice, stream));
            }

            CUDA_CHECK(cudaMemcpyAsync(h_batch_outputs[stream_idx], d_batch_outputs[stream_idx],
                                       batch_size * OUTPUT_SIZE * sizeof(Real), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int b = 0; b < batch_size; b++) {
                int img_idx = i + b;
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (h_batch_outputs[stream_idx][b * OUTPUT_SIZE + j] > h_batch_outputs[stream_idx][b * OUTPUT_SIZE + pred]) pred = j;
                    if (labels[img_idx][j] > labels[img_idx][actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
        }

        for (int s = 0; s < config->num_streams; s++) {
            CUDA_CHECK(cudaFreeHost(h_batch_images[s]));
            CUDA_CHECK(cudaFreeHost(h_batch_outputs[s]));
            CUDA_CHECK(cudaFree(d_batch_images[s]));
            CUDA_CHECK(cudaFree(d_batch_outputs[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
        CUDA_CHECK(cudaFree(d_hidden));
        CUDA_CHECK(cudaFree(d_output));
    } else {
        for (int i = 0; i < numImages; i++) {
            Real hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward_cpu(net, images[i], hidden, output);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
    }
    *test_acc_out = (correct / (double)numImages) * 100;
    printf("%s Test Accuracy: %.2f%%\n", use_gpu ? "GPU" : "CPU", *test_acc_out);
}

Real** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    Real** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0f;
        }
    }
    fclose(file);
    return images;
}

Real** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    Real** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    fclose(file);
    return labels;
}

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

int main() {
    printf("MNIST Neural Network - Dynamic Launch, Variable Precision, Parallel Softmax\n\n");

    // Print applied optimizations
    printf("=== Applied Optimizations ===\n");
    printf("1. Dynamic Launch Configuration: Block and grid dimensions are computed dynamically using configurable block sizes (1D: %d threads, 2D: %dx%d) to optimize kernel launches for different input sizes and GPU architectures.\n",
           config.block_size_1d, config.block_size_2d_x, config.block_size_2d_y);
    printf("2. Variable Precision: Supports configurable precision via DATA_TYPE macro, allowing use of double, float, to balance performance and accuracy.\n");
    printf("3. Parallel Softmax: Softmax kernel uses parallel reductions for max and sum operations, distributing computation across multiple threads to improve GPU utilization.\n");
    printf("4. Pinned Host Memory: Used cudaMallocHost for host-side allocations (temp_W1, temp_W2) to enable faster device to host memory transfers.\n");
    printf("5. Reused Allocated Memory: Allocated d_hidden, d_output, d_d_output, and d_d_hidden once at the start of training and evaluation, reusing them across iterations and epochs, and freeing them only when complete.\n");
    printf("6. Asynchronous Transfers with Variable Streams: Used %d CUDA streams to overlap data transfers (batch images, labels, and outputs) with computation, processing one batch while transferring another.\n", config.num_streams);
    printf("\n");


    Real** train_images = loadMNISTImages("data/train-images-idx3-ubyte/train-images-idx3-ubyte", 60000);
    Real** train_labels = loadMNISTLabels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte", 60000);
    Real** test_images = loadMNISTImages("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 10000);
    Real** test_labels = loadMNISTLabels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", 10000);

    NeuralNetwork* net_cpu = createNetwork();
    double cpu_total_time, cpu_loss, cpu_train_acc, cpu_test_acc;
    printf("Running CPU implementation...\n");
    train(net_cpu, train_images, train_labels, 60000, false, &cpu_total_time, &cpu_loss, &cpu_train_acc, &config);
    evaluate(net_cpu, test_images, test_labels, 10000, false, &cpu_test_acc, &config);

    NeuralNetwork* net_gpu = createNetwork();
    double gpu_total_time, gpu_loss, gpu_train_acc, gpu_test_acc;
    printf("\nRunning GPU implementation...\n");
    train(net_gpu, train_images, train_labels, 60000, true, &gpu_total_time, &gpu_loss, &gpu_train_acc, &config);
    evaluate(net_gpu, test_images, test_labels, 10000, true, &gpu_test_acc, &config);

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

    freeNetwork(net_cpu);
    freeNetwork(net_gpu);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);

    return 0;
}
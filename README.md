 
Neural Network Acceleration on GPUs for MNIST Classification
Overview
This project accelerates a neural network for MNIST digit classification using CUDA on GPUs, with four implementations:

V1: Sequential CPU
V2: Naive GPU with CUDA
V3: Optimized GPU with dynamic configurations and streams
V4: Advanced GPU using Tensor Cores via cuBLAS with TF32

Prerequisites

Hardware: CPU (V1), NVIDIA GPU (Ampere+ for V4)
Software: GCC (V1), CUDA Toolkit 11.x+, cuBLAS, NVCC
Dataset: MNIST files in data/ (download from http://yann.lecun.com/exdb/mnist/)
OS: Linux (Ubuntu tested)

Setup
Place MNIST dataset files in data/:

train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte

Run Instructions
V1: Sequential CPU
make clean && make src/V1
./src/nn.exe


Output: Epoch-wise loss, train accuracy, ~22.38s training time, ~96.78% test accuracy

V2: Naive GPU
nvcc -O2 -o src/n src/v2.cu
./src/n


Output: CPU/GPU metrics, ~183.16s GPU time, speedup, accuracy comparison

V3: Optimized GPU
nvcc -O2 -o src/V3/n src/v3.cu
./src/n


Output: Optimizations, CPU/GPU metrics, ~6.78s GPU time, ~3.82x speedup, ~96.20% test accuracy

V4: Tensor Core GPU
nvcc -arch=sm_80 -O2 -lcublas -o src/n src/v4.cu
./src/n


Output: Tensor Core details, CPU/GPU metrics, ~5.82s GPU time, ~4.51x speedup, ~91.93% test accuracy

Notes

Accuracy: V4 may have lower accuracy (~91.93%) due to TF32; V3 balances speed/accuracy (~96.20%)
Performance: V3 (~3.82x) and V4 (~4.51x) outperform V1; V2 is slower due to naive design
Troubleshooting: Verify CUDA/cuBLAS, dataset placement, GPU compatibility (nvidia-smi)



#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define GLOBAL_N 10
const int M = (1 << GLOBAL_N), N = (1 << GLOBAL_N), K = (1 << GLOBAL_N); /* matrix size */
#define BLOCK_SIZE (1 << 3) /* thread block size */

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row+ BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel_Shared(const Matrix A, const Matrix B, Matrix C);
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul_Shared(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    {
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    }

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    {
        size_t size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    }
    
    // Allocate C in device memory
    Matrix d_C; 
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    {
        size_t size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);
    }
    
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel_Shared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    // Read C from device memory
    {
        size_t size = C.width * C.height * sizeof(float);
        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    }

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel_Shared(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        } 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A = {.width = A.width, .height = A.height};
    {
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);   
    }

    Matrix d_B = {.width = B.width, .height = B.height};
    {
        size_t size = B.width * B.height * sizeof(float);
        cudaMalloc(&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);   
    }

    Matrix d_C = {.width = C.width, .height = C.height};
    {
        size_t size = C.width * C.height * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        MatMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_A.elements);    
    cudaFree(d_B.elements);    
    cudaFree(d_C.elements);    
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.width; ++e) {
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}

void MatMul_CPU(const Matrix A, const Matrix B, Matrix C) {
  {
      int i = 0, j = 0, k = 0;
      for (i = 0; i < C.height; i++) {
          for (j = 0; j < C.width; j++) {
              C.elements[i * C.width + j] = 0.0;
              for (k = 0; k < A.width; k++) {
                  C.elements[i * C.width + j] += A.elements[i * A.width + k] * B.elements[k * B.width + j];
              }
          }
      }
  }

  return;
}

void MatMul_multiCPU(const Matrix A, const Matrix B, Matrix C) {
    {
        int i = 0, j = 0, k = 0;
    #pragma omp parallel for private(i, j, k)
        for (i = 0; i < C.height; i++) {
            for (j = 0; j < C.width; j++) {
                C.elements[i * C.width + j] = 0.0;
                for (k = 0; k < A.width; k++) {
                    C.elements[i * C.width + j] += A.elements[i * A.width + k] * B.elements[k * B.width + j];
                }
            }
        }
    }
    
    return;
}

void check_mat(const Matrix C, const Matrix CC) {
    int flg = 0;

    {
        int i = 0, j = 0;
        for (i = 0; i < C.height; i++) {
            for (j = 0; j < C.width; j++) {
                if (fabs(C.elements[C.width * i + C.height] - CC.elements[C.width * i + C.height]) > 1e-10) {
                    flg = 1;
                }
            }
        }
    }

    printf(flg == 1 ? "Calculation error.\n" : "OK.\n");
    
    return;
}
// const int M = 10, N = 10, K = 10, BS = 2;

int main(int argc,char *argv[]){
    Matrix A = {.width = K, .height = M, .stride = K, .elements = (float *) malloc((K * M) * sizeof(float))};
    Matrix B = {.width = N, .height = K, .stride = N, .elements = (float *) malloc((N * K) * sizeof(float))};
    Matrix C = {.width = N, .height = M, .stride = N, .elements = (float *) malloc((N * M) * sizeof(float))};
    Matrix C2 = {.width = N, .height = M, .stride = N, .elements = (float *) malloc((N * M) * sizeof(float))};
    Matrix C3 = {.width = N, .height = M, .stride = N, .elements = (float *) malloc((N * M) * sizeof(float))};

    // float *pA = NULL, *pB = NULL, *pC = NULL;
    // __device__ Matrix dA = {.width = K, .height = M, .stride = K, .elements = (float *) malloc((K * M) * sizeof(float))};
    // __device__ Matrix dB = {.width = K, .height = M, .stride = K, .elements = (float *) malloc((K * M) * sizeof(float))};
    // __device__ Matrix dC = {.width = K, .height = M, .stride = K, .elements = (float *) malloc((K * M) * sizeof(float))};

    // Matrix CC = {.width = N, .height = M, .stride = N, .elements = (float *) malloc((N * M) * sizeof(float))};

    // set initial value
    {
        srand(248309);

        int i = 0;
        for (i = 0; i < (K * M); i++) {
            A.elements[i] = ((float) rand()) / ((float) RAND_MAX);
            // printf("A[%d] = %3.1f\n", i, A.elements[i]);
        }

        for (i = 0; i < (N * K); i++) {
            B.elements[i] = ((float) rand()) / ((float) RAND_MAX);
            // printf("B[%d] = %3.1f\n", i, B.elements[i]);
        }
    }

    // GPU shared-matmul
    {
        struct timeval stime;
        gettimeofday(&stime, NULL);

        MatMul_Shared(A, B, C);

        struct timeval etime;
        gettimeofday(&etime, NULL);  

        float nettime = (etime.tv_sec - stime.tv_sec) + (etime.tv_usec - stime.tv_usec) * 1.0e-6;
        printf("Elapsed time[s] for GPU shared-matmul: %f\n", nettime);
    }

    // GPU matmul
    {
        struct timeval stime;
        gettimeofday(&stime, NULL);

        MatMul(A, B, C2);

        struct timeval etime;
        gettimeofday(&etime, NULL);  

        float nettime = (etime.tv_sec - stime.tv_sec) + (etime.tv_usec - stime.tv_usec) * 1.0e-6;
        printf("Elapsed time[s] for GPU matmul: %f\n", nettime);
    }

    // // CPU matmul
    // {
    //     struct timeval stime;
    //     gettimeofday(&stime, NULL);

    //     MatMul_CPU(A, B, C3);

    //     struct timeval etime;
    //     gettimeofday(&etime, NULL);  

    //     float nettime = (etime.tv_sec - stime.tv_sec) + (etime.tv_usec - stime.tv_usec) * 1.0e-6;
    //     printf("Elapsed time[s] for CPU matmul: %f\n", nettime);
    // }

    // check_mat(C, C2);
    // check_mat(C, C3);

    return 0;
}
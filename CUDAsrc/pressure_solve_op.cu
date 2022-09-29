

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <numeric>

using namespace std;

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cdpErrchk_blas(ans) { cdpAssert_blas((ans), __FILE__, __LINE__); }
inline void cdpAssert_blas(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %s %s %d\n", code, file, line);
      if (abort) assert(0);
   }
}

__global__ void calcZ_v4(const int *dimensions, const int dim_product, const int maxDataPerRow, const float *laplace_matrix,
 const float *p, float *z, const bool* boolPeriodic, const int* diagonalOffsets, bool* domainBoundaryBool, const bool* laplace_rank_deficient) {
   // all shared memory needs to be in one 1D array --> append periodic Offsets to the diagonalOffsets
    /*extern __shared__ int diagonalOffsets[];
    //extern __shared__ int periodicOffsets[];

    // Build diagonalOffsets on the first thread of each block and write it to shared memory
    if(threadIdx.x == 0) {
        const int diagonal = maxDataPerRow / 2;
        diagonalOffsets[diagonal] = 0;
        diagonalOffsets[maxDataPerRow + diagonal] = 0;
        int factor = 1;

        for(int i = 0, offset = 1; i < diagonal; i++, offset++) {
            diagonalOffsets[diagonal - offset] = -factor;
            diagonalOffsets[diagonal + offset] = factor;
            factor *= dimensions[i];
            // periodic Offsets appended to diagonal
            diagonalOffsets[maxDataPerRow + diagonal - offset] = factor  * int(boolPeriodic[i]);
            diagonalOffsets[maxDataPerRow + diagonal + offset] = -factor * int(boolPeriodic[i]);
        }
    }
    for (size_t i = 0; i < maxDataPerRow*2; i++) {
      printf("%d  ",diagonalOffsets[i] );
    }
    printf("\n");
    __syncthreads();*/

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim_product-1 || (row < dim_product && laplace_rank_deficient[0]==false)){
    //----------------------------------------------------------------------
    // COULD THIS FOR LOOP BE THE PROBLEM!!!
    //----------------------------------------------------------------------
    //for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x){
        const int diagonal = row * maxDataPerRow;
        int helper = 0;
        int dimSize = maxDataPerRow/2;
        int modulo = 0;
        int divisor = dim_product;

        domainBoundaryBool[maxDataPerRow*row + dimSize] = 0;

        for(int i = dimSize - 1; i >= 0; i--) {
            divisor = divisor / dimensions[i];
            helper = (modulo == 0 ? row : (row % modulo)) / divisor;
            domainBoundaryBool[maxDataPerRow*row + dimSize - (i+1)] = 1-min(helper,1);
            domainBoundaryBool[maxDataPerRow*row + dimSize + (i+1)] = max(helper-dimensions[i]+2,0);
            modulo = divisor;
        }

        float tmp = 0;
        int current_index;
        /*printf("row %d   ",row );
        for (size_t i = 0; i < maxDataPerRow; i++) {
          printf("%d ", domainBoundaryBool[i]);
        }
        printf("    ");*/
        for(int i = diagonal; i < diagonal + maxDataPerRow; i++) {
            // when accessing out of bound memory in p, laplace_matrix[i] is always zero. So no illegal mem-access will be made.
            // If this causes problems add :
            // if(row + offsets[i - diagonalOffsets] >= 0 && row + offsets[i - diagonalOffsets] < dim_product)
            current_index = row + diagonalOffsets[i - diagonal] +
            domainBoundaryBool[maxDataPerRow*row + i - diagonal] * diagonalOffsets[i - diagonal + maxDataPerRow];
            /*if(current_index<0 || current_index>=dim_product){
              printf("row %d   index %d\n", row, current_index);
            }
            if(i > dim_product*maxDataPerRow){ printf("row  %d  i %d\n",row, i );}*/
            tmp += laplace_matrix[i] * p[current_index];//p[row + diagonalOffsets[i - diagonal] + domainBoundaryBool[i - diagonal] * diagonalOffsets[i - diagonal + maxDataPerRow]];
            //p[current_index]; // No modulo here (as the general way in the thesis suggests)
            //printf("%d  ", current_index );
        }
        //printf(" result %f\n", tmp );
        z[row] = tmp;
    }
    else if (row == dim_product-1){
        float tmp = 0;
        /*for (size_t i = 0; i < dim_product; i++) {
          tmp += p[i];
        }*/
        //tmp = p[row];
        z[row]=1e8*p[row]; //tmp;
    }
}

__global__ void checkResiduum(const int dim_product, const float* r, const float threshold, bool *threshold_reached) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x) {
        if (r[row] >= threshold) {
          *threshold_reached = false;
          break;
        }
    }
}

__global__ void initVariablesWithGuess(const int dim_product, const float *divergence, float* A_times_x_0, float *p, float *r, bool *threshold_reached) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim_product) {
        float tmp = divergence[row] - A_times_x_0[row];
        p[row] = tmp;
        r[row] = tmp;

    }
    if(row == 0) *threshold_reached = false;
}


__global__ void printarray(float* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %f ", a[i]);
  }
  printf("\n");
}

__global__ void calcDiagonalOffsets(int* diagonalOffsets, const int maxDataPerRow, const int* dimensions,
  const bool* boolPeriodic){

    const int diagonal = maxDataPerRow / 2;
    diagonalOffsets[diagonal] = 0;
    diagonalOffsets[maxDataPerRow + diagonal] = 0;
    int factor = 1;

    for(int i = 0, offset = 1; i < diagonal; i++, offset++) {
        diagonalOffsets[diagonal - offset] = -factor;
        diagonalOffsets[diagonal + offset] = factor;
        factor *= dimensions[i];
        // periodic Offsets appended to diagonal
        diagonalOffsets[maxDataPerRow + diagonal - offset] = factor  * int(boolPeriodic[i]);
        diagonalOffsets[maxDataPerRow + diagonal + offset] = -factor * int(boolPeriodic[i]);
  }
}

// global blas handle, initialize only once (warning - currently not free'd!)
bool           initBlasHandle = true;
cublasHandle_t blasHandle;

void LaunchPressureKernel(const int* dimensions, const int dim_product, const int dim_size,
                          const float *laplace_matrix,
                          float* p, float* z, float* r, float* divergence, float* x,
                          const float *oneVector,
                          bool* threshold_reached,
                          const float* accuracy_ptr,
                          const int* max_iterations_ptr,
                          const int batch_size,
                          int* iterations_gpu,
                          const bool* boolPeriodic,
                          const bool* laplace_rank_deficient)
{
    float accuracy = 0.;
    int max_iterations = 0;
    cudaMemcpy(&accuracy, accuracy_ptr, sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_iterations, max_iterations_ptr, sizeof(int),cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    if(initBlasHandle) {
        cdpErrchk_blas(cublasCreate_v2(&blasHandle));
        cublasSetPointerMode_v2(blasHandle, CUBLAS_POINTER_MODE_HOST);
        initBlasHandle = false;
    }

    // CG helper variables variables init
    float *alpha = new float[batch_size], *beta = new float[batch_size];
    const float oneScalar = 1.0f;
    bool *threshold_reached_cpu = new bool[batch_size];
    float *p_r = new float[batch_size], *p_z = new float[batch_size], *r_z = new float[batch_size];

    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;

    int* diagonalOffsets;
    cudaMalloc(&diagonalOffsets, (dim_size*2+1)*sizeof(int));
    calcDiagonalOffsets<<<1,1>>>(diagonalOffsets, dim_size*2+1, dimensions, boolPeriodic);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    bool* domainBoundaryBool;
    cudaMalloc(&domainBoundaryBool, (dim_size*2+1)*dim_product*sizeof(bool));
    // Initialize the helper variables
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // First calc A * x_0, save result to z:
    for(int i = 0; i < batch_size; i++) {
        calcZ_v4<<<gridSize, blockSize, 2*(dim_size * 2 + 1)>>>(dimensions,
        //calcZ_v4<<<1, 1, 2*(dim_size * 2 + 1)>>>(dimensions,
                                                            dim_product,
                                                            dim_size * 2 + 1,
                                                            laplace_matrix,
                                                            x + i * dim_product,
                                                            z + i * dim_product,
                                                            boolPeriodic,
                                                            diagonalOffsets,
                                                            domainBoundaryBool,
                                                            laplace_rank_deficient);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Second apply result to the helper variables
    for(int i = 0; i < batch_size; i++) {
        int offset = i * dim_product;
        initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product,
                                                        divergence + offset,
                                                        z + offset,
                                                        p + offset,
                                                        r + offset,
                                                        threshold_reached + i);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    // Init residuum checker variables
    CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcZ_v4, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Do CG-Solve
    int checker = 1;
    int iterations = 0;
    for (; iterations < max_iterations; iterations++) {
        //printf("iteration %d\n", iterations);
        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            calcZ_v4<<<gridSize, blockSize, 2*(dim_size * 2 + 1)>>>
            //calcZ_v4<<<1,1, 2*(dim_size * 2 + 1)>>>
            (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix,
               p + i * dim_product, z + i * dim_product,
               boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            cublasSdot_v2(blasHandle, dim_product, p + i * dim_product, 1, r + i * dim_product, 1, p_r + i);
            cublasSdot_v2(blasHandle, dim_product, p + i * dim_product, 1, z + i * dim_product, 1, p_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            alpha[i] = 0.;
            if(fabs(p_z[i])>0.) alpha[i] = p_r[i] / p_z[i];
            cublasSaxpy_v2(blasHandle, dim_product, alpha + i, p + i * dim_product, 1, x + i * dim_product, 1);

            alpha[i] = -alpha[i];
            cublasSaxpy_v2(blasHandle, dim_product, alpha + i, z + i * dim_product, 1, r + i * dim_product, 1);

        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Check the residuum every 5 steps to keep memcopys between H&D low
        // Tests have shown, that 5 is a good avg trade-of between memcopys and extra computation and increases the performance
        if (checker % 5 == 0) {
            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                // Use fewer occupancy here, because in most cases residual will be to high and therefore
                checkResiduum<<<8, blockSize>>>(dim_product, r + i * dim_product, accuracy, threshold_reached + i);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            bool done = true;
            for(int i = 0; i < batch_size; i++) {
                if (!threshold_reached_cpu[i]) {
                    done = false;
                    break;
                }
            }
            if(done){
                iterations++;
                break;
            }
            CUDA_CHECK_RETURN(cudaMemset(threshold_reached, 1, sizeof(bool) * batch_size));
        }
        checker++;

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            cublasSdot_v2(blasHandle, dim_product, r + i * dim_product, 1, z + i * dim_product, 1, r_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            beta[i] = -r_z[i] / p_z[i];
            cublasSscal_v2(blasHandle, dim_product, beta + i, p + i * dim_product, 1);
            cublasSaxpy_v2(blasHandle, dim_product, &oneScalar, r + i * dim_product, 1, p + i * dim_product, 1);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    delete[] alpha, beta, threshold_reached_cpu, p_r, p_z, r_z;
//    printf("I: %i\n", iterations);

    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaFree(diagonalOffsets);
    cudaFree(domainBoundaryBool);
    //printf("end\n");
    //printarray<<<1,1>>>(x, 5);
    //CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}

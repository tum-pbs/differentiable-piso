

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include "cuda_helpers.h"

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

#define cdpErrchk_rand(ans) { cdpAssert_rand((ans), __FILE__, __LINE__); }
inline void cdpAssert_rand(curandStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %s %s %d\n", code, file, line);
      if (abort) assert(0);
   }
}

/*#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)*/

template <typename T>
__global__ void calcZ_v4(const int *dimensions, const int dim_product, const int maxDataPerRow, const T *laplace_matrix,
 const T *p, T *z, const bool* boolPeriodic, const int* diagonalOffsets, bool* domainBoundaryBool, const bool* laplace_rank_deficient, const T vectorSum) {

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product;
        row += blockDim.x * gridDim.x){
        const int diagonal = row * maxDataPerRow;
        int helper = 0;
        int dimSize = maxDataPerRow/2;
        int modulo = 0;
        int divisor = dim_product;

        domainBoundaryBool[maxDataPerRow*row + dimSize] = 0;

        //printf("\n row %d    accessing dBB: %d - %d  and lap: \n", row, maxDataPerRow*row,maxDataPerRow*row+dimSize);
        for(int i = dimSize - 1; i >= 0; i--) {
            divisor = divisor / dimensions[i];
            helper = (modulo == 0 ? row : (row % modulo)) / divisor;
            domainBoundaryBool[maxDataPerRow*row + dimSize - (i+1)] = 1-min(helper,1);
            domainBoundaryBool[maxDataPerRow*row + dimSize + (i+1)] = max(helper-dimensions[i]+2,0);
            modulo = divisor;
        }

        T tmp = 0;
        int current_index;
        for(int i = diagonal; i < diagonal + maxDataPerRow; i++) {
            current_index = row + diagonalOffsets[i - diagonal] +
            domainBoundaryBool[maxDataPerRow*row + i - diagonal] * diagonalOffsets[i - diagonal + maxDataPerRow];
            /*printf("sum of %d + %d + %d * %d  =  %d    lap[i]: %f \n",row,  diagonalOffsets[i - diagonal],
                  domainBoundaryBool[maxDataPerRow*row + i - diagonal] ,
                  diagonalOffsets[i - diagonal + maxDataPerRow],current_index, laplace_matrix[i]);*/
            tmp += laplace_matrix[i] * p[current_index*(laplace_matrix[i]!=0.0)];
        }
        z[row] = tmp + vectorSum;
    }
}

template <typename T>
__global__ void checkResiduum(const int dim_product, const T* r, const float threshold, bool *threshold_reached) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x) {
        if (abs(r[row]) >= threshold) {
          *threshold_reached = false;
          break;
        }
    }
}

template <typename T>
__global__ void initVariablesWithGuess(const int dim_product, const T *divergence, T* A_times_x_0, T *p, T *r, bool *threshold_reached) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim_product) {
        T tmp = divergence[row] - A_times_x_0[row];
        p[row] = tmp;
        r[row] = tmp;

    }
    if(row == 0) *threshold_reached = false;
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

void LaunchPressureKernel(const int* dimensions, const int dim_product, const int dim_size, const double *laplace_matrix,
                          double* p, double* z, double* r, double* divergence, double* x, bool* threshold_reached,
                          const float* accuracy_ptr, const int* max_iterations_ptr, const int batch_size,
                          int* iterations_gpu, const bool* boolPeriodic, const bool* laplace_rank_deficient, const bool init_with_zeros,
                          const int residual_reset_steps, const int randomized_restarts, const int unrolling_step)
{
    double minus_half = -.5f;
    float accuracy = 0.;
    int max_iterations = 0;
    bool h_laplace_rank_deficient = false;
    cudaMemcpy(&accuracy, accuracy_ptr, sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_iterations, max_iterations_ptr, sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_laplace_rank_deficient, laplace_rank_deficient, sizeof(bool),cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    if(initBlasHandle) {
        cdpErrchk_blas(cublasCreate_v2(&blasHandle));
        cublasSetPointerMode_v2(blasHandle, CUBLAS_POINTER_MODE_HOST);
        initBlasHandle = false;
    }

    // if laplace matrix is rank < n -> shift singularity by solving (A+e^Te)x = b instead
    // shift "magnitude" scaled by divervence Asum
    double vectorSum[batch_size]  = {0};
    double vectorSum_scaling[batch_size] = {0};
    for (size_t i = 0; i < batch_size && h_laplace_rank_deficient==true; i++) {
      cublasDasum(blasHandle, dim_product, laplace_matrix + dim_size, 2*dim_size+1, &vectorSum_scaling[i]);
      vectorSum_scaling[i] *= .1/dim_product;
      //printf("vector scaling %f\n", vectorSum_scaling[i]);
    }
    double* d_vectorSum_scaling;
    cudaMalloc(&d_vectorSum_scaling, batch_size*sizeof(double));
    cudaMemcpy(d_vectorSum_scaling, vectorSum_scaling, batch_size*sizeof(double), cudaMemcpyHostToDevice);


    // CG helper variables variables init
    double *alpha = new double[batch_size], *beta = new double[batch_size];
    double oneScalar = 1.0f;
    bool *threshold_reached_cpu = new bool[batch_size];
    double *p_r = new double[batch_size], *p_z = new double[batch_size], *r_z = new double[batch_size];

    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;

    if (init_with_zeros){
      cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initWithZeros<double>, 0, 0);
      gridSize = (dim_product*batch_size + blockSize - 1) / blockSize;
      initWithZeros<<<gridSize, blockSize>>>(x, batch_size*dim_product);
    }
    int* diagonalOffsets;
    cudaMalloc(&diagonalOffsets, 2*(dim_size*2+1)*sizeof(int));
    calcDiagonalOffsets<<<1,1>>>(diagonalOffsets, dim_size*2+1, dimensions, boolPeriodic);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    //printarray_int<<<1,1>>>(diagonalOffsets, 2*(dim_size*2+1));

    bool* domainBoundaryBool;
    cudaMalloc(&domainBoundaryBool, (dim_size*2+1)*dim_product*sizeof(bool));
    // Initialize the helper variables
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4<double>, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // First calc A * x_0, save result to z:
    for(int i = 0; i < batch_size; i++) {
        if(h_laplace_rank_deficient){
          cublasDdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
        }
        //printf("sum %f\n", vectorSum[i]);
        //calcZ_v4<<<1, 1, 2*(dim_size * 2 + 1)>>>
        calcZ_v4<<<gridSize, blockSize>>>
        //calcZ_v4<<<1,1>>>
        (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
          boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    bool restart = true;
    double max_residual;
    int max_residual_index;
    curandGenerator_t random_generator;
    cdpErrchk_rand(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    int checker = 1;
    int iterations = 0;
    size_t restart_num = 0;

    double rinitsum = 0;
    double xinitsum = 0;
    for (; restart_num <= randomized_restarts && restart==true ; restart_num++) {
        // Second apply result to the helper variables
        // printf("restart number %d\n", restart_num);
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess<double>, 0, 0);
        gridSize = (dim_product + blockSize - 1) / blockSize;
        for(int i = 0; i < batch_size; i++) {
            int offset = i * dim_product;
            initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product, divergence + offset, z + offset,
                                                            p + offset, r + offset, threshold_reached + i);
        }
        // UNNECESSARY SUM FOR PRINTCHECK
        cublasDasum(blasHandle, dim_product, r, 1, &rinitsum);

        cublasDasum(blasHandle, dim_product, x, 1, &xinitsum);


        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Init residuum checker variables
        CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4<double>, 0, 0);
        gridSize = (dim_product + blockSize - 1) / blockSize;

        // Do CG-Solve

        for (; iterations < max_iterations; iterations++) {
        //if (false){
            // reset residual every residual_reset_steps steps to avoid residual divergence
            if((iterations + 1) % residual_reset_steps == 0){
              for(int i = 0; i < batch_size; i++) {
                int offset = i * dim_product;
                if(h_laplace_rank_deficient){
                  cublasDdot(blasHandle, dim_product, x + offset, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
                }
                calcZ_v4<<<gridSize, blockSize>>>
                (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + offset, z + offset,
                  boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);

                initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product, divergence + offset, z + offset,
                                                                p + offset, r + offset, threshold_reached + i);
              }
              CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            }

            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                if(h_laplace_rank_deficient){
                  cublasDdot(blasHandle, dim_product, p + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);

                }
                //printf("iter  %d    sum  %f\n", iterations, vectorSum[i]);
                //calcZ_v4<<<1,1, 2*(dim_size * 2 + 1)>>>
                calcZ_v4<<<gridSize, blockSize>>>
                (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, p + i * dim_product, z + i * dim_product,
                   boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);

            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                cublasDdot_v2(blasHandle, dim_product, p + i * dim_product, 1, r + i * dim_product, 1, p_r + i);
                cublasDdot_v2(blasHandle, dim_product, p + i * dim_product, 1, z + i * dim_product, 1, p_z + i);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                alpha[i] = 0.;
                if(fabs(p_z[i])>0.) alpha[i] = p_r[i] / p_z[i];
                cublasDaxpy_v2(blasHandle, dim_product, alpha + i, p + i * dim_product, 1, x + i * dim_product, 1);

                alpha[i] = -alpha[i];
                cublasDaxpy_v2(blasHandle, dim_product, alpha + i, z + i * dim_product, 1, r + i * dim_product, 1);

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

            // print residuum sum
            /*float rAsum = 0;
            cublasSasum(blasHandle, dim_product, r, 1, &rAsum);
            printf("iteration %d   res_Asum %f   rank def %d\n",iterations, rAsum , int(h_laplace_rank_deficient));*/


            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                cublasDdot_v2(blasHandle, dim_product, r + i * dim_product, 1, z + i * dim_product, 1, r_z + i);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                beta[i] = -r_z[i] / p_z[i];
                cublasDscal_v2(blasHandle, dim_product, beta + i, p + i * dim_product, 1);
                cublasDaxpy_v2(blasHandle, dim_product, &oneScalar, r + i * dim_product, 1, p + i * dim_product, 1);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }

        // check whether restart might be necessary - initialise with random data
        restart = false;
        for (size_t i = 0; i < batch_size; i++) {
        //if (false){ int i=0;
          if(h_laplace_rank_deficient){
            cublasDdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
          }
          //calcZ_v4<<<1, 1, 2*(dim_size * 2 + 1)>>>
          calcZ_v4<<<gridSize, blockSize>>>
          (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
            boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
          CUDA_CHECK_RETURN(cudaDeviceSynchronize());
          oneScalar = -1.f;
          cublasDaxpy(blasHandle, dim_product, &oneScalar, divergence + i * dim_product, 1, z + i * dim_product, 1);
          oneScalar = 1.f;
          cublasIdamax(blasHandle, dim_product, z + i * dim_product, 1, &max_residual_index);
          cudaMemcpy(&max_residual, &(z+i*dim_product)[max_residual_index+1], sizeof(float), cudaMemcpyDeviceToHost);
          CUDA_CHECK_RETURN(cudaDeviceSynchronize());
          if(max_residual>accuracy && restart_num<randomized_restarts){
            // initialise new guess with small random perturbation in +- vectorSum_scaling/2
            double rescaling = .01;
            cdpErrchk_rand(curandGenerateUniformDouble(random_generator, x + i * dim_product, dim_product));
            cdpErrchk_blas(cublasDscal(blasHandle, dim_product, &rescaling, x + i * dim_product, 1));
            cdpErrchk_blas(cublasDscal(blasHandle, dim_product, vectorSum_scaling + i, x + i * dim_product, 1));
            cdpErrchk_blas(cublasDaxpy(blasHandle, dim_product, &minus_half, d_vectorSum_scaling + i, 0, x + i * dim_product, 1));
            // re-initialise all vector dependent on new initial guess
            if(h_laplace_rank_deficient){
              cublasDdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
            }
            calcZ_v4<<<gridSize, blockSize>>>
            (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
              boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
            printf("reinitialised with random, max res %f ,  iterations %d\n", max_residual, iterations );
            restart = true;
            iterations = 0;
            threshold_reached_cpu[i] = false;
          }
        }
    }

    cudaMemcpy(threshold_reached, threshold_reached_cpu, batch_size*sizeof(bool), cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    double rAsum = 0;
    cublasDasum(blasHandle, dim_product, r, 1, &rAsum);
    //printf("STEP %d       restart  %d  residual_initSum  %f   x_initSum  %f   iteration %d   res_Asum %f   rank def %d\n",unrolling_step,restart_num,rinitsum,xinitsum, iterations, rAsum , int(h_laplace_rank_deficient));


    delete[] alpha, beta, threshold_reached_cpu, p_r, p_z, r_z;
//    printf("I: %i\n", iterations);
    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaFree(diagonalOffsets);
    cudaFree(domainBoundaryBool);
    cudaFree(d_vectorSum_scaling);
    curandDestroyGenerator(random_generator);
    //printf("end\n");
    //printarray<<<1,1>>>(x, 5);
    //CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}


void LaunchPressureKernel(const int* dimensions, const int dim_product, const int dim_size, const float *laplace_matrix,
                          float* p, float* z, float* r, float* divergence, float* x, bool* threshold_reached,
                          const float* accuracy_ptr, const int* max_iterations_ptr, const int batch_size,
                          int* iterations_gpu, const bool* boolPeriodic, const bool* laplace_rank_deficient, const bool init_with_zeros,
                          const int residual_reset_steps, const int randomized_restarts, const int unrolling_step)
{
    float minus_half = -.5f;
    float accuracy = 0.;
    int max_iterations = 0;
    bool h_laplace_rank_deficient = false;
    cudaMemcpy(&accuracy, accuracy_ptr, sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_iterations, max_iterations_ptr, sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_laplace_rank_deficient, laplace_rank_deficient, sizeof(bool),cudaMemcpyDeviceToHost);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    if(initBlasHandle) {
        cdpErrchk_blas(cublasCreate_v2(&blasHandle));
        cublasSetPointerMode_v2(blasHandle, CUBLAS_POINTER_MODE_HOST);
        initBlasHandle = false;
    }

    // if laplace matrix is rank < n -> shift singularity by solving (A+e^Te)x = b instead
    // shift "magnitude" scaled by divervence Asum
    float vectorSum[batch_size]  = {0};
    float vectorSum_scaling[batch_size] = {0};
    for (size_t i = 0; i < batch_size && h_laplace_rank_deficient==true; i++) {
      cublasSasum(blasHandle, dim_product, laplace_matrix + dim_size, 2*dim_size+1, &vectorSum_scaling[i]);
      vectorSum_scaling[i] *= .1/dim_product;
      //printf("vector scaling %f\n", vectorSum_scaling[i]);
    }
    float* d_vectorSum_scaling;
    cudaMalloc(&d_vectorSum_scaling, batch_size*sizeof(float));
    cudaMemcpy(d_vectorSum_scaling, vectorSum_scaling, batch_size*sizeof(float), cudaMemcpyHostToDevice);


    // CG helper variables variables init
    float *alpha = new float[batch_size], *beta = new float[batch_size];
    float oneScalar = 1.0f;
    bool *threshold_reached_cpu = new bool[batch_size];
    float *p_r = new float[batch_size], *p_z = new float[batch_size], *r_z = new float[batch_size];

    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;

    if (init_with_zeros){
      cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initWithZeros<float>, 0, 0);
      gridSize = (dim_product*batch_size + blockSize - 1) / blockSize;
      initWithZeros<<<gridSize, blockSize>>>(x, batch_size*dim_product);
    }
    int* diagonalOffsets;
    cudaMalloc(&diagonalOffsets, 2*(dim_size*2+1)*sizeof(int));
    calcDiagonalOffsets<<<1,1>>>(diagonalOffsets, dim_size*2+1, dimensions, boolPeriodic);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    //printarray_int<<<1,1>>>(diagonalOffsets, 2*(dim_size*2+1));

    bool* domainBoundaryBool;
    cudaMalloc(&domainBoundaryBool, (dim_size*2+1)*dim_product*sizeof(bool));
    // Initialize the helper variables
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4<float>, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // First calc A * x_0, save result to z:
    for(int i = 0; i < batch_size; i++) {
        if(h_laplace_rank_deficient){
          cublasSdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
        }
        //printf("sum %f\n", vectorSum[i]);
        //calcZ_v4<<<1, 1, 2*(dim_size * 2 + 1)>>>
        calcZ_v4<<<gridSize, blockSize>>>
        (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
          boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
          CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    bool restart = true;
    float max_residual;
    int max_residual_index;
    curandGenerator_t random_generator;
    cdpErrchk_rand(curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    int checker = 1;
    int iterations = 0;
    size_t restart_num = 0;

    float rinitsum = 0;
    float xinitsum = 0;
    for (; restart_num <= randomized_restarts && restart==true ; restart_num++) {
        // Second apply result to the helper variables
        // printf("restart number %d\n", restart_num);
        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess<float>, 0, 0);
        gridSize = (dim_product + blockSize - 1) / blockSize;
        for(int i = 0; i < batch_size; i++) {
            int offset = i * dim_product;
            initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product, divergence + offset, z + offset,
                                                            p + offset, r + offset, threshold_reached + i);
        }
        // UNNECESSARY SUM FOR PRINTCHECK
        cublasSasum(blasHandle, dim_product, r, 1, &rinitsum);

        cublasSasum(blasHandle, dim_product, x, 1, &xinitsum);


        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Init residuum checker variables
        CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4<float>, 0, 0);
        gridSize = (dim_product + blockSize - 1) / blockSize;

        // Do CG-Solve
        for (; iterations < max_iterations; iterations++) {
            // reset residual every residual_reset_steps steps to avoid residual divergence
            if((iterations + 1) % residual_reset_steps == 0){
              for(int i = 0; i < batch_size; i++) {
                int offset = i * dim_product;
                if(h_laplace_rank_deficient){
                  cublasSdot(blasHandle, dim_product, x + offset, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
                }
                calcZ_v4<<<gridSize, blockSize>>>
                (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + offset, z + offset,
                  boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);

                initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product, divergence + offset, z + offset,
                                                                p + offset, r + offset, threshold_reached + i);
              }
              CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            }

            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                if(h_laplace_rank_deficient){
                  cublasSdot(blasHandle, dim_product, p + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);

                }
                //printf("iter  %d    sum  %f\n", iterations, vectorSum[i]);
                //calcZ_v4<<<1,1, 2*(dim_size * 2 + 1)>>>
                calcZ_v4<<<gridSize, blockSize>>>
                (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, p + i * dim_product, z + i * dim_product,
                   boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);

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

            // print residuum sum
            /*float rAsum = 0;
            cublasSasum(blasHandle, dim_product, r, 1, &rAsum);
            printf("iteration %d   res_Asum %f   rank def %d\n",iterations, rAsum , int(h_laplace_rank_deficient));*/


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

        // check whether restart might be necessary - initialise with random data
        restart = false;
        for (size_t i = 0; i < batch_size; i++) {
          if(h_laplace_rank_deficient){
            cublasSdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
          }
          //calcZ_v4<<<1, 1, 2*(dim_size * 2 + 1)>>>
          calcZ_v4<<<gridSize, blockSize>>>
          (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
            boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
          CUDA_CHECK_RETURN(cudaDeviceSynchronize());
          oneScalar = -1.f;
          cublasSaxpy(blasHandle, dim_product, &oneScalar, divergence + i * dim_product, 1, z + i * dim_product, 1);
          oneScalar = 1.f;
          cublasIsamax(blasHandle, dim_product, z + i * dim_product, 1, &max_residual_index);
          cudaMemcpy(&max_residual, &(z+i*dim_product)[max_residual_index+1], sizeof(float), cudaMemcpyDeviceToHost);
          CUDA_CHECK_RETURN(cudaDeviceSynchronize());
          if(max_residual>accuracy && restart_num<randomized_restarts){
            // initialise new guess with small random perturbation in +- vectorSum_scaling/2
            float rescaling = .01;
            cdpErrchk_rand(curandGenerateUniform(random_generator, x + i * dim_product, dim_product));
            cdpErrchk_blas(cublasSscal(blasHandle, dim_product, &rescaling, x + i * dim_product, 1));
            cdpErrchk_blas(cublasSscal(blasHandle, dim_product, vectorSum_scaling + i, x + i * dim_product, 1));
            cdpErrchk_blas(cublasSaxpy(blasHandle, dim_product, &minus_half, d_vectorSum_scaling + i, 0, x + i * dim_product, 1));
            // re-initialise all vector dependent on new initial guess
            if(h_laplace_rank_deficient){
              cublasSdot(blasHandle, dim_product, x + i * dim_product, 1, d_vectorSum_scaling + i, 0, &vectorSum[i]);
            }
            calcZ_v4<<<gridSize, blockSize>>>
            (dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, x + i * dim_product, z + i * dim_product,
              boolPeriodic, diagonalOffsets, domainBoundaryBool, laplace_rank_deficient, vectorSum[i]);
            printf("reinitialised with random, max res %f ,  iterations %d\n", max_residual, iterations );
            restart = true;
            iterations = 0;
            threshold_reached_cpu[i] = false;
          }
        }
    }

    cudaMemcpy(threshold_reached, threshold_reached_cpu, batch_size*sizeof(bool), cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    float rAsum = 0;
    cublasSasum(blasHandle, dim_product, r, 1, &rAsum);
    //printf("STEP %d       restart  %d  residual_initSum  %f   x_initSum  %f   iteration %d   res_Asum %f   rank def %d\n",unrolling_step,restart_num,rinitsum,xinitsum, iterations, rAsum , int(h_laplace_rank_deficient));


    delete[] alpha, beta, threshold_reached_cpu, p_r, p_z, r_z;
//    printf("I: %i\n", iterations);
    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaFree(diagonalOffsets);
    cudaFree(domainBoundaryBool);
    cudaFree(d_vectorSum_scaling);
    curandDestroyGenerator(random_generator);
    //printf("end\n");
    //printarray<<<1,1>>>(x, 5);
    //CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}

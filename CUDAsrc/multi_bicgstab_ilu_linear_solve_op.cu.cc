
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <thread>

#include <cusolverSp.h>
#include <cusparse.h>
//#include <cublas.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include <cublas_v2.h>

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;

  exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
inline void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %p %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

#define cdpErrchk_sparse(ans) { cdpAssert_sparse((ans), __FILE__, __LINE__); }
inline void cdpAssert_sparse(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}

#define cdpErrchk_blas(ans) { cdpAssert_blas((ans), __FILE__, __LINE__); }
inline void cdpAssert_blas(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}


#define cdpErrchk_solver(ans) { cdpAssert_solver((ans), __FILE__, __LINE__); }
inline void cdpAssert_solver(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}

#define CUSPARSE_ALG CUSPARSE_ALG_NAIVE

// Assumes symmetric sparsity pattern (might also be filled with zeros)
__global__ void transpose_csr_into_cpy(float* csr_values, int* csr_row_ptr, int* csr_col_ind, float* csr_values_dest, const int matrix_shape){

  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < matrix_shape; row += blockDim.x * gridDim.x)
  {
      int j;
      for (int i = csr_row_ptr[row]; i < csr_row_ptr[row+1]; i++){
        j = csr_row_ptr[csr_col_ind[i]];
        while(csr_col_ind[j]!=row) j++;
        csr_values_dest[j] = csr_values[i];
      }
  }
}



__host__ void BicgstabIluLinearSolveLauncher(float * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
    const float* rhs, const int nnz_a, const int batch_size, const int matrix_shape,const float *x_old,
  float tol, int max_it,
  float *s, float *s_hat,  float *p, float *p_hat, float *r, float *rh, float *v, float *t, float *z, float * x,  //  float *csr_values, before x
  const bool transpose_operation, const int unrolling_step, bool* warning, cudaStream_t stream)//, const cublasHandle_t* b_handle_ptr, const cusparseHandle_t* handle_ptr)
{
  //cublasHandle_t b_handle = *b_handle_ptr;
  //cusparseHandle_t handle = *handle_ptr;
  auto DTYPE=CUDA_R_32F;
  cublasHandle_t b_handle = NULL;
  cdpErrchk_blas(cublasCreate(&b_handle));
  cusparseHandle_t handle = NULL;
  cdpErrchk_sparse(cusparseCreate(&handle));

  cublasSetStream(b_handle, stream);
  cusparseSetStream(handle, stream);

  cusparseOperation_t transpose_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

  float *ilu_csr_values;
  float *trans_csr_values;
  int *trans_csr_col_ind;
  int *trans_csr_row_ptr;
  float* temp_dtype;
  int *temp_int;
  cdpErrchk(cudaMalloc((void**) &ilu_csr_values, nnz_a*sizeof(float)));
  if (transpose_operation){
    cdpErrchk(cudaMalloc((void**) &trans_csr_values, nnz_a*sizeof(float)));
    cdpErrchk(cudaMalloc((void**) &trans_csr_col_ind, nnz_a*sizeof(int)));
    cdpErrchk(cudaMalloc((void**) &trans_csr_row_ptr, (matrix_shape+1)*sizeof(int)));
    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsr2csc(handle, matrix_shape, matrix_shape, nnz_a, csr_valuesA, csr_row_ptr, csr_col_ind,
                                      trans_csr_values, trans_csr_col_ind, trans_csr_row_ptr,
                                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    //do a pointer swap (swapped back in the end)
    temp_dtype = csr_valuesA;
    csr_valuesA = trans_csr_values;
    trans_csr_values = temp_dtype;

    temp_int = csr_col_ind;
    csr_col_ind = trans_csr_col_ind;
    trans_csr_col_ind = temp_int;

    temp_int = csr_row_ptr;
    csr_row_ptr = trans_csr_row_ptr;
    trans_csr_row_ptr = temp_int;
  }
  //printf("allocation done\n");
  // Copy csr_values in new array since iLU overwrites array
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle, nnz_a, csr_valuesA, 1, ilu_csr_values, 1));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  cusparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrL = NULL;
  cusparseMatDescr_t descrU = NULL;

  csrilu02Info_t infoA = NULL;
  csrsv2Info_t  infoL  = NULL;
  csrsv2Info_t  infoU  = NULL;

  int pBufferSizeA;
  int pBufferSizeL;
  int pBufferSizeU;
  int pBufferSize;
  void *pBuffer = NULL;

  int structural_zero;
  int numerical_zero;
  const cusparseSolvePolicy_t policyA = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyU = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t transL  = transpose_op;
  const cusparseOperation_t transU  = transpose_op;

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrA));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrU));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrilu02Info(&infoA));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoU));

  cusparseScsrilu02_bufferSize(handle, matrix_shape, nnz_a,
    descrA, ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, &pBufferSizeA);
  cusparseScsrsv2_bufferSize(handle, transL, matrix_shape, nnz_a,
    descrL, ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, &pBufferSizeL);
  cusparseScsrsv2_bufferSize(handle, transU, matrix_shape, nnz_a,
    descrU, ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, &pBufferSizeU);

  pBufferSize = max(pBufferSizeA,max(pBufferSizeL,pBufferSizeU));
  cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));

  //printf("LU buffer allocated\n");

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrilu02_analysis(handle, matrix_shape, nnz_a, descrA,
    ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
    printf("BiCG STEP %d   A(%d,%d) is missing, transpose %d\n", unrolling_step,structural_zero, structural_zero, transpose_operation);
    printf("DATA    \n");
    print_section<<<1,1>>>(ilu_csr_values, csr_row_ptr, structural_zero+1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    printf("Original DATA    \n");
    print_section<<<1,1>>>(csr_valuesA, csr_row_ptr, structural_zero+1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    printf("INDICES \n");
    print_section_int<<<1,1>>>(csr_col_ind, csr_row_ptr,structural_zero+1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    printf("ROW     \n");
    printarray_int<<<1,1>>>(&csr_row_ptr[structural_zero],3);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  }

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrilu02(handle, matrix_shape, nnz_a, descrA,
    ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
     printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }



  cusparseScsrsv2_analysis(handle, transL, matrix_shape, nnz_a, descrL,
    ilu_csr_values, csr_row_ptr, csr_col_ind,
    infoL, policyL, pBuffer);

  cusparseScsrsv2_analysis(handle, transU, matrix_shape, nnz_a, descrU,
    ilu_csr_values, csr_row_ptr, csr_col_ind,
    infoU, policyU, pBuffer);

  void *bicgBuffer = NULL;
  size_t bicgBufferSize;

  float help = 1.;
  float help2 = 0.;
  float alpha = 1.;
  float rho = 1.;
  float rhop = 1.;
  float omega = 1.;
  float beta;
  float nrm_r;

  int it_count = 0;

  float init_nrm = -1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, x, 1, &init_nrm));
  float mat_vec_nrm = - 1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, nnz_a, csr_valuesA, 1, &mat_vec_nrm));
  float rhs_nrm = - 1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, rhs, 1, &rhs_nrm));

  if(isnan(init_nrm)||isnan(mat_vec_nrm)||isnan(rhs_nrm)){
    printf("BiCG STEP: %d  InitNorm  %f    mat_vec_nrm  %f   rhs_nrm %f    xpoint  %p     xoldp  %p     matpoint  %p\n",
    unrolling_step, init_nrm,mat_vec_nrm, rhs_nrm, (void*)x, (void*) x_old,(void*)csr_valuesA);
    bool warn = true;
    cdpErrchk(cudaMemcpy(warning, &warn,sizeof(bool), cudaMemcpyHostToDevice));
  }

  float resultnorm;


  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, x_old, 1, x, 1));
  //printf("start of loop\n");
  int restart_num =2;
  for (int restart = 0; restart<restart_num; restart++)
  {
    // COMPUTE RESIDUAL r = b - A * X_0
      // r = A * x_0 + 0.*r
    help=1; help2=0;
    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx_bufferSize(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                                &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                                x, DTYPE, &help2,  DTYPE, r,   DTYPE, DTYPE, &bicgBufferSize));

    cdpErrchk(cudaMalloc((void**)&bicgBuffer, bicgBufferSize));

    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                     &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                     x, DTYPE, &help2,  DTYPE, r,   DTYPE, DTYPE, bicgBuffer));
      // r = b - r
    help2 = -1.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, r, 1));
    help = 1.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &help, rhs, 1, r, 1));

    // initial guess lucky Check
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
    //printf("init resid norm %f      xpoint  %p   \n", nrm_r, (void*) x);
    if (nrm_r < tol){
      goto endofloop;
    }

    // copy p = r  &  r_hat = r
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, r, 1, p, 1));
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, r, 1, rh, 1));

    // initialise v & p with zeros
    help2 = 0.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, v, 1));
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, p, 1));


    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, x, 1, &init_nrm));
    // MAIN BICGSTAB LOOP
    //printf("main loop\n");
    for (int i = 0; i < max_it; i++) {
      //printf("i % d\n",i );
      it_count ++;
      // rho = <r_hat, r>
      rhop = rho;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, r, 1, rh, 1, &rho));
      //12: beta = (rho_{i} / rho_{i-1}) ( alpha / \mega )
      beta= (rho/rhop)*(alpha/omega);
      omega = -omega;
      help = 1.0;
      //13: p = r + beta * (p - omega * v)
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, v, 1, p, 1));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &beta, p, 1));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &help,  r, 1, p, 1));
      omega = - omega;

        // solve L*z = rhs
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
         ilu_csr_values, csr_row_ptr, csr_col_ind, infoL,
         p, z, policyL, pBuffer));
      // solve U*x = z
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
         ilu_csr_values, csr_row_ptr, csr_col_ind, infoU,
         z, p_hat, policyU, pBuffer));

      // v = A * p_hat
      help = 1.;
      help2 = 0.;
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op,  matrix_shape, matrix_shape, nnz_a,
                                       &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                       p_hat,   DTYPE, &help2,  DTYPE, v,   DTYPE, DTYPE, bicgBuffer));

      // alpha = rho_i / <r_hat,v>
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, rh, 1, v, 1, &alpha));
      alpha = rho/alpha;

      // x = x + alpha * p_hat
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &alpha, p_hat, 1, x, 1));

      // s = r - alpha * v    ::: S = R :::
      alpha = - alpha;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &alpha, v, 1, r, 1));
      alpha = - alpha;
      // convergence Check
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
      //printf("iteration %d   norm %f\n",i, nrm_r);
      if (nrm_r < tol){
        break;
      }

      //M \hat{s} = r (sparse lower and upper triangular solves)
      help = 1.;

      // solve L*z = r
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
        ilu_csr_values, csr_row_ptr, csr_col_ind, infoL,
        r, z, policyL, pBuffer));
      // solve U*s_hat = z
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
        ilu_csr_values, csr_row_ptr, csr_col_ind, infoU,
        z, s_hat, policyU, pBuffer));


      // t = A * s_hat
      help2 = 0.;
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                       &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                       s_hat,   DTYPE, &help2,  DTYPE, t,   DTYPE, DTYPE, bicgBuffer));

      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, t, 1, r, 1, &help));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, t, 1, t, 1, &help2));
      omega = help/help2;

      // x = x + omega * s_hat
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, s_hat, 1, x, 1));

      // r = s - omega * t
      omega = -omega;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, t, 1, r, 1));
      omega = - omega;

      // convergence check
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
      //printf("iteration %d   norm %f\n", i, nrm_r);
      if (nrm_r < tol){
        break;
      }
    }
    if (nrm_r > tol*100||isnan(nrm_r)){
      int minGridSize = 0;
      int gridSize    = 0;
      int blockSize   = 0;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initWithZeros<float>, 0, 0);
      gridSize = (matrix_shape + blockSize - 1) / blockSize;
      initWithZeros<<<gridSize, blockSize>>>(x, matrix_shape);
      float  matnorm;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, nnz_a, csr_valuesA,1,&matnorm));
      //cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help, x, 1))
      printf("\n\nRE-INIT WITH ZEROS!  STEP:  %d   csrvalNorm: %f \n\n", unrolling_step, matnorm);
      CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    else{
      break;
    }
  }
  endofloop:

  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, x, 1, &resultnorm));
  if (isnan(resultnorm)){
    printf("BiCG STEP: %d  InitNorm  %f    rhs_nrm  %f    ResultNorm  %f    ResidNorm  %f    after iter  %d    xpoint  %p\n",
    unrolling_step, init_nrm, rhs_nrm, resultnorm, nrm_r , it_count, (void *)x);
  }

  int rptr_cp[3] = {0};
  cdpErrchk(cudaMemcpy(rptr_cp, &csr_row_ptr[matrix_shape-2],3*sizeof(int), cudaMemcpyDeviceToHost));
  //printf("row pointer %d %d %d\n", rptr_cp[0], rptr_cp[1], rptr_cp[2]);
    // step 6: free resources
    if (transpose_operation){
      temp_dtype = csr_valuesA;
      csr_valuesA = trans_csr_values;
      trans_csr_values = temp_dtype;

      temp_int = csr_col_ind;
      csr_col_ind = trans_csr_col_ind;
      trans_csr_col_ind = temp_int;

      temp_int = csr_row_ptr;
      csr_row_ptr = trans_csr_row_ptr;
      trans_csr_row_ptr = temp_int;
      cudaFree(trans_csr_values);
      cudaFree(trans_csr_col_ind);
      cudaFree(trans_csr_row_ptr);
    }
    // _____________ NON FINAL: RETURN  LU FACTORISED ARRAY ___________________
    //  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle,nnz_a, csr_values, 1, csr_valuesA,1));
    //_______________________________________________________________________

    cudaFree(pBuffer);
    cudaFree(bicgBuffer);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrL);
    cusparseDestroyMatDescr(descrU);
    cusparseDestroyCsrilu02Info(infoA);
    cusparseDestroyCsrsv2Info(infoL);
    cusparseDestroyCsrsv2Info(infoU);
    cusparseDestroy(handle);
    cublasDestroy(b_handle);
    cudaFree(ilu_csr_values);
}

__host__ void MultiBicgstabIluLinearSolveLauncher(float * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
    const float* rhs, const int batch_size, const int* matrix_shape, const int dim_size,const float *x_old,
  float* accuracy, int max_it,
  float *s, float *s_hat,  float *p, float *p_hat, float *r, float *rh, float *v, float *t, float *z, float * x,  //  float *csr_values, before x
  const bool transpose_operation, const int unrolling_step, bool* warning)
{

    /*cublasHandle_t b_handle = NULL;
    cdpErrchk_blas(cublasCreate(&b_handle));
    cusparseHandle_t handle = NULL;
    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreate(&handle));*/

    int matrix_shape_cpu[3];
    cdpErrchk(cudaMemcpy(matrix_shape_cpu, matrix_shape, dim_size*sizeof(int), cudaMemcpyDeviceToHost));

    int nnZ[3] = {0};  // additional zero at beginning, nnZ also has purpose to act as array-offsets, which is 0 for first solve
    int matrix_shape_offset = 0;
    for (size_t i = 0; i < dim_size; i++) {
      matrix_shape_offset += matrix_shape_cpu[i];
      cdpErrchk(cudaMemcpyAsync(&nnZ[i], &csr_row_ptr[matrix_shape_offset+i], sizeof(int), cudaMemcpyDeviceToHost));
    }
    float tol;
    cdpErrchk(cudaMemcpy(&tol, accuracy, sizeof(float), cudaMemcpyDeviceToHost));
    //cdpErrchk(cudaMemcpy(nnZ, &csr_row_ptr[matrix_shape_X], sizeof(int), cudaMemcpyDeviceToHost));
    //cdpErrchk(cudaMemcpy(&nnZ[1], &csr_row_ptr[matrix_shape_X+1+matrix_shape_Y], sizeof(int), cudaMemcpyDeviceToHost));
    cdpErrchk(cudaDeviceSynchronize());
    //printf("tol %f\n",tol );
    //printf("USING DUAL SOVE, nnz  %d  %d\n", nnZ[0], nnZ[1] );
    /*std::thread linSolveX(BicgstabIluLinearSolveLauncher,
                          csr_valuesA, csr_row_ptr, csr_col_ind, rhs, nnZ[0], batch_size, matrix_shape_X, x_old,
                          tol, max_it, s, s_hat, p, p_hat, r, rh, v, t, z, x, transpose_operation, unrolling_step); //, &b_handle, &handle);
    std::thread linSolveY(BicgstabIluLinearSolveLauncher,
                          &csr_valuesA[nnZ[0]], &csr_row_ptr[matrix_shape_X+1], &csr_col_ind[nnZ[0]], &rhs[matrix_shape_X], nnZ[1], batch_size, matrix_shape_Y, &x_old[matrix_shape_X],
                          tol, max_it, &s[matrix_shape_X], &s_hat[matrix_shape_X], &p[matrix_shape_X], &p_hat[matrix_shape_X], &r[matrix_shape_X], &rh[matrix_shape_X], &v[matrix_shape_X],
                          &t[matrix_shape_X], &z[matrix_shape_X], &x[matrix_shape_X], transpose_operation, unrolling_step);//, &b_handle, &handle);
    linSolveX.join();
    linSolveY.join();*/

    int nnZ_offset = 0;
    matrix_shape_offset = 0;
    std::thread solves[3];
    cudaStream_t streams[3];
    for (size_t i = 0; i < dim_size; i++) {
      cudaStreamCreate(&streams[i]);
      //printf("starting thread %d\n",i );
      solves[i] = std::thread(BicgstabIluLinearSolveLauncher, &csr_valuesA[nnZ_offset], &csr_row_ptr[matrix_shape_offset+i], &csr_col_ind[nnZ_offset], &rhs[matrix_shape_offset],
                                      nnZ[i], batch_size, matrix_shape_cpu[i],
                                      &x_old[matrix_shape_offset], tol, max_it, &s[matrix_shape_offset], &s_hat[matrix_shape_offset], &p[matrix_shape_offset],
                                      &p_hat[matrix_shape_offset], &r[matrix_shape_offset], &rh[matrix_shape_offset], &v[matrix_shape_offset], &t[matrix_shape_offset],
                                      &z[matrix_shape_offset], &x[matrix_shape_offset], transpose_operation, unrolling_step, warning, streams[i]);
      matrix_shape_offset += matrix_shape_cpu[i];
      nnZ_offset += nnZ[i];
    }
    for (size_t i = 0; i < dim_size; i++) {
      solves[i].join();
      cudaStreamDestroy(streams[i]);
      //printf("join finished for %d\n", i);
    }

    /*int nnZ_offset = 0;
    matrix_shape_offset = 0;
    for (size_t i = 0; i < dim_size; i++) {
      BicgstabIluLinearSolveLauncher( &csr_valuesA[nnZ_offset], &csr_row_ptr[matrix_shape_offset+i], &csr_col_ind[nnZ_offset], &rhs[matrix_shape_offset],
                                      nnZ[i], batch_size, matrix_shape_cpu[i],
                                      &x_old[matrix_shape_offset], tol, max_it, &s[matrix_shape_offset], &s_hat[matrix_shape_offset], &p[matrix_shape_offset],
                                      &p_hat[matrix_shape_offset], &r[matrix_shape_offset], &rh[matrix_shape_offset], &v[matrix_shape_offset], &t[matrix_shape_offset],
                                      &z[matrix_shape_offset], &x[matrix_shape_offset], transpose_operation, unrolling_step, warning);
      matrix_shape_offset += matrix_shape_cpu[i];
      nnZ_offset += nnZ[i];
    }*/
    /*BicgstabIluLinearSolveLauncher(csr_valuesA, csr_row_ptr, csr_col_ind, rhs, nnZ[0], batch_size, matrix_shape_X, x_old,
                          tol, max_it, s, s_hat, p, p_hat, r, rh, v, t, z, x, transpose_operation);//, &b_handle, &handle);
    printf("matrix shape x %d\n", matrix_shape_X);
    BicgstabIluLinearSolveLauncher(&csr_valuesA[nnZ[0]], &csr_row_ptr[matrix_shape_X+1], &csr_col_ind[nnZ[0]], &rhs[matrix_shape_X], nnZ[1], batch_size, matrix_shape_Y, &x_old[matrix_shape_X],
                          tol, max_it, &s[matrix_shape_X], &s_hat[matrix_shape_X], &p[matrix_shape_X], &p_hat[matrix_shape_X], &r[matrix_shape_X], &rh[matrix_shape_X], &v[matrix_shape_X],
                          &t[matrix_shape_X], &z[matrix_shape_X], &x[matrix_shape_X], transpose_operation);//, &b_handle, &handle);*/
}



// ##############################################################################################
// --------------------------------------  DOUBLE VERSION  --------------------------------------
// ##############################################################################################


void BicgstabIluLinearSolveLauncherDouble(double * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
    const double* rhs, const int nnz_a, const int batch_size, const int matrix_shape,const double *x_old,
  float tol, int max_it,
  double *s, double *s_hat,  double *p, double *p_hat, double *r, double *rh, double *v, double *t, double *z, double * x,  //  double *csr_values, before x
  const bool transpose_operation, const int unrolling_step, bool* warning, cudaStream_t stream )//, const cublasHandle_t* b_handle_ptr, const cusparseHandle_t* handle_ptr)
{
  //cublasHandle_t b_handle = *b_handle_ptr;
  //cusparseHandle_t handle = *handle_ptr;
  auto DTYPE=CUDA_R_64F;
  cublasHandle_t b_handle = NULL;
  cdpErrchk_blas(cublasCreate(&b_handle));
  cusparseHandle_t handle = NULL;
  cdpErrchk_sparse(cusparseCreate(&handle));

  cublasSetStream(b_handle, stream);
  cusparseSetStream(handle, stream);

  cusparseOperation_t transpose_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

  double *ilu_csr_values;
  double *trans_csr_values;
  int *trans_csr_col_ind;
  int *trans_csr_row_ptr;
  double* temp_dtype;
  int *temp_int;
  cdpErrchk(cudaMalloc((void**) &ilu_csr_values, nnz_a*sizeof(double)));
  if (transpose_operation){
    cdpErrchk(cudaMalloc((void**) &trans_csr_values, nnz_a*sizeof(double)));
    cdpErrchk(cudaMalloc((void**) &trans_csr_col_ind, nnz_a*sizeof(int)));
    cdpErrchk(cudaMalloc((void**) &trans_csr_row_ptr, (matrix_shape+1)*sizeof(int)));
    cusparseSetStream(handle, stream);
    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsr2csc(handle, matrix_shape, matrix_shape, nnz_a, csr_valuesA, csr_row_ptr, csr_col_ind,
                                      trans_csr_values, trans_csr_col_ind, trans_csr_row_ptr,
                                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));

    //do a pointer swap (swapped back in the end)
    temp_dtype = csr_valuesA;
    csr_valuesA = trans_csr_values;
    trans_csr_values = temp_dtype;

    temp_int = csr_col_ind;
    csr_col_ind = trans_csr_col_ind;
    trans_csr_col_ind = temp_int;

    temp_int = csr_row_ptr;
    csr_row_ptr = trans_csr_row_ptr;
    trans_csr_row_ptr = temp_int;
  }
  //printf("allocation done\n");
  // Copy csr_values in new array since iLU overwrites array
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDcopy(b_handle, nnz_a, csr_valuesA, 1, ilu_csr_values, 1));
  CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
  cusparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrL = NULL;
  cusparseMatDescr_t descrU = NULL;

  csrilu02Info_t infoA = NULL;
  csrsv2Info_t  infoL  = NULL;
  csrsv2Info_t  infoU  = NULL;

  int pBufferSizeA;
  int pBufferSizeL;
  int pBufferSizeU;
  int pBufferSize;
  void *pBuffer = NULL;

  int structural_zero;
  int numerical_zero;
  const cusparseSolvePolicy_t policyA = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyU = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t transL  = transpose_op;
  const cusparseOperation_t transU  = transpose_op;

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrA));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateMatDescr(&descrU));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrilu02Info(&infoA));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoL));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoU));

  cusparseDcsrilu02_bufferSize(handle, matrix_shape, nnz_a,
    descrA, ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, &pBufferSizeA);
  cusparseDcsrsv2_bufferSize(handle, transL, matrix_shape, nnz_a,
    descrL, ilu_csr_values, csr_row_ptr, csr_col_ind, infoL, &pBufferSizeL);
  cusparseDcsrsv2_bufferSize(handle, transU, matrix_shape, nnz_a,
    descrU, ilu_csr_values, csr_row_ptr, csr_col_ind, infoU, &pBufferSizeU);

  pBufferSize = max(pBufferSizeA,max(pBufferSizeL,pBufferSizeU));
  cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));

  //printf("LU buffer allocated\n");

  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrilu02_analysis(handle, matrix_shape, nnz_a, descrA,
    ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
    printf("BiCG STEP %d   A(%d,%d) is missing, transpose %d\n", unrolling_step,structural_zero, structural_zero, transpose_operation);
    printf("DATA    \n");
    print_section<<<1,1>>>(ilu_csr_values, csr_row_ptr, structural_zero+1);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    printf("Original DATA    \n");
    print_section<<<1,1>>>(csr_valuesA, csr_row_ptr, structural_zero+1);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    printf("INDICES \n");
    print_section_int<<<1,1>>>(csr_col_ind, csr_row_ptr,structural_zero+1);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    printf("ROW     \n");
    printarray_int<<<1,1>>>(&csr_row_ptr[structural_zero],3);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
  }

  CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
  cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrilu02(handle, matrix_shape, nnz_a, descrA,
    ilu_csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
     printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }



  cusparseDcsrsv2_analysis(handle, transL, matrix_shape, nnz_a, descrL,
    ilu_csr_values, csr_row_ptr, csr_col_ind,
    infoL, policyL, pBuffer);

  cusparseDcsrsv2_analysis(handle, transU, matrix_shape, nnz_a, descrU,
    ilu_csr_values, csr_row_ptr, csr_col_ind,
    infoU, policyU, pBuffer);

  void *bicgBuffer = NULL;
  size_t bicgBufferSize;

  double help = 1.;
  double help2 = 0.;
  double alpha = 1.;
  double rho = 1.;
  double rhop = 1.;
  double omega = 1.;
  double beta;
  double nrm_r;

  int it_count = 0;

  double init_nrm = -1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, x, 1, &init_nrm));
  double mat_vec_nrm = - 1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, nnz_a, csr_valuesA, 1, &mat_vec_nrm));
  double rhs_nrm = - 1;
  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, rhs, 1, &rhs_nrm));

  if(isnan(init_nrm)||isnan(mat_vec_nrm)||isnan(rhs_nrm)){
    printf("BiCG STEP: %d  InitNorm  %f    mat_vec_nrm  %f   rhs_nrm %f    xpoint  %p     xoldp  %p     matpoint  %p\n",
    unrolling_step, init_nrm,mat_vec_nrm, rhs_nrm, (void*)x, (void*) x_old,(void*)csr_valuesA);
    bool warn = true;
    cdpErrchk(cudaMemcpy(warning, &warn,sizeof(bool), cudaMemcpyHostToDevice));
  }

  double resultnorm;


  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDcopy(b_handle, matrix_shape, x_old, 1, x, 1));
  //printf("start of loop\n");
  int restart_num =2;
  for (int restart = 0; restart<restart_num; restart++)
  {
    // COMPUTE RESIDUAL r = b - A * X_0
      // r = A * x_0 + 0.*r
    help=1; help2=0;
    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx_bufferSize(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                                &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                                x, DTYPE, &help2,  DTYPE, r,   DTYPE, DTYPE, &bicgBufferSize));

    cdpErrchk(cudaMalloc((void**)&bicgBuffer, bicgBufferSize));

    cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                     &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                     x, DTYPE, &help2,  DTYPE, r,   DTYPE, DTYPE, bicgBuffer));
      // r = b - r
    help2 = -1.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDscal(b_handle, matrix_shape, &help2, r, 1));
    help = 1.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &help, rhs, 1, r, 1));

    // initial guess lucky Check
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
    //printf("init resid norm %f      xpoint  %p   \n", nrm_r, (void*) x);
    if (nrm_r < tol){
      goto endofloop;
    }

    // copy p = r  &  r_hat = r
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDcopy(b_handle, matrix_shape, r, 1, p, 1));
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDcopy(b_handle, matrix_shape, r, 1, rh, 1));

    // initialise v & p with zeros
    help2 = 0.;
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDscal(b_handle, matrix_shape, &help2, v, 1));
    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDscal(b_handle, matrix_shape, &help2, p, 1));


    cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, x, 1, &init_nrm));
    // MAIN BICGSTAB LOOP
    //printf("main loop\n");
    for (int i = 0; i < max_it; i++) {
      //printf("i % d\n",i );
      it_count ++;
      // rho = <r_hat, r>
      rhop = rho;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDdot(b_handle, matrix_shape, r, 1, rh, 1, &rho));
      //12: beta = (rho_{i} / rho_{i-1}) ( alpha / \mega )
      beta= (rho/rhop)*(alpha/omega);
      omega = -omega;
      help = 1.0;
      //13: p = r + beta * (p - omega * v)
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &omega, v, 1, p, 1));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDscal(b_handle, matrix_shape, &beta, p, 1));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &help,  r, 1, p, 1));
      omega = - omega;

        // solve L*z = rhs
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
         ilu_csr_values, csr_row_ptr, csr_col_ind, infoL,
         p, z, policyL, pBuffer));
      // solve U*x = z
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
         ilu_csr_values, csr_row_ptr, csr_col_ind, infoU,
         z, p_hat, policyU, pBuffer));

      // v = A * p_hat
      help = 1.;
      help2 = 0.;
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op,  matrix_shape, matrix_shape, nnz_a,
                                       &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                       p_hat,   DTYPE, &help2,  DTYPE, v,   DTYPE, DTYPE, bicgBuffer));

      // alpha = rho_i / <r_hat,v>
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDdot(b_handle, matrix_shape, rh, 1, v, 1, &alpha));
      alpha = rho/alpha;

      // x = x + alpha * p_hat
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &alpha, p_hat, 1, x, 1));

      // s = r - alpha * v    ::: S = R :::
      alpha = - alpha;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &alpha, v, 1, r, 1));
      alpha = - alpha;
      // convergence Check
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
      //printf("iteration %d   norm %f\n",i, nrm_r);
      if (nrm_r < tol){
        break;
      }

      //M \hat{s} = r (sparse lower and upper triangular solves)
      help = 1.;

      // solve L*z = r
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
        ilu_csr_values, csr_row_ptr, csr_col_ind, infoL,
        r, z, policyL, pBuffer));
      // solve U*s_hat = z
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseDcsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
        ilu_csr_values, csr_row_ptr, csr_col_ind, infoU,
        z, s_hat, policyU, pBuffer));


      // t = A * s_hat
      help2 = 0.;
      cusparseSetStream(handle, stream); cdpErrchk_sparse(cusparseCsrmvEx(handle, CUSPARSE_ALG, transpose_op, matrix_shape, matrix_shape, nnz_a,
                                       &help, DTYPE, descrA, csr_valuesA, DTYPE, csr_row_ptr, csr_col_ind,
                                       s_hat,   DTYPE, &help2,  DTYPE, t,   DTYPE, DTYPE, bicgBuffer));

      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDdot(b_handle, matrix_shape, t, 1, r, 1, &help));
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDdot(b_handle, matrix_shape, t, 1, t, 1, &help2));
      omega = help/help2;

      // x = x + omega * s_hat
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &omega, s_hat, 1, x, 1));

      // r = s - omega * t
      omega = -omega;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDaxpy(b_handle, matrix_shape, &omega, t, 1, r, 1));
      omega = - omega;

      // convergence check
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
      //printf("iteration %d   norm %f\n", i, nrm_r);
      if (nrm_r < tol){
        break;
      }
    }
    if (nrm_r > tol*100||isnan(nrm_r)){
      int minGridSize = 0;
      int gridSize    = 0;
      int blockSize   = 0;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initWithZeros<double>, 0, 0);
      gridSize = (matrix_shape + blockSize - 1) / blockSize;
      initWithZeros<<<gridSize, blockSize>>>(x, matrix_shape);
      double  matnorm;
      cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, nnz_a, csr_valuesA,1,&matnorm));
      //cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help, x, 1))
      printf("\n\nRE-INIT WITH ZEROS!  STEP:  %d   csrvalNorm: %f \n\n", unrolling_step, matnorm);
      CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
    }
    else{
      break;
    }
  }
  endofloop:

  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasDnrm2(b_handle, matrix_shape, x, 1, &resultnorm));
  if (isnan(resultnorm)){
    printf("BiCG STEP: %d  InitNorm  %f    rhs_nrm  %f    ResultNorm  %f    ResidNorm  %f    after iter  %d    xpoint  %p\n",
    unrolling_step, init_nrm, rhs_nrm, resultnorm, nrm_r , it_count, (void *)x);
  }

  int rptr_cp[3] = {0};
  cdpErrchk(cudaMemcpy(rptr_cp, &csr_row_ptr[matrix_shape-2],3*sizeof(int), cudaMemcpyDeviceToHost));
  //printf("row pointer %d %d %d\n", rptr_cp[0], rptr_cp[1], rptr_cp[2]);
    // step 6: free resources
    if (transpose_operation){
      temp_dtype = csr_valuesA;
      csr_valuesA = trans_csr_values;
      trans_csr_values = temp_dtype;

      temp_int = csr_col_ind;
      csr_col_ind = trans_csr_col_ind;
      trans_csr_col_ind = temp_int;

      temp_int = csr_row_ptr;
      csr_row_ptr = trans_csr_row_ptr;
      trans_csr_row_ptr = temp_int;
      cudaFree(trans_csr_values);
      cudaFree(trans_csr_col_ind);
      cudaFree(trans_csr_row_ptr);
    }
    // _____________ NON FINAL: RETURN  LU FACTORISED ARRAY ___________________
    //  cublasSetStream(b_handle, stream); cdpErrchk_blas(cublasScopy(b_handle,nnz_a, csr_values, 1, csr_valuesA,1));
    //_______________________________________________________________________

    cudaFree(pBuffer);
    cudaFree(bicgBuffer);

    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrL);
    cusparseDestroyMatDescr(descrU);
    cusparseDestroyCsrilu02Info(infoA);
    cusparseDestroyCsrsv2Info(infoL);
    cusparseDestroyCsrsv2Info(infoU);
    cusparseDestroy(handle);
    cublasDestroy(b_handle);
    cudaFree(ilu_csr_values);
    //printf("Finished One solve\n");
}

 void MultiBicgstabIluLinearSolveLauncher(double * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
    const double* rhs, const int batch_size, const int* matrix_shape, const int dim_size, const double *x_old,
  float* accuracy, int max_it,
  double *s, double *s_hat,  double *p, double *p_hat, double *r, double *rh, double *v, double *t, double *z, double * x,  //  double *csr_values, before x
  const bool transpose_operation, const int unrolling_step, bool* warning)
{

    /*cublasHandle_t b_handle = NULL;
    cdpErrchk_blas(cublasCreate(&b_handle));
    cusparseHandle_t handle = NULL;
    cdpErrchk_sparse(cusparseCreate(&handle));*/

    int matrix_shape_cpu[3];
    cdpErrchk(cudaMemcpy(matrix_shape_cpu, matrix_shape, dim_size*sizeof(int), cudaMemcpyDeviceToHost));

    int nnZ[3] = {0};  // additional zero at beginning, nnZ also has purpose to act as array-offsets, which is 0 for first solve
    int matrix_shape_offset = 0;
    for (size_t i = 0; i < dim_size; i++) {
      matrix_shape_offset += matrix_shape_cpu[i];
      cdpErrchk(cudaMemcpyAsync(&nnZ[i], &csr_row_ptr[matrix_shape_offset+i], sizeof(int), cudaMemcpyDeviceToHost));
    }
    float tol;
    cdpErrchk(cudaMemcpy(&tol, accuracy, sizeof(float), cudaMemcpyDeviceToHost));
    //cdpErrchk(cudaMemcpy(nnZ, &csr_row_ptr[matrix_shape_X], sizeof(int), cudaMemcpyDeviceToHost));
    //cdpErrchk(cudaMemcpy(&nnZ[1], &csr_row_ptr[matrix_shape_X+1+matrix_shape_Y], sizeof(int), cudaMemcpyDeviceToHost));
    cdpErrchk(cudaDeviceSynchronize());

    //printf("USING DUAL SOVE, nnz  %d  %d\n", nnZ[0], nnZ[1] );
    /*std::thread linSolveX(BicgstabIluLinearSolveLauncher,
                          csr_valuesA, csr_row_ptr, csr_col_ind, rhs, nnZ[0], batch_size, matrix_shape_X, x_old,
                          tol, max_it, s, s_hat, p, p_hat, r, rh, v, t, z, x, transpose_operation, unrolling_step); //, &b_handle, &handle);
    std::thread linSolveY(BicgstabIluLinearSolveLauncher,
                          &csr_valuesA[nnZ[0]], &csr_row_ptr[matrix_shape_X+1], &csr_col_ind[nnZ[0]], &rhs[matrix_shape_X], nnZ[1], batch_size, matrix_shape_Y, &x_old[matrix_shape_X],
                          tol, max_it, &s[matrix_shape_X], &s_hat[matrix_shape_X], &p[matrix_shape_X], &p_hat[matrix_shape_X], &r[matrix_shape_X], &rh[matrix_shape_X], &v[matrix_shape_X],
                          &t[matrix_shape_X], &z[matrix_shape_X], &x[matrix_shape_X], transpose_operation, unrolling_step);//, &b_handle, &handle);
    linSolveX.join();
    linSolveY.join();*/

    int nnZ_offset = 0;
    matrix_shape_offset = 0;
    std::thread solves[3];
    cudaStream_t streams[3];
    for (size_t i = 0; i < dim_size; i++) {
      cudaStreamCreate(&streams[i]);
      //printf("starting thread %d\n",i );
      solves[i] = std::thread(BicgstabIluLinearSolveLauncherDouble, &csr_valuesA[nnZ_offset], &csr_row_ptr[matrix_shape_offset+i], &csr_col_ind[nnZ_offset], &rhs[matrix_shape_offset],
                                      nnZ[i], batch_size, matrix_shape_cpu[i],
                                      &x_old[matrix_shape_offset], tol, max_it, &s[matrix_shape_offset], &s_hat[matrix_shape_offset], &p[matrix_shape_offset],
                                      &p_hat[matrix_shape_offset], &r[matrix_shape_offset], &rh[matrix_shape_offset], &v[matrix_shape_offset], &t[matrix_shape_offset],
                                      &z[matrix_shape_offset], &x[matrix_shape_offset], transpose_operation, unrolling_step, warning, streams[i]);
      matrix_shape_offset += matrix_shape_cpu[i];
      nnZ_offset += nnZ[i];
    }
    for (size_t i = 0; i < dim_size; i++) {
      solves[i].join();
      cudaStreamDestroy(streams[i]);
      //printf("join finished for %d\n", i);
    }

    /*int nnZ_offset = 0;
    matrix_shape_offset = 0;
    for (size_t i = 0; i < dim_size; i++) {
      BicgstabIluLinearSolveLauncherDouble( &csr_valuesA[nnZ_offset], &csr_row_ptr[matrix_shape_offset+i], &csr_col_ind[nnZ_offset], &rhs[matrix_shape_offset],
                                      nnZ[i], batch_size, matrix_shape_cpu[i],
                                      &x_old[matrix_shape_offset], tol, max_it, &s[matrix_shape_offset], &s_hat[matrix_shape_offset], &p[matrix_shape_offset],
                                      &p_hat[matrix_shape_offset], &r[matrix_shape_offset], &rh[matrix_shape_offset], &v[matrix_shape_offset], &t[matrix_shape_offset],
                                      &z[matrix_shape_offset], &x[matrix_shape_offset], transpose_operation, unrolling_step, warning, (cudaStream_t) 0);
      matrix_shape_offset += matrix_shape_cpu[i];
      nnZ_offset += nnZ[i];
    }*/
    /*BicgstabIluLinearSolveLauncher(csr_valuesA, csr_row_ptr, csr_col_ind, rhs, nnZ[0], batch_size, matrix_shape_X, x_old,
                          tol, max_it, s, s_hat, p, p_hat, r, rh, v, t, z, x, transpose_operation);//, &b_handle, &handle);
    printf("matrix shape x %d\n", matrix_shape_X);
    BicgstabIluLinearSolveLauncher(&csr_valuesA[nnZ[0]], &csr_row_ptr[matrix_shape_X+1], &csr_col_ind[nnZ[0]], &rhs[matrix_shape_X], nnZ[1], batch_size, matrix_shape_Y, &x_old[matrix_shape_X],
                          tol, max_it, &s[matrix_shape_X], &s_hat[matrix_shape_X], &p[matrix_shape_X], &p_hat[matrix_shape_X], &r[matrix_shape_X], &rh[matrix_shape_X], &v[matrix_shape_X],
                          &t[matrix_shape_X], &z[matrix_shape_X], &x[matrix_shape_X], transpose_operation);//, &b_handle, &handle);*/
}

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cusolverSp.h>
#include <cusparse.h>
//#include <cublas.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void printarray(T* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %f ", a[i]);
  }
  printf("\n");
}

__global__ void printarray_int(int* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %d ", a[i]);
  }
  printf("\n");
}

template <typename T>
__global__ void print_section(T* cval, int* rptr, int dim){
  for (size_t i = rptr[dim-1];i < rptr[dim]; i++) {
    printf(" %f ", cval[i]);
  }
  printf("\n");
}


__global__ void print_section_int(int* cval, int* rptr, int dim){
  for (size_t i = rptr[dim-1];i < rptr[dim]; i++) {
    printf(" %d ", cval[i]);
  }
  printf("\n");
}

template <typename T>
__global__ void initWithZeros(T *array, const int nnZ)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nnZ; row += blockDim.x * gridDim.x)
  {
      array[row] = 0.;
  }
}

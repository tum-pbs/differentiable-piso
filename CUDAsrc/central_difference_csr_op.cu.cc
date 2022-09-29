#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <cusolverSp.h>
#include <cusparse.h>

#include <cusolverSp_LOWLEVEL_PREVIEW.h>
//#include <cublas.h>
#include <cuda_runtime_api.h>

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;

  exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

typedef float dtype;

__device__ void calcGridLocation(int* location, const int* dimensions, const int dimProduct, const int dimSize,  const int row){
    int currentOffset = dimProduct;
    int currentIndex = row;
    for (short d = dimSize - 1; d>=0; d--){
      currentOffset /= dimensions[d];
      location[d] = currentIndex / currentOffset;
      currentIndex = currentIndex % currentOffset;
    }
}

__device__ void calcCellFluxesX(float* fluxes, const float* velocity, const int* dimensions, const int* dimPad, const int* padDepth,
  const float* cellArea, const int dimProduct, const int* dimProductPad, const int dimSize, const int* location){
    // INPUT:
    // add dimPad: conatins array offsets to get a cell of the PADDED VEL in each direction like ((1,Nx_x_pad,Nx_x_pad*Nx_y_pad),(1,Ny_x_pad,Ny_x_pad*Ny_y_pad),...) appended for each dimension
    // add dimensions: contains dimensions of various velocity arrays like ((Nx_x, Nx_y, Nx_z), (Ny_x, Ny_y, Ny_z) ...) (first index vel-direction, second index dimension)
    // add dimProductPad: total cells of the padded velocity fields like (Nx_padded,Ny_padded,Nz_padded)
    //int location[3];
    float helperVelocity;

    /*int currentOffset = dimProduct;
    int currentIndex = row;
    for (short d = dimSize - 1; d>=0; d--){
      currentOffset /= dimensions[d];                                 // THIS LOCATION IN 'dimension' IS X-SPECIFIC
      location[d] = currentIndex / currentOffset + padDepth[d*2];
      currentIndex = currentIndex % currentOffset;
    }*/

    int currentIndex = 0;
    int velocityOffset = 0; //increases as the loop goes through x,y,z velocities
    // loop over dimensions to fill flux = (x_low, x_up, y_low, y_up, z_low, z_up)
    for (short i=0; i<dimSize; i++){
      currentIndex = 0;
      for (short d=0; d<dimSize; d++){
        //currentIndex += location[d] * dimPad[i*dimSize+d];
        currentIndex += (location[d] + padDepth[2*d]) * dimPad[i*dimSize+d];
      }
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2]   = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - 1]) * cellArea[i];

      //printf(" fInd:[%d, ", currentIndex);
      currentIndex += dimPad[i*dimSize+i];
      //printf("%d], vO %d \n", currentIndex, velocityOffset);
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2+1] = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - 1]) * cellArea[i];

      velocityOffset += dimProductPad[i];
    }
}



__device__ void calcCellFluxesY(float* fluxes, const float* velocity, const int* dimensions, const int* dimPad, const int* padDepth,
  const float* cellArea, const int dimProduct, const int* dimProductPad, const int dimSize, const int* location){
    float helperVelocity;

    int currentIndex = 0;
    int velocityOffset = 0; //increases as the loop goes through x,y,z velocities
    // loop over dimensions to fill flux = (x_low, x_up, y_low, y_up, z_low, z_up)
    for (short i=0; i<dimSize; i++){
      currentIndex = 0;
      for (short d=0; d<dimSize; d++){
        currentIndex += (location[d] + padDepth[2*d]) * dimPad[i*dimSize+d];
      }
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2]   = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]) * cellArea[i];

      //printf("i:%d  cInd:%d vO:%d  hvel:%f hvel2:%f\n", i,currentIndex,velocityOffset, helperVelocity,  velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]);
      //printf("flux %f\n",fluxes[i*2] );
      currentIndex += dimPad[i*dimSize+i];
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2+1] = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]) * cellArea[i];

      //printf("i:%d  cInd:%d cO:%d  hvel:%f hvel2:%f\n", i,currentIndex,velocityOffset, helperVelocity, velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]);
      //printf("flux %f\n",fluxes[i*2+1] );
      velocityOffset += dimProductPad[i];
    }
}

__device__ void calcCellFluxesZ(float* fluxes, const float* velocity, const int* dimensions, const int* dimPad, const int* padDepth,
  const float* cellArea, const int dimProduct, const int* dimProductPad, const int dimSize, const int* location){
    float helperVelocity;

    int currentIndex = 0;
    int velocityOffset = 0; //increases as the loop goes through x,y,z velocities
    // loop over dimensions to fill flux = (x_low, x_up, y_low, y_up, z_low, z_up)
    for (short i=0; i<dimSize; i++){
      currentIndex = 0;
      for (short d=0; d<dimSize; d++){
        currentIndex += (location[d] + padDepth[2*d]) * dimPad[i*dimSize+d];
      }
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2]   = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - dimPad[i*dimSize+2]]) * cellArea[i];

      //printf("i:%d  cInd:%d vO:%d  hvel:%f hvel2:%f\n", i,currentIndex,velocityOffset, helperVelocity,  velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]);
      //printf("flux %f\n",fluxes[i*2] );
      currentIndex += dimPad[i*dimSize+i];
      helperVelocity = velocity[velocityOffset + currentIndex];
      fluxes[i*2+1] = .5 * (helperVelocity + velocity[velocityOffset + currentIndex - dimPad[i*dimSize+2]]) * cellArea[i];

      //printf("i:%d  cInd:%d cO:%d  hvel:%f hvel2:%f\n", i,currentIndex,velocityOffset, helperVelocity, velocity[velocityOffset + currentIndex - dimPad[i*dimSize+1]]);
      //printf("flux %f\n",fluxes[i*2+1] );
      velocityOffset += dimProductPad[i];
    }
}

// Converts coordinates of the simulation grid to indices of the extended mask grid with a shift to get the indices of the neighbors
//  mask_idx_before = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);
__device__ int gridIDXpaddedCenteredMasks(const int *dimensions, const int dimSize,  const int *location, int staggeredDim, int neighbourDim, int offset)//const int *cords, int cords_offset, int dim_index_offset, int offset)
{                                       //(const int *dimensions, const int dimSize, const int *location, int staggeredDim, int neighbourDim, int offset) int dim_index_offset, int offset)
    int factor = 1;
    int result = 0;
    for (int i = 0; i < dimSize; i++)
    {
        if (i == neighbourDim)
            result += factor * (location[i] + 1 + offset);
        else
            result += factor * (location[i] + 1);

        factor *= dimensions[i] + 2 - (i==staggeredDim);
    }
    return result;
}

__global__ void calcAdvetionMatrixX(const float* velocity, float* const csrMatVal, int* const csrColInd, int* const csrRowPtr, float* const diagonalArray, const bool* boolDirichletMask, const float* active_mask, const float* accessible_mask,
  const int* dimensions, const int* dimPad, const int* padDepth, const float* cellArea, const float* gridSpacing, const float* viscosity, const bool boolViscosityField,  const int dimProduct, const int* dimProductPad,
  const int dimSize, const bool* noSlipWall, const bool* boolPeriodic, const float* beta){
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < dimProduct; row += blockDim.x * gridDim.x){
    // stores indices like [x- , x+, center, y-, y+]
    //printf("start row %d    ", row);
    bool tempBoundaryBool;
    int arrayIndices[2*3+1];
    int arrayIndicesOrdered[2*3+1];
    float fluxes[2*3];
    float diagonalValue;
    bool domainBoundaryBool[2*3];
    int location[3];
    bool PRINT = 0;
    calcGridLocation(location, &dimensions[dimSize*0], dimProduct,  dimSize, row);
    // GET BOUNDARY INFORMATION OF CURRENT CELL, stored in domainBoundaryBool:  (low_x,high_x, low_y,high_y, low_z,high_z), False when on Boundary
    //int currentOffset = dimProduct;                                  // THIS dimProduct IS X-SPECIFIC
    //int currentIndex = row;
    for (short d = dimSize - 1; d>=0; d--){
      //currentOffset /= dimensions[d];                               // THIS LOCATION IN 'dimension' IS X-SPECIFIC
      //location[d] = currentIndex / currentOffset;
      domainBoundaryBool[2*d] = min(location[d],1);
      domainBoundaryBool[2*d+1] = min(dimensions[d]-1-location[d],1);                   // THIS LOCATION IN 'dimensions' IS X-SPECIFIC
      //currentIndex = currentIndex % currentOffset;
    }

    // CALCULATE ARRAY INDICES IN ASCENDING ORDER OF APPEARANCE, normally low_z comes first, followed by low_y,low_x , center...
    // arrayIndices has a different ordering in 3D version!!: (low_x,high_x, low_y,high_y, low_z,high_z, center)
    for (short d = 0; d<dimSize; d++){
      arrayIndices[2*(dimSize-1-d)] = csrRowPtr[row]+d;
      arrayIndices[2*(dimSize-1-d)+1] = csrRowPtr[row]+2*dimSize-d;
    }
    arrayIndices[2*dimSize]=csrRowPtr[row]+dimSize;
    //printf(" arrayInd [%d, %d, %d, %d, %d]    ", arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4]);
    // account for periodic boundary conditions,
    // being on a Z-boundary has largest impact, shifting the order of the subranges y,x,center
    // on a periodic Y-boundary, only x,center subrange are canged; and so on....
    for (short d = dimSize - 1; d>=0; d--){
      arrayIndices[2*d] += (d+1)*2*(1-domainBoundaryBool[2*d])*boolPeriodic[d];       // if we are on a lower periodic boundary, this index is the new largest index (in the sub-range)
      arrayIndices[2*d] += (1-domainBoundaryBool[2*d+1])*boolPeriodic[d];
      arrayIndices[2*d+1] -= (d+1)*2*(1-domainBoundaryBool[2*d+1])*boolPeriodic[d];   // if we are on a higher periodic boundary, this index is the new lowest index (in the sub-range)
      arrayIndices[2*d+1] -= (1-domainBoundaryBool[2*d])*boolPeriodic[d];
      for (short s = 0; s<d; s++){
        arrayIndices[2*s]   += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
        arrayIndices[2*s+1] += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
      }
      arrayIndices[2*dimSize] += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
      //printf("d %d: arrayInd [%d, %d, %d, %d, %d]    ", d, arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4]);
    }
    //printf("  arrayInd [%d, %d, %d, %d, %d]    ", arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4]);
    // acount for non-periodic boundaries: entries for neuman boundaries do not exist -> shift all following entries in a row down by one
    for (short i=0; i<2*dimSize+1; i++){
      arrayIndicesOrdered[arrayIndices[i]-csrRowPtr[row]]=i;
    }
    //printf("  arrayIndORD [%d, %d, %d, %d, %d]    ", arrayIndicesOrdered[0], arrayIndicesOrdered[1], arrayIndicesOrdered[2], arrayIndicesOrdered[3], arrayIndicesOrdered[4]);
    //printf("  dBB [%d, %d, %d, %d]    ", domainBoundaryBool[0], domainBoundaryBool[1], domainBoundaryBool[2], domainBoundaryBool[3]);
    for (short d=0; d<2*dimSize; d++){
      if (arrayIndicesOrdered[d]==dimSize*2) continue;
      for (short i=d+1; i<2*dimSize+1; i++){
        arrayIndices[arrayIndicesOrdered[i]] -= (1-domainBoundaryBool[arrayIndicesOrdered[d]])*
                                                (1-boolPeriodic[arrayIndicesOrdered[d]/2]);
      }
    }
    //printf("  arrayInd [%d, %d, %d, %d, %d]\n", arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4]);
    int currentOffset = dimProduct;
    // DIRICHLET BOUNDARY CONDITION:                     // THIS dimProduct IS X-SPECIFIC
    if (boolDirichletMask[row] == 1){
    //if (accessible_mask[gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0,0,-1)]==0.0f||accessible_mask[gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0,0,0)]==0.0f){
        for(short d=dimSize-1; d>=0; d--){
          currentOffset /= dimensions[d];                               // THIS LOCATION IN 'dimension' IS X-SPECIFIC
          // lower boundary colInd
          if (domainBoundaryBool[2*d]){
            csrColInd[arrayIndices[2*d]] = (row - currentOffset);
          }
          else if(boolPeriodic[d]){
            csrColInd[arrayIndices[2*d]] = (row + currentOffset*(dimensions[d]-1-(d==0)));    // if periodicity is in flow direction, staggered copy requires neighbour to be one step further away
          }
          // upper boundary colInd
          if (domainBoundaryBool[2*d+1]){
            csrColInd[arrayIndices[2*d+1]] = (row + currentOffset);
          }
          else if(boolPeriodic[d]){
            csrColInd[arrayIndices[2*d+1]] = (row - currentOffset*(dimensions[d]-1-(d==0)));  // if periodicity is in flow direction, staggered copy requires neighbour to be one step further away
          }
        }

        csrMatVal[arrayIndices[2*dimSize]] = 1.;
        csrColInd[arrayIndices[2*dimSize]] = row;
        diagonalArray[row] = 0.;
        continue;
    }

    calcCellFluxesX(fluxes, velocity, dimensions, dimPad, padDepth, cellArea, dimProduct, dimProductPad, dimSize, location);
    if((row==0||row==6279)&&PRINT){
        printf("Xrow %d, flux:  [%f, %f, %f, %f, %f, %f]  \n ", row, fluxes[0], fluxes[1], fluxes[2], fluxes[3], fluxes[4], fluxes[5]);
        printf("Xrow %d, arInd:  [%d, %d, %d, %d, %d, %d, %d]  \n ", row, arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4], arrayIndices[5], arrayIndices[6]);
    }

    diagonalValue = 0;
    int centeredNeighborIdx;
    for (short d=dimSize-1; d>=0; d--){
        currentOffset /= dimensions[d];
        // lower boundary
        centeredNeighborIdx = gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0, d, -1);
        tempBoundaryBool = (active_mask[centeredNeighborIdx]==1.0f) ||
                           (domainBoundaryBool[2*d]&&noSlipWall[centeredNeighborIdx]);
                           //(accessible_mask[gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0, d, -1)]==1.0f&&domainBoundaryBool[2*d]);
        if (tempBoundaryBool){
            csrMatVal[arrayIndices[2*d]] = fluxes[2*d] * .5 +                                                   // non-uniform: .5 implies equally spaced grid (midpoint interpolation),
                                           viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d];  //   needs to change for general bilinear interp.! also: cellArea/gridSpacing needs to stay (this is volume integral)
        }
        if (domainBoundaryBool[2*d]){
            csrColInd[arrayIndices[2*d]] = row - currentOffset;
        }
        else if (boolPeriodic[d]) {
            csrColInd[arrayIndices[2*d]] = row + currentOffset * (dimensions[d]-1-(d==0));
        }
        diagonalValue +=  fluxes[2*d] * (2-tempBoundaryBool)*.5 -
                          viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d] * (tempBoundaryBool + (d!=0) * (1 - tempBoundaryBool) * noSlipWall[centeredNeighborIdx] * 2);
        if((row==0||row==1)&&PRINT){
          printf("Xrow %d d%d  diag  %f    tBB %d    nSW %d\n", row,d, diagonalValue, tempBoundaryBool, noSlipWall[centeredNeighborIdx]);
        }

        //printf("d %d: l: %d", d, tempBoundaryBool);
        // upper boundary
        centeredNeighborIdx = gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0, d, 1-(d==0));
        tempBoundaryBool = (active_mask[centeredNeighborIdx]==1.0f) ||
                           (domainBoundaryBool[2*d+1]&&noSlipWall[centeredNeighborIdx]);
                           //(accessible_mask[gridIDXpaddedCenteredMasks(dimensions, dimSize, location, 0, d, 1-(d==0))]==1.0f&&domainBoundaryBool[2*d+1]);
        if (tempBoundaryBool){
            csrMatVal[arrayIndices[2*d+1]] =  - fluxes[2*d+1] * .5 +
                                              viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d];
        }
        if (domainBoundaryBool[2*d+1]){
            csrColInd[arrayIndices[2*d+1]] = row + currentOffset;
        }
        else if (boolPeriodic[d]) {
            csrColInd[arrayIndices[2*d+1]] = row - currentOffset * (dimensions[d]-1-(d==0));
        }
        diagonalValue +=  - fluxes[2*d+1] * (2-tempBoundaryBool)*.5 -
                          viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d] * (tempBoundaryBool + (d!=0) * (1 - tempBoundaryBool) * noSlipWall[centeredNeighborIdx] * 2);
        if((row==0||row==1)&&PRINT){
          printf("Xrow %d d%d  diag  %f    tBB %d    nSW %d\n", row,d, diagonalValue, tempBoundaryBool, noSlipWall[centeredNeighborIdx]);
        }
        //printf(" h: %d   ",tempBoundaryBool );
    }
    csrMatVal[arrayIndices[2*dimSize]] = diagonalValue - beta[0];   // beta = dx*dx*dz / dt     --> local value in non-uniform meshes
    csrColInd[arrayIndices[2*dimSize]] = row;
    diagonalArray[row] = diagonalValue;
    if((row==0||row==1)&&PRINT){
      printf("Xrow %d   diag  %f\n", row, diagonalValue);
      printf("Xrow %d   diag-b  %f\n", row, diagonalValue - beta[0]);
    }
    //printf("\n");
  }
}


__global__ void calcAdvetionMatrixY(const float* velocity, float* const csrMatVal, int* const csrColInd, int* const csrRowPtr, float* const diagonalArray, const bool* boolDirichletMask, const float* active_mask, const float* accessible_mask,
  const int* dimensions, const int* dimPad, const int* padDepth, const float* cellArea, const float* gridSpacing, const float* viscosity, const bool boolViscosityField,  const int dimProduct, const int* dimProductPad,
  const int dimSize, const bool* noSlipWall, const bool* boolPeriodic, const float* beta){
  for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < dimProduct; row += blockDim.x * gridDim.x){

    //printf("\nstart row %d    ", row);
    bool tempBoundaryBool;
    int arrayIndices[2*3+1];
    int arrayIndicesOrdered[2*3+1];
    float fluxes[2*3];
    float diagonalValue;
    bool domainBoundaryBool[2*3];
    int location[3];

    calcGridLocation(location, &dimensions[dimSize*1], dimProduct,  dimSize, row);
    // GET BOUNDARY INFORMATION OF CURRENT CELL, stored in domainBoundaryBool:  (low_x,high_x, low_y,high_y, low_z,high_z), False when on Boundary
    for (short d = dimSize - 1; d>=0; d--){
      domainBoundaryBool[2*d] = min(location[d],1);
      domainBoundaryBool[2*d+1] = min(dimensions[dimSize+d]-1-location[d],1);                   // THIS LOCATION IN 'dimensions' IS X-SPECIFIC
    }

    // CALCULATE ARRAY INDICES IN ASCENDING ORDER OF APPEARANCE, normally low_z comes first, followed by low_y,low_x , center...
    // arrayIndices has a different ordering in 3D version!!: (low_x,high_x, low_y,high_y, low_z,high_z, center)
    for (short d = 0; d<dimSize; d++){
      arrayIndices[2*(dimSize-1-d)] = csrRowPtr[row]+d;
      arrayIndices[2*(dimSize-1-d)+1] = csrRowPtr[row]+2*dimSize-d;
    }
    arrayIndices[2*dimSize]=csrRowPtr[row]+dimSize;
    // account for periodic boundary conditions,
    // being on a Z-boundary has largest impact, shifting the order of the subranges y,x,center
    // on a periodic Y-boundary, only x,center subrange are canged; and so on....
    for (short d = dimSize - 1; d>=0; d--){
      arrayIndices[2*d] += (d+1)*2*(1-domainBoundaryBool[2*d])*boolPeriodic[d];       // if we are on a lower periodic boundary, this index is the new largest index (in the sub-range)
      arrayIndices[2*d] += (1-domainBoundaryBool[2*d+1])*boolPeriodic[d];
      arrayIndices[2*d+1] -= (d+1)*2*(1-domainBoundaryBool[2*d+1])*boolPeriodic[d];   // if we are on a higher periodic boundary, this index is the new lowest index (in the sub-range)
      arrayIndices[2*d+1] -= (1-domainBoundaryBool[2*d])*boolPeriodic[d];
      for (short s = 0; s<d; s++){
        arrayIndices[2*s]   += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
        arrayIndices[2*s+1] += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
      }
      arrayIndices[2*dimSize] += (domainBoundaryBool[2*d]-domainBoundaryBool[2*d+1])*boolPeriodic[d];
    }

    // acount for non-periodic boundaries: entries for neuman boundaries do not exist -> shift all following entries in a row down by one
    for (short i=0; i<2*dimSize+1; i++){
      arrayIndicesOrdered[arrayIndices[i]-csrRowPtr[row]]=i;
    }
    for (short d=0; d<2*dimSize; d++){
      if (arrayIndicesOrdered[d]==dimSize*2) continue;
      for (short i=d+1; i<2*dimSize+1; i++){
        arrayIndices[arrayIndicesOrdered[i]] -= (1-domainBoundaryBool[arrayIndicesOrdered[d]])*
                                                (1-boolPeriodic[arrayIndicesOrdered[d]/2]);
      }
    }
    int currentOffset = dimProduct;
    // DIRICHLET BOUNDARY CONDITION:                     // THIS dimProduct IS X-SPECIFIC
    if (boolDirichletMask[row] == 1){
        for(short d=dimSize-1; d>=0; d--){
          currentOffset /= dimensions[dimSize+d];                               // THIS LOCATION IN 'dimension' IS X-SPECIFIC
          // lower boundary colInd
          if (domainBoundaryBool[2*d]){
            csrColInd[arrayIndices[2*d]] = (row - currentOffset);
          }
          else if(boolPeriodic[d]){
            csrColInd[arrayIndices[2*d]] = (row + currentOffset*(dimensions[dimSize+d]-1-(d==1)));    // if periodicity is in flow direction, staggered copy requires neighbour to be one step further away
          }
          // upper boundary colInd
          if (domainBoundaryBool[2*d+1]){
            csrColInd[arrayIndices[2*d+1]] = (row + currentOffset);
          }
          else if(boolPeriodic[d]){
            csrColInd[arrayIndices[2*d+1]] = (row - currentOffset*(dimensions[dimSize+d]-1-(d==1)));  // if periodicity is in flow direction, staggered copy requires neighbour to be one step further away
          }
        }

        csrMatVal[arrayIndices[2*dimSize]] = 1.;
        csrColInd[arrayIndices[2*dimSize]] = row;
        diagonalArray[row] = 0.;
        continue;
    }


    calcCellFluxesY(fluxes, velocity, dimensions, dimPad, padDepth, cellArea, dimProduct, dimProductPad, dimSize, location);
    /*if(row==0){
        printf("Yrow %d, flux:  [%f, %f, %f, %f, %f, %f]  \n ", row, fluxes[0], fluxes[1], fluxes[2], fluxes[3], fluxes[4], fluxes[5]);
        printf("Yrow %d, arInd:  [%d, %d, %d, %d, %d, %d]  \n ", row, arrayIndices[0], arrayIndices[1], arrayIndices[2], arrayIndices[3], arrayIndices[4], arrayIndices[5]);
    }*/

    //printf("row %d   fluxes [%f, %f, %f, %f]\n", row, fluxes[0], fluxes[1], fluxes[2], fluxes[3]);
    diagonalValue = 0;
    int centeredNeighborIdx;
    for (short d=dimSize-1; d>=0; d--){
        currentOffset /= dimensions[dimSize+d];
        // lower boundary
        centeredNeighborIdx = gridIDXpaddedCenteredMasks(&dimensions[dimSize], dimSize, location, 1, d, -1);
        tempBoundaryBool = (active_mask[centeredNeighborIdx]==1.0f) ||
                           (domainBoundaryBool[2*d]&&noSlipWall[centeredNeighborIdx]);
                          //(accessible_mask[gridIDXpaddedCenteredMasks(&dimensions[dimSize], dimSize, location, 1, d, -1)]==1.0f&&domainBoundaryBool[2*d]);
        if (tempBoundaryBool){
            csrMatVal[arrayIndices[2*d]] = fluxes[2*d] * .5 +
                                           viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d];
        }
        if (domainBoundaryBool[2*d]){
            csrColInd[arrayIndices[2*d]] = row - currentOffset;
        }
        else if (boolPeriodic[d]) {
            csrColInd[arrayIndices[2*d]] = row + currentOffset * (dimensions[dimSize+d]-1-(d==1));
        }
        diagonalValue +=  fluxes[2*d] * (2-tempBoundaryBool)*.5 -
                          viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d]  * (tempBoundaryBool + (d!=1) * (1 - tempBoundaryBool) * noSlipWall[centeredNeighborIdx] * 2);
        /*if(row==0){
          printf("Yrow %d d%d  diag  %f    tBB %d    nSW %d\n", row,d, diagonalValue, tempBoundaryBool, noSlipWall[centeredNeighborIdx]);
        }*/
        //printf("d %d: l: %d", d, tempBoundaryBool);
        // upper boundary
        centeredNeighborIdx = gridIDXpaddedCenteredMasks(&dimensions[dimSize], dimSize, location, 1, d, 1-(d==1));
        tempBoundaryBool = (active_mask[centeredNeighborIdx]==1.0f) ||
                           (domainBoundaryBool[2*d+1]&&noSlipWall[centeredNeighborIdx]);
                          //(accessible_mask[gridIDXpaddedCenteredMasks(&dimensions[dimSize], dimSize, location, 1, d, 1-(d==1))]==1.0f&&domainBoundaryBool[2*d+1]);
        if (tempBoundaryBool){
            csrMatVal[arrayIndices[2*d+1]] =  - fluxes[2*d+1] * .5 +
                                              viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d];
        }
        if (domainBoundaryBool[2*d+1]){
            csrColInd[arrayIndices[2*d+1]] = row + currentOffset;
        }
        else if (boolPeriodic[d]) {
            csrColInd[arrayIndices[2*d+1]] = row - currentOffset * (dimensions[dimSize+d]-1-(d==1));
        }
        diagonalValue +=  - fluxes[2*d+1] * (2-tempBoundaryBool)*.5 -
                          viscosity[row * boolViscosityField] * cellArea[d] / gridSpacing[d]  * (tempBoundaryBool + (d!=1) * (1 - tempBoundaryBool) * noSlipWall[centeredNeighborIdx] * 2);
        /*if(row==0){
          printf("Yrow %d d%d  diag  %f    tBB %d    nSW %d\n", row,d, diagonalValue, tempBoundaryBool, noSlipWall[centeredNeighborIdx]);
        }*/
        //printf(" h: %d   ",tempBoundaryBool );
    }
    csrMatVal[arrayIndices[2*dimSize]] = diagonalValue - beta[0];
    csrColInd[arrayIndices[2*dimSize]] = row;
    //printf("row %d  matVal[%f, %f, %f, %f, %f]  colInd[%d, %d, %d, %d, %d]\n", row,
            //csrMatVal[arrayIndices[0]], csrMatVal[arrayIndices[1]], csrMatVal[arrayIndices[2]], csrMatVal[arrayIndices[3]], csrMatVal[arrayIndices[4]],
            //csrColInd[arrayIndices[0]], csrColInd[arrayIndices[1]], csrColInd[arrayIndices[2]], csrColInd[arrayIndices[3]], csrColInd[arrayIndices[4]] );
    diagonalArray[row] = diagonalValue;
    /*if(row==0){
      printf("Yrow %d   diag  %f\n", row, diagonalValue);
      printf("Yrow %d   diag-b  %f\n", row, diagonalValue - beta[0]);
    }*/
  }
}


__global__ void initWithZeros(dtype *array, const int nnZ)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nnZ; row += blockDim.x * gridDim.x)
  {
      array[row] = 0.;
  }
}

__global__ void initWithZeros_int(int* intArray, const int nnZ)
{
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nnZ; row += blockDim.x * gridDim.x)
  {
      intArray[row] = 0.;
  }
}

__global__ void calcCsrRowPtrGpu(int* const csrRowPtr, const int* dimensions_specific, const bool* boolPeriodic, const int dimSize, const int dimProduct)
{
  int location[3]={0};
  int rowPtr;
  int currentOffset;
  int currentIndex;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dimProduct; row += blockDim.x * gridDim.x)
  {
      currentOffset = dimProduct;
      currentIndex = row;
      for (short d = dimSize - 1; d>=0; d--){
          currentOffset /= dimensions_specific[d];
          location[d] = currentIndex / currentOffset;
          currentIndex = currentIndex % currentOffset;
      }
      // row Ptr if all cells had all neighbours
      rowPtr = (row+1) * (1 + 2 * dimSize);
      if (dimSize==3){
          rowPtr -= min(location[2],1) * (dimensions_specific[0]*dimensions_specific[1]*(1-boolPeriodic[2]));
          rowPtr -= location[2] * 2 * (dimensions_specific[0]*(1-boolPeriodic[1]) + dimensions_specific[1]*(1-boolPeriodic[0]));
          rowPtr -= ((1-min(location[2],1)) + (1+max(location[2]+1-dimensions_specific[2],-1))) *
                    ((location[0]+1)+dimensions_specific[0]*(location[1]))*(1-boolPeriodic[2]);
      }

      rowPtr -= min(location[1],1) * (dimensions_specific[0]*(1-boolPeriodic[1]));
      //rowPtr -= max(location[1]-1,0) * 2 * (1-boolPeriodic[0]);
      rowPtr -= ((1-min(location[1],1)) + (1+max(location[1]+1-dimensions_specific[1],-1))) *
                (location[0]+1)*(1-boolPeriodic[1]);

      rowPtr -= (location[1]*2 + 1 + (1+max(location[0]+1-dimensions_specific[0],-1))) * (1-boolPeriodic[0]);

      csrRowPtr[row+1]=rowPtr;
  }
}


__global__ void calcDimPad(int* dimPad, int* dimProductPad, const int* dimensions, const int* padDepth, const int dimSize){
  for (int velComp = blockIdx.x * blockDim.x + threadIdx.x; velComp < dimSize; velComp += blockDim.x * gridDim.x){
  //for (size_t velComp = 0; velComp < dimSize; velComp++) {
      dimPad[dimSize*velComp] = 1;
      dimProductPad[velComp] = (padDepth[0]+padDepth[1]+dimensions[dimSize*velComp]);
      for (size_t d = 1; d < dimSize; d++){
        dimPad[dimSize*velComp + d] = dimPad[dimSize*velComp + d - 1] * (padDepth[2*(d-1)] + padDepth[2*(d-1)+1] + dimensions[velComp*dimSize+(d-1)]);
        dimProductPad[velComp] *= (padDepth[d*2]+padDepth[d*2+1]+dimensions[velComp*dimSize+d]);
      }
  }
}


__global__ void printarray(const float* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %f \n ", a[i]);
  }
  printf("\n");
}


__global__ void printarray_int(const int* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %d \n ", a[i]);
  }
  printf("\n");
}

__global__ void printarray_bool(const bool* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %d \n ", a[i]);
  }
  printf("\n");
}
// VERSION UTILISING GPU ROW POINTERS
__host__ void CentralDifferenceMatrixCsrKernelLauncher(const float* velocity, float* const csrMatVal, int* const csrColInd, int* const csrRowPtr, float* const diagonalArray,
  const bool* boolDirichletMask, const float* active_mask, const float* accessible_mask, const float* viscosity, const int* dimensions, const int* padDepth, const float* cellArea, const float* gridSpacing, const bool boolViscosityField, const int dimSize,
  const bool* noSlipWall, const bool* boolPeriodic, const float* beta,const int unrolling_step){

  // STEP 0: PREPARATION DONE BEFORE MAIN STEPS -------------------------------------------------------------------------
  // optimise performance by splitting operation in streams
  const int nStreams        = dimSize*2+1;
  cudaStream_t streams[nStreams];
  cudaEvent_t  events[nStreams];
  int minGridSizes[nStreams] = {0};
  int gridSizes[nStreams]    = {0};
  int blockSizes[nStreams]   = {0};

  // gets domain dimension
  int hostDimensions[dimSize*dimSize];
  cudaMemcpy(hostDimensions, dimensions, dimSize*dimSize*sizeof(int), cudaMemcpyDeviceToHost);

  bool hostBoolPeriodic[dimSize];
  cudaMemcpy(hostBoolPeriodic, boolPeriodic, dimSize*sizeof(bool), cudaMemcpyDeviceToHost);

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  // TODO: combine for loops
  // domain shape products
  int hostDimProduct[dimSize];
  for (size_t i = 0; i < dimSize; i++) {
    int temp = hostDimensions[i*dimSize];
    for (size_t j = 1; j < dimSize; j++) {
        temp *= hostDimensions[i*dimSize+j];
    }
    hostDimProduct[i]= temp;
  }

  //printarray_bool<<<1,1>>>(noSlipWall, hostDimProduct[0] );
  // number of non-zeros in the advection matrices of x and y
  int nnZ[dimSize];
  for (size_t i = 0; i < dimSize; i++) {
    nnZ[i] = hostDimProduct[i]*(dimSize*2+1);
    for (size_t j = 0; j < dimSize; j++) {
      nnZ[i] -= 2*hostDimProduct[i]/hostDimensions[i*dimSize+j]*(1-hostBoolPeriodic[j]);
      //printf("nnz: %d \n",nnZ[i]);
    }
  }

  // sum over all nnZs
  int nnZsum = 0;
  for (size_t i = 0; i < dimSize; i++) {
    nnZsum += nnZ[i];
  }
  //printf("nnz sum %d\n", nnZsum);

  // offsets to start of x and y parts in rowPtr and boundaryBool arrays
  int rowPtrOffset[3]        = { 0, hostDimProduct[0] + 1, hostDimProduct[0]+hostDimProduct[1]+2 };
  //int boundaryBoolOffset[3]  = { 0, 2 * dimSize * hostDimProduct[0], 2 * dimSize * hostDimProduct[0] + 2 * dimSize * hostDimProduct[1] };

  // Initialize Streams and Events
  for (size_t i = 0; i < nStreams; i++) {
    CUDA_CHECK_RETURN( cudaStreamCreate(&streams[i]) );
    CUDA_CHECK_RETURN( cudaEventCreate(&events[i]) );
  }

  // calculate dimPad, containing offsets that help calculate the index in a padded vel array for unpadded location
  int *dimPad;
  int *dimProductPad;
  CUDA_CHECK_RETURN(cudaMalloc((void **) &dimPad, dimSize*dimSize*sizeof(int)));
  CUDA_CHECK_RETURN(cudaMalloc((void **) &dimProductPad, dimSize*sizeof(int)));
  calcDimPad<<<1,dimSize, 0 , streams[nStreams-1]>>>(dimPad, dimProductPad,dimensions, padDepth, dimSize);

  // GPU csrRowPtr for x and y on streams 3 and 4
  cudaOccupancyMaxPotentialBlockSize(&minGridSizes[dimSize], &blockSizes[dimSize], calcCsrRowPtrGpu, 0, 0);
  for (size_t d = 0; d < dimSize; d++) {
    gridSizes[dimSize+d] = (hostDimProduct[d] + blockSizes[dimSize]-1)/blockSizes[dimSize];
  }
  int helper =0;
  for (size_t d = 0; d < dimSize; d++) {
    calcCsrRowPtrGpu<<<gridSizes[dimSize+d], blockSizes[dimSize], 0, streams[dimSize+d]>>>(&csrRowPtr[rowPtrOffset[d]], &dimensions[dimSize*d], boolPeriodic, dimSize, hostDimProduct[d]);
    cudaMemcpyAsync(&csrRowPtr[rowPtrOffset[d]], &helper, sizeof(int), cudaMemcpyHostToDevice, streams[dimSize+d]);
  }


  // Launch initialisation of csrValues and csrColInd on 2 streams
  cudaOccupancyMaxPotentialBlockSize(&minGridSizes[0], &blockSizes[0], initWithZeros, 0, 0);
  cudaOccupancyMaxPotentialBlockSize(&minGridSizes[1], &blockSizes[1], initWithZeros_int, 0, 0);
  gridSizes[0] = (nnZsum + blockSizes[0] - 1) / blockSizes[0];
  gridSizes[1] = (nnZsum + blockSizes[1] - 1) / blockSizes[1];
  initWithZeros<<<gridSizes[0], blockSizes[0], 0, streams[0]>>>(csrMatVal, nnZsum);
  initWithZeros_int<<<gridSizes[1], blockSizes[1], 0, streams[1]>>>(csrColInd, nnZsum);


  // STEP 2: CALCULATE X AND Y ADVECTION MATRICES ON DIFFERENT STREAMS

  cudaOccupancyMaxPotentialBlockSize(&minGridSizes[0], &blockSizes[0], calcAdvetionMatrixX, 0, 0);
  cudaOccupancyMaxPotentialBlockSize(&minGridSizes[1], &blockSizes[1], calcAdvetionMatrixY, 0, 0);
  gridSizes[0] = (hostDimProduct[0] + blockSizes[0] - 1) / blockSizes[0];
  gridSizes[1] = (hostDimProduct[1] + blockSizes[1] - 1) / blockSizes[1];

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  calcAdvetionMatrixX<<<gridSizes[0], blockSizes[0], 0, streams[0]>>>( //,
  //calcAdvetionMatrixX<<<1,1, 0, streams[0]>>>(
    velocity, csrMatVal, csrColInd, csrRowPtr, diagonalArray,
    boolDirichletMask, active_mask, accessible_mask,
    dimensions,
    dimPad, padDepth, cellArea, gridSpacing,
    viscosity, boolViscosityField, hostDimProduct[0], dimProductPad,
    dimSize, noSlipWall, boolPeriodic, beta);

  calcAdvetionMatrixY<<<gridSizes[1], blockSizes[1], 0, streams[1]>>>( //,
  //calcAdvetionMatrixY<<<1,1, 0, streams[1]>>>( //,
    velocity, &csrMatVal[nnZ[0]], &csrColInd[nnZ[0]], &csrRowPtr[rowPtrOffset[1]], &diagonalArray[hostDimProduct[0]],
    &boolDirichletMask[hostDimProduct[0]], active_mask, accessible_mask,
    dimensions,
    dimPad, padDepth, cellArea, gridSpacing,
    &viscosity[boolViscosityField*hostDimProduct[0]], boolViscosityField, hostDimProduct[1], dimProductPad,
    dimSize, noSlipWall, boolPeriodic, beta);
    //dimSize, &noSlipWall[hostDimProduct[0]], boolPeriodic, beta);

  // STEP 3: CLEAN UP
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  for (short i = 0; i < nStreams; i++) {
    CUDA_CHECK_RETURN( cudaStreamDestroy( streams[i] ));
    CUDA_CHECK_RETURN( cudaEventDestroy( events[i] ));
  }
}


#include <cuda_runtime.h>
#include <iostream>


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;

  exit(1);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

// Converts coordinates of the simulation grid to indices of the extended mask grid with a shift to get the indices of the neighbors
//  mask_idx_before = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);
__device__ int gridIDXWithOffsetShifted(const int *dimensions, const int dim_size, const int *cords, int cords_offset, int dim_index_offset, int offset)
{
    int factor = 1;
    int result = 0;
    for (int i = 0; i < dim_size; i++)
    {
        if (i == dim_index_offset)
            result += factor * (cords[i + cords_offset * dim_size] + offset);
        else
            result += factor * (cords[i + cords_offset * dim_size] + 1);

        factor *= dimensions[i];
    }
    return result;
}

// Calculates the array position of a staggered-grid neighbour of the centered grid
// staggered_dimensions: [(x,y,z)_x, (x,y,z)_y, (x,y,z)_z]
// Assumes following datastructure: staggered-array = [(cen_array)_z, (cen_array)_y, (cen_array)_x]
__device__ int gridIDXForStaggered(const int *dimensions, const int *staggered_dimensions, const int dim_size, const int dim_product,
  const int *cords, int cords_offset, int dim_index_offset, int offset)
{
    //int k =0;
    int factor = 1;
    int result = 0;
    for (int i = dim_size - 1; i >= 0; i--)
    {
        if (i == dim_index_offset)
        {
            for (int j = 0; j < dim_size; j++)
            {
                if (j == dim_index_offset)
                    result += factor * (cords[j + cords_offset * dim_size] + offset);

                else
                    result += factor * (cords[j + cords_offset * dim_size] );

                factor *= staggered_dimensions[j + i * dim_size];
            }
            break;
        }
        else
            result += dim_product / dimensions[i] * (dimensions[i]+1);
    }
    return result;
}

// TODO: wtf going on in CordsByRow?
__device__ void CordsByRow(int row, const int *dimensions, const int dim_size, const int dim_product, int *cords)
{
    int modulo = 0;
    int divisor = dim_product;

    for (int i = dim_size - 1; i >= 0; i--)
    {
        divisor = divisor / dimensions[i];

        cords[i + row * dim_size] = (modulo == 0 ? row : (row % modulo)) / divisor; // 0 mod 0 not possible due to c++ restrictions
        modulo = divisor;
    }
}

template <typename T>
__global__ void calcPISOLaplaceMatrix(const int *dimensions, const int dim_size, const int dim_product, const float *active_mask,
const float *fluid_mask, const int *mask_dimensions, T *laplace_matrix, int *cords,
 const float *advection_influence, const int *staggered_dimensions)
{// Laplacian for Pressure-Increment in PISO (Dirichlet BC for pressure are always zero!)
// dimensions: array containing size of each dimensions
// dim_size: how many dimensions exist
// dim_product; product of dimensions (e.g. size of flattened centered grid)
// mask_dimsensions: array conatinain size of mask in each dimsension (= dimensions +2 for each dim)

// TODO: laplace_matrix as signed char... (why? just because its smaller?) need to replace this with float due to A_0#
// TODO: trace laplace_matrix changeing back to its first appearance (as signed char and change to float)
// TODO: maybe reuse diagonal tensor of advection matrix for this, is kind of flat anyways...
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x)
    { // TODO: reduce this by half since the matrix is symmetrical and only the half needs to be created? => Already pretty fast
        // Derive the coordinates of the dim_size-Dimensional mask by the laplace row id
        CordsByRow(row, dimensions, dim_size, dim_product, cords);

        // Every thread accesses the laplaceDataBuffer at different areas. index_pointer points to the current position of the current thread
        int index_pointer = row * (dim_size * 2 + 1);

        // forward declaration of variables, that are reused
        int mask_idx = 0;
        int mask_idx_before = 0;
        int mask_idx_after = 0;

        int advection_index_before = 0;
        int advection_index_after = 0;

		// dim_size-Dimensional "Cubes" have exactly dim_size * 2  "neighbors", value to be put on the diagonal
		// TODO: In PISO this is the sum of neighouring A_0, other solution: set to zero and deduct neighbouring influence
		T diagonal = 0.0f;

        // get the index on the extended mask grid of the current cell
        // TODO: use similar approach for advection_pointer on staggered grid with mask_dimensions as staggered_dimensions= dimensions +1
		int rowMaskIdx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, 0, 1);

        // Check neighbors if they are solids. For every solid neighbor increment diagonal by one
        // TODO: this is where PISO is different, solid here encodes Neumann BC (sure about that?), where both the off-diagonal and diagonal are removed
		for (int j = dim_size - 1; j >= 0; j--)
        {
            // get the index on the extended mask grid of the neighbor cells
            mask_idx_before = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);
            mask_idx_after = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 2);

            //if(active_mask[mask_idx_before] == 1.0f || pressure_dirichlet_mask[mask_idx_before] == 0.0f){
            if (!(active_mask[mask_idx_before] == 0.0f && fluid_mask[mask_idx_before] == 0.0f)&& active_mask[rowMaskIdx]!= 0.0f){
                advection_index_before = gridIDXForStaggered(dimensions, staggered_dimensions, dim_size, dim_product, cords, row, j, 0);
                diagonal -= advection_influence[advection_index_before];
            }

            //if(active_mask[mask_idx_after] == 1.0f || pressure_dirichlet_mask[mask_idx_after] == 0.0f){
            if(!(active_mask[mask_idx_after] == 0.0f && fluid_mask[mask_idx_after] == 0.0f)&& active_mask[rowMaskIdx]!= 0.0f){
                advection_index_after = gridIDXForStaggered(dimensions, staggered_dimensions, dim_size, dim_product, cords, row, j, 1);
                diagonal -= advection_influence[advection_index_after];
            }
        }

        // Check the "left"/"before" neighbors if they are fluid and add them to the laplaceData
        // TODO: treats neumann & dirichlet collectively, in PISO: dirichlet BC demands p_inc_BC=0 (as long as pressure BC do not change over time)
        // TODO: --> no need for entry in Laplacian
        for (int j = dim_size - 1; j >= 0; j--)
        {
            mask_idx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);

            //if (active_mask[mask_idx] == 1)// && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            if (active_mask[mask_idx] == 1 && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            { // fluid - fluid
                // TODO: get connecting A_0 value for respective neighbour
                advection_index_before = gridIDXForStaggered(dimensions, staggered_dimensions, dim_size, dim_product, cords, row, j, 0);
				        laplace_matrix[index_pointer] = advection_influence[advection_index_before];
            }
            //else if (active_mask[mask_idx] == 0 && fluid_mask[mask_idx] == 1)
            //{ // Empty / open cell
                // pass, because we initialized the data with zeros
            //}
            index_pointer++;
        }

        // Add the diagonal value
		    laplace_matrix[index_pointer] = diagonal;
        index_pointer++;

        // Finally add the "right"/"after" neighbors
        // TODO: same as for lower neighbours
        for (int j = 0; j < dim_size; j++)
        {
            mask_idx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 2);

            //if (active_mask[mask_idx] == 1)// && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            if (active_mask[mask_idx] == 1 && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            { // fluid - fluid
                //TODO: Add A_0 here
                advection_index_after = gridIDXForStaggered(dimensions, staggered_dimensions, dim_size, dim_product, cords, row, j, 1);
                laplace_matrix[index_pointer] = advection_influence[advection_index_after];
            }

            index_pointer++;
        }
    }
}

template <typename T>
__global__ void setUpData( const int dim_size, const int dim_product, T *laplace_matrix) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < dim_product * (dim_size * 2 + 1);
         row += blockDim.x * gridDim.x)
    {
        laplace_matrix[row] = 0;
    }
}

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dim_size, const int dim_product,  const float *active_mask,
const float *fluid_mask, const int *mask_dimensions, float *laplace_matrix, int *cords, const float *advection_influence,
const int *staggered_dimensions) {
    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              setUpData<float>, 0, 0);
    gridSize = (dim_product * (dim_size * 2 + 1) + blockSize - 1) / blockSize;

    // Init Laplace Matrix with zeros
    setUpData<<<gridSize, blockSize>>>(dim_size, dim_product, laplace_matrix);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcPISOLaplaceMatrix<float>, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Calculate the Laplace Matrix
    calcPISOLaplaceMatrix<<<gridSize, blockSize>>>(dimensions, dim_size, dim_product, active_mask, fluid_mask, mask_dimensions, laplace_matrix, cords, advection_influence, staggered_dimensions);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dim_size, const int dim_product,  const float *active_mask,
const float *fluid_mask, const int *mask_dimensions, double *laplace_matrix, int *cords, const float *advection_influence,
const int *staggered_dimensions) {
    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              setUpData<double>, 0, 0);
    gridSize = (dim_product * (dim_size * 2 + 1) + blockSize - 1) / blockSize;

    // Init Laplace Matrix with zeros
    setUpData<<<gridSize, blockSize>>>(dim_size, dim_product, laplace_matrix);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcPISOLaplaceMatrix<double>, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Calculate the Laplace Matrix
    calcPISOLaplaceMatrix<<<gridSize, blockSize>>>(dimensions, dim_size, dim_product, active_mask, fluid_mask, mask_dimensions, laplace_matrix, cords, advection_influence, staggered_dimensions);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
/*
template <typename T>
void LaplaceMatrixKernelLauncher(const int *dimensions, const int dim_size, const int dim_product,  const float *active_mask,
const float *fluid_mask, const int *mask_dimensions, T *laplace_matrix, int *cords, const float *advection_influence,
const int *staggered_dimensions) {
    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              setUpData<T>, 0, 0);
    gridSize = (dim_product * (dim_size * 2 + 1) + blockSize - 1) / blockSize;

    // Init Laplace Matrix with zeros
    setUpData<<<gridSize, blockSize>>>(dim_size, dim_product, laplace_matrix);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcPISOLaplaceMatrix<T>, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Calculate the Laplace Matrix
    calcPISOLaplaceMatrix<<<gridSize, blockSize>>>(dimensions, dim_size, dim_product, active_mask, fluid_mask, mask_dimensions, laplace_matrix, cords, advection_influence, staggered_dimensions);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}*/

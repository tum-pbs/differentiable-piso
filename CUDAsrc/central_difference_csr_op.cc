
#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("CentralDifferenceMatrixCsr")
    .Input("velocity: float32")
    .Input("input_csr_mat_val: float32")
    .Input("input_csr_col_ind: int32")
    .Input("input_csr_row_ptr: int32")
    .Input("input_diagonal_array: float32")
    .Input("input_bool_dirichlet_mask: bool")
    .Input("input_active_mask: float32")
    .Input("input_accessible_mask: float32")
    .Input("input_viscosity: float32")
    .Input("input_dimensions: int32")
    .Input("input_pad_depth: int32")
    .Input("input_cell_area: float32")
    .Input("input_grid_spacing: float32")
    .Input("input_no_slip_walls: bool")
    .Input("input_bool_periodic: bool")
    .Input("beta: float32")
    .Attr("unrolling_step: int")
    .Output("csr_values: float32")
    .Output("csr_col_ind: int32")
    .Output("csr_row_ptr: int32")
    .Output("diagonal_array: float32");


void CentralDifferenceMatrixCsrKernelLauncher(const float* velocity, float* const csrMatVal, int* const csrColInd,int* const csrRowPtr,
  float* const diagonalArray, const bool* boolDirichletMask, const float* active_mask, const float* accessible_mask, const float* viscosity, const int* dimensions,
  const int* padDepth, const float* cellArea, const float* gridSpacing, const bool boolViscosityField, const int dimSize, const bool* noSlipWall,
  const bool* boolPeriodic, const float* beta, const int unrolling_step);

class CentralDifferenceMatrixCsrOp : public OpKernel
{
  private:
    //float beta;
    int unrolling_step;

  public:
    explicit CentralDifferenceMatrixCsrOp(OpKernelConstruction *context) : OpKernel(context) {
      //context->GetAttr("beta",&beta);
      context->GetAttr("unrolling_step", &unrolling_step);
    }

  void Compute(OpKernelContext *context) override
  {
    const Tensor &inputVelocity = context->input(0);
    Tensor input_csr_mat_val = context->input(1);
    Tensor input_csr_col_ind = context->input(2);
    Tensor input_csr_row_ptr = context->input(3);
    Tensor input_diagonal_array = context->input(4);
    const Tensor &input_bool_dirichlet_mask = context->input(5);
    const Tensor &input_active_mask = context->input(6);
    const Tensor &input_accessible_mask = context->input(7);
    const Tensor &input_viscosity = context->input(8);
    const Tensor &input_dimensions = context->input(9);
    const Tensor &input_pad_depth = context->input(10);
    const Tensor &input_cell_area = context->input(11);
    const Tensor &input_grid_spacing = context->input(12);
    const Tensor &input_no_slip_walls = context->input(13);
    const Tensor &input_bool_periodic = context->input(14);

    const Tensor &input_beta = context->input(15);

    auto velocity = inputVelocity.flat<float>();
    auto csrMatVal = input_csr_mat_val.flat<float>();
    auto csrColInd = input_csr_col_ind.flat<int32>();
    auto csrRowPtr = input_csr_row_ptr.flat<int32>();
    auto diagonalArray = input_diagonal_array.flat<float>();
    auto boolDirichletMask = input_bool_dirichlet_mask.flat<bool>();
    auto activeMask = input_active_mask.flat<float>();
    auto accessibleMask = input_accessible_mask.flat<float>();
    auto viscosity = input_viscosity.flat<float>();
    auto dimensions = input_dimensions.flat<int32>();
    auto padDepth = input_pad_depth.flat<int32>();
    auto cellArea = input_cell_area.flat<float>();
    auto gridSpacing = input_grid_spacing.flat<float>();
    auto noSlipWall = input_no_slip_walls.flat<bool>();
    auto boolPeriodic = input_bool_periodic.flat<bool>();

    auto beta = input_beta.flat<float>();

    int dimSize = int(sqrt(dimensions.size()));

    context -> set_output(0, input_csr_mat_val);
    context -> set_output(1, input_csr_col_ind);
    context -> set_output(2, input_csr_row_ptr);
    context -> set_output(3, input_diagonal_array);

    bool boolViscosityField;
    if (input_viscosity.shape().dim_size(0) > 1)
      boolViscosityField = true;
    else boolViscosityField = false;

    CentralDifferenceMatrixCsrKernelLauncher(velocity.data(), csrMatVal.data(), csrColInd.data(), csrRowPtr.data(), diagonalArray.data(),
    boolDirichletMask.data(), activeMask.data(), accessibleMask.data(), viscosity.data(), dimensions.data(), padDepth.data(),
    cellArea.data(), gridSpacing.data(), boolViscosityField, dimSize, noSlipWall.data(), boolPeriodic.data(), beta.data(), unrolling_step);

  }
};

REGISTER_KERNEL_BUILDER(Name("CentralDifferenceMatrixCsr").Device(DEVICE_GPU), CentralDifferenceMatrixCsrOp);


#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BicgstabIluLinearSolve")
    .Input("csr_values: T")
    .Input("csr_row_ptr: int32")
    .Input("csr_col_ind: int32")
    .Input("rhs: T")
    .Input("input_x: T")
    .Input("s: T")
    .Input("s_hat: T")
    .Input("p: T")
    .Input("p_hat: T")
    .Input("r: T")
    .Input("r_hat: T")
    .Input("v: T")
    .Input("t: T")
    .Input("z: T")
    //.Input("csr_values_ilu: float32")
    .Input("x_buffer: T")

    .Attr("T: {float, double} = DT_FLOAT")
    .Attr("nnz_a: int")
    .Attr("batch_size: int")
    .Attr("matrix_shape: int")
    .Attr("tol: float")
    .Attr("max_it: int")
    .Attr("transpose_op: bool")
    .Output("out_csr_values: T")
    .Output("out_csr_row_ptr: int32")
    .Output("out_csr_col_ind: int32")
    .Output("x: T");

void BicgstabIluLinearSolveLauncher( float * csr_values, int* csr_row_ptr, int* csr_col_ind,
    const float* rhs, const int nnz_a, const int batch_size, const int matrix_shape,const  float *x, float tol, int max_it,
  float *s, float *s_hat,  float *p, float *p_hat, float *r, float *rh, float *v, float *t, float *z,  float * x_copy, //float * csr_values_ilu,
  const bool transpose_op);

void BicgstabIluLinearSolveLauncher( double * csr_values, int* csr_row_ptr, int* csr_col_ind,
    const double* rhs, const int nnz_a, const int batch_size, const int matrix_shape,const  double *x, float tol, int max_it,
  double *s, double *s_hat,  double *p, double *p_hat, double *r, double *rh, double *v, double *t, double *z,  double * x_copy, //float * csr_values_ilu,
  const bool transpose_op);

template <typename T>
class BicgstabIluLinearSolveOp : public OpKernel
{
  private:
    int nnz_a;
    int batch_size;
    int matrix_shape;
    float tol;
    int max_it;
    bool transpose_op;

  public:
    explicit BicgstabIluLinearSolveOp(OpKernelConstruction *context) : OpKernel(context) {
      context->GetAttr("nnz_a", &nnz_a);
      context->GetAttr("batch_size", &batch_size);
      context->GetAttr("matrix_shape", &matrix_shape);
      context->GetAttr("tol", &tol);
      context->GetAttr("max_it", &max_it);
      context->GetAttr("transpose_op", &transpose_op);
    }


  void Compute(OpKernelContext *context) override
  {

    //printf("LINEAR SOLVE size %d   transposeOP %s\n",matrix_shape, transpose_op ? "true":"false");
    //printf("csr_val\n" );
    Tensor input_csr_values = context->input(0);
    //printf("csr_rp\n" );
    Tensor input_csr_row_ptr = context->input(1);
    //printf("csr_ind\n" );
    Tensor input_csr_col_ind = context->input(2);
    //printf("rhs\n" );
    const Tensor &input_rhs = context->input(3);
    //printf("X\n" );
    const Tensor &input_x =  context->input(4);
    //printf("input read\n");
    Tensor s_in = context->input(5);
    Tensor s_hat_in = context->input(6);
    Tensor p_in = context->input(7);
    Tensor p_hat_in = context->input(8);
    Tensor r_in = context->input(9);
    Tensor rh_in = context->input(10);
    Tensor v_in = context->input(11);
    Tensor t_in = context->input(12);
    Tensor z_in = context->input(13);
    //Tensor csr_values_ilu_in = context->input(14);
    Tensor x_buffer_in = context->input(14);

    auto csr_values = input_csr_values.flat<T>();
    auto csr_row_ptr = input_csr_row_ptr.flat<int32>();
    auto csr_col_ind = input_csr_col_ind.flat<int32>();
    auto rhs = input_rhs.flat<T>();
    auto x = input_x.flat<T>();

    auto s = s_in.flat<T>();
    auto s_hat = s_hat_in.flat<T>();
    auto p = p_in.flat<T>();
    auto p_hat = p_hat_in.flat<T>();
    auto r = r_in.flat<T>();
    auto rh = rh_in.flat<T>();
    auto v = v_in.flat<T>();
    auto t = t_in.flat<T>();
    auto z = z_in.flat<T>();
    //auto csr_values_ilu = csr_values_ilu_in.flat<float>();
    auto x_buffer = x_buffer_in.flat<T>();

    // SPACE FOR COPY OF X (actual copy done in cuda) - TF PYTHON REUSES  SOME TENSORS (WIERD)
    /*Tensor* x_copy;
    TensorShape x_copy_shape;
    x_copy_shape.AddDim(input_x.shape().dim_size(0));
    OP_REQUIRES_OK(context, context->allocate_output(3, x_copy_shape, &x_copy));*/

    context->set_output(0, input_csr_values);
    context->set_output(1, input_csr_row_ptr);
    context->set_output(2, input_csr_col_ind);
    context->set_output(3, x_buffer_in);

    //std::cout << typeid(cords_flat).name();
    //std::cout << "launch cuda";
    BicgstabIluLinearSolveLauncher(csr_values.data(), csr_row_ptr.data(), csr_col_ind.data(),
      rhs.data(), nnz_a, batch_size, matrix_shape, x.data(), tol, max_it,
      s.data(),  s_hat.data(),  p.data(), p_hat.data(), r.data(),  rh.data(), v.data(),  t.data(),  z.data(),
      x_buffer.data(), transpose_op); //  csr_values_ilu.data(),
  }
};

REGISTER_KERNEL_BUILDER(Name("BicgstabIluLinearSolve").Device(DEVICE_GPU).TypeConstraint<float>("T"), BicgstabIluLinearSolveOp<float>);
REGISTER_KERNEL_BUILDER(Name("BicgstabIluLinearSolve").Device(DEVICE_GPU).TypeConstraint<double>("T"), BicgstabIluLinearSolveOp<double>);


#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("MultiBicgstabIluLinearSolve")
    .Attr("T: {float, double} = DT_FLOAT")
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
    .Input("warning: bool")
    .Input("matrix_shape: int32")
    .Input("accuracy: float32")
    .Attr("batch_size: int")
    //.Attr("matrix_shape_X: int")
    //.Attr("matrix_shape_Y: int")
    .Attr("max_it: int")
    .Attr("transpose_op: bool")
    .Attr("unrolling_step: int")

    .Output("out_csr_values: T")
    .Output("out_csr_row_ptr: int32")
    .Output("out_csr_col_ind: int32")
    .Output("x: T")
    .Output("out_warning: bool");


/*template <typename T>
void MultiBicgstabIluLinearSolveLauncher( T * csr_values, int* csr_row_ptr, int* csr_col_ind,
    const T* rhs, const int batch_size, const int matrix_shape_X, const int matrix_shape_Y, const  T *x, T tol, int max_it,
  T *s, T *s_hat,  T *p, T *p_hat, T *r, T *rh, T *v, T *t, T *z,  T * x_copy, //float * csr_values_ilu,
  const bool transpose_op, const int unrolling_step, bool* warning);*/

void MultiBicgstabIluLinearSolveLauncher( float * csr_values, int* csr_row_ptr, int* csr_col_ind,
    const float* rhs, const int batch_size, const int* matrix_shape, const int dim_size, const  float *x, float* tol, int max_it,
  float *s, float *s_hat,  float *p, float *p_hat, float *r, float *rh, float *v, float *t, float *z,  float * x_copy, //float * csr_values_ilu,
  const bool transpose_op, const int unrolling_step, bool* warning);

void MultiBicgstabIluLinearSolveLauncher( double * csr_values, int* csr_row_ptr, int* csr_col_ind,
    const double* rhs, const int batch_size, const int* matrix_shape, const int dim_size, const  double *x, float* tol, int max_it,
  double *s, double *s_hat,  double *p, double *p_hat, double *r, double *rh, double *v, double *t, double *z,  double * x_copy, //float * csr_values_ilu,
  const bool transpose_op, const int unrolling_step, bool* warning);

template <typename T>
class MultiBicgstabIluLinearSolveOp : public OpKernel
{
  private:
    int batch_size;
    //int matrix_shape_X;
    //int matrix_shape_Y;
  
    int max_it;
    bool transpose_op;
    int unrolling_step;

  public:
    explicit MultiBicgstabIluLinearSolveOp(OpKernelConstruction *context) : OpKernel(context) {
      context->GetAttr("batch_size", &batch_size);
      //context->GetAttr("matrix_shape_X", &matrix_shape_X);
      //context->GetAttr("matrix_shape_Y", &matrix_shape_Y);

      context->GetAttr("max_it", &max_it);
      context->GetAttr("transpose_op", &transpose_op);
      context->GetAttr("unrolling_step", &unrolling_step);
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
    Tensor warning_in = context->input(15);
    Tensor matrix_shape_in = context->input(16);
    Tensor accuracy_in = context->input(17);

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

    auto x_buffer = x_buffer_in.flat<T>();
    auto warning = warning_in.flat<bool>();
    auto matrix_shape = matrix_shape_in.flat<int32>();
    auto accuracy = accuracy_in.flat<float>();

    int dim_size = matrix_shape.size();

    context->set_output(0, input_csr_values);
    context->set_output(1, input_csr_row_ptr);
    context->set_output(2, input_csr_col_ind);
    context->set_output(3, x_buffer_in);
    context->set_output(4, warning_in);

    //std::cout << typeid(cords_flat).name();
    //std::cout << "launch cuda";
    MultiBicgstabIluLinearSolveLauncher(csr_values.data(), csr_row_ptr.data(), csr_col_ind.data(),
      rhs.data(), batch_size, matrix_shape.data(), dim_size, x.data(), accuracy.data(), max_it,
      s.data(),  s_hat.data(),  p.data(), p_hat.data(), r.data(),  rh.data(), v.data(),  t.data(),  z.data(),
      x_buffer.data(), transpose_op, unrolling_step, warning.data()); //  csr_values_ilu.data(),
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiBicgstabIluLinearSolve").Device(DEVICE_GPU).TypeConstraint<float>("T"), MultiBicgstabIluLinearSolveOp<float>);
REGISTER_KERNEL_BUILDER(Name("MultiBicgstabIluLinearSolve").Device(DEVICE_GPU).TypeConstraint<double>("T"), MultiBicgstabIluLinearSolveOp<double>);

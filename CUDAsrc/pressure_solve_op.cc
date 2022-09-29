#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <sys/time.h>

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("PressureSolveOp")
    .Attr("T: {float, double} = DT_FLOAT")
    .Input("dimensions: int32")
    .Input("mask_dimensions: int32")
    .Input("active_mask: float32")
    .Input("fluid_mask: float32")
    .Input("laplace_matrix: T")

    .Input("divergence: T")
    .Input("p: T")
    .Input("r: T")
    .Input("z: T")
    .Input("pressure: T")

    .Input("advection_influence: float32")
    .Input("staggered_dimensions: int32")
    .Input("input_laplace_rank_deficient: bool")

    //.Input("dim_product: int32")
    .Input("accuracy: float32")
    .Input("max_iterations: int32")
    .Input("bool_periodic: bool")

    .Attr("init_with_zeros: bool")
    .Attr("residual_reset_steps: int")
    .Attr("randomized_restarts: int")
    .Attr("unrolling_step: int")
    /*.Attr("dim_product: int")
    .Attr("accuracy: float")
    .Attr("max_iterations: int")*/

    .Output("pressure_out: T")
    .Output("iterations: int32")
    .Output("lapalce_matrix: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(5)); // divergence
        return Status::OK();
    });


void LaunchPressureKernel(const int *dimensions, const int dim_product, const int dimSize,
            const double* laplace_matrix,
            double* p, double* z, double* r, double *divergence, double* x,
            bool* thresholdReached,
            const float* accuracy,
            const int* max_iterations,
            const int batch_size,
            int* iterations_gpu,
            const bool* boolPeriodic,
            const bool* laplace_rank_deficient,
            const bool init_with_zeros,
            const int residual_reset_steps,
            const int randomized_restarts,
            const int unrolling_step);

void LaunchPressureKernel(const int *dimensions, const int dim_product, const int dimSize,
            const float* laplace_matrix,
            float* p, float* z, float* r, float *divergence, float* x,
            bool* thresholdReached,
            const float* accuracy,
            const int* max_iterations,
            const int batch_size,
            int* iterations_gpu,
            const bool* boolPeriodic,
            const bool* laplace_rank_deficient,
            const bool init_with_zeros,
            const int residual_reset_steps,
            const int randomized_restarts,
            const int unrolling_step);

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dimSize, const int dim_product,
            const float *active_mask, const float *fluid_mask, const int *maskDimensions,
            float *laplace_matrix, int *cords, const float *advection_influence, const int *staggered_dimensions);

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dimSize, const int dim_product,
            const float *active_mask, const float *fluid_mask, const int *maskDimensions,
            double *laplace_matrix, int *cords, const float *advection_influence, const int *staggered_dimensions);

template <typename T>
class PressureSolveOp : public OpKernel {
    private:
      bool init_with_zeros;
      int residual_reset_steps;
      int randomized_restarts;
      int unrolling_step;
        /*int dim_product;
        float accuracy;
        int max_iterations;*/

    public:
        explicit PressureSolveOp(OpKernelConstruction* context) : OpKernel(context) {
          context->GetAttr("init_with_zeros", &init_with_zeros);
          context->GetAttr("residual_reset_steps", &residual_reset_steps);
          context->GetAttr("randomized_restarts", &randomized_restarts);
          context->GetAttr("unrolling_step", &unrolling_step);
        /*context->GetAttr("dim_product", &dim_product);
        context->GetAttr("accuracy", &accuracy);
        context->GetAttr("max_iterations", &max_iterations);*/
    }

    void Compute(OpKernelContext* context) override {
        auto begin = std::chrono::high_resolution_clock::now();

        const Tensor &input_accuracy  = context->input(13);
        const Tensor &input_max_iterations = context->input(14);
        auto accuracy = input_accuracy.flat<float>();
        auto max_iterations = input_max_iterations.flat<int32>();

        // General
        const Tensor& input_dimensions = context->input(0);

        // Laplace related
        const Tensor &input_mask_dimensions = context->input(1);
        const Tensor &input_active_mask = context->input(2);
        const Tensor &input_fluid_mask = context->input(3);
        Tensor input_laplace_matrix = context->input(4);
        const Tensor &input_advection_influence = context->input(10);
        const Tensor &input_staggered_dimensions = context->input(11);
        const Tensor &input_laplace_rank_deficient = context->input(12);

        // Pressure Solve
        Tensor input_divergence = context->input(5);
        Tensor input_p = context->input(6);
        Tensor input_r = context->input(7);
        Tensor input_z = context->input(8);
        Tensor input_pressure = context->input(9);
        const Tensor& input_bool_periodic = context->input(15);

        // Flattening
        auto dimensions = input_dimensions.flat<int32>();

        // TODO: laplace_matrix can no longer be a signed char (A_0 contribution!)
        auto mask_dimensions = input_mask_dimensions.flat<int32>();
        auto active_mask = input_active_mask.flat<float>();
        auto fluid_mask = input_fluid_mask.flat<float>();
        auto laplace_matrix = input_laplace_matrix.flat<T>();
        auto advection_influence = input_advection_influence.flat<float>();
        auto staggered_dimensions = input_staggered_dimensions.flat<int32>();
        auto laplace_rank_deficient = input_laplace_rank_deficient.flat<bool>();

        auto divergence = input_divergence.flat<T>();
        auto p = input_p.flat<T>();
        auto r = input_r.flat<T>();
        auto z = input_z.flat<T>();
        auto pressure = input_pressure.flat<T>();

        auto boolPeriodic = input_bool_periodic.flat<bool>();

        int batch_size = input_divergence.shape().dim_size(0);
        int dim_size = dimensions.size();

        int dim_product = 1;
        for (size_t i = 0; i < dim_size; i++) {
            dim_product *= input_divergence.shape().dim_size(i+1);
        }

        //printf("kernel calls\n");
        auto end = std::chrono::high_resolution_clock::now();

//        printf("General Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

        begin = std::chrono::high_resolution_clock::now();
        // Laplace:
        // Laplace Helper
        Tensor cords; // cords allocation does not really impact the performance. However it could be outsourced to be reused.
        TensorShape cords_shape;
        cords_shape.AddDim(dim_product);
        cords_shape.AddDim(dim_size);
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32, cords_shape, &cords));
        auto cords_flat = cords.flat<int32>();

        end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


        begin = std::chrono::high_resolution_clock::now();
        //printf("Laplace Launch\n");
        LaplaceMatrixKernelLauncher(dimensions.data(), dim_size, dim_product, active_mask.data(), fluid_mask.data(), mask_dimensions.data(),
        laplace_matrix.data(), cords_flat.data(), advection_influence.data(), staggered_dimensions.data());

        end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Matrix Generation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


        begin = std::chrono::high_resolution_clock::now();

        TensorShape threshold_shape;
        threshold_shape.AddDim(batch_size);
        Tensor threshold_reached_tensor;
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_BOOL, threshold_shape, &threshold_reached_tensor));
        auto threshold_reached = threshold_reached_tensor.flat<bool>();

        context->set_output(0, input_pressure);
        context->set_output(2, input_laplace_matrix);

        TensorShape iterations_shape;
        iterations_shape.AddDim(1);
        Tensor* iterations_tensor;

        OP_REQUIRES_OK(context, context->allocate_output(1, iterations_shape, &iterations_tensor));
        auto iterations_flat = iterations_tensor->flat<int>();

        end = std::chrono::high_resolution_clock::now();

//        printf("Pressure Solve Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);
        //printf("Solver Launch\n");
        begin = std::chrono::high_resolution_clock::now();
        LaunchPressureKernel(dimensions.data(), dim_product, dim_size,
                              laplace_matrix.data(),
                              p.data(), z.data(), r.data(), divergence.data(), pressure.data(),
                              threshold_reached.data(),
                              accuracy.data(),
                              max_iterations.data(),
                              batch_size,
                              iterations_flat.data(),
                              boolPeriodic.data(),
                              laplace_rank_deficient.data(),
                              init_with_zeros, residual_reset_steps, randomized_restarts, unrolling_step);
        end = std::chrono::high_resolution_clock::now();


//        printf("Pressure Solve took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);
//        printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

  }
};

REGISTER_KERNEL_BUILDER(Name("PressureSolveOp").Device(DEVICE_GPU).TypeConstraint<float>("T"), PressureSolveOp<float>);
REGISTER_KERNEL_BUILDER(Name("PressureSolveOp").Device(DEVICE_GPU).TypeConstraint<double>("T"), PressureSolveOp<double>);

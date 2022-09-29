from .piso_tf import *

# --- Load Custom Ops ---
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, '../CUDAbuild/pressure_solve_op.so')
if not os.path.isfile(kernel_path):
    raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
pressure_op = tf.load_op_library(kernel_path)


@tf.custom_gradient
def _differentiable_solve(dimensions, mask_dimensions, active_mask, accessible_mask, laplace_matrix,
                          divergence, p, r, z, guess, one_vector, input_advection_influence,
                          staggered_dimensions, input_pressure_dirichlet_mask,
                          accuracy, max_iterations, solve_count):
    pressure, iteration = pressure_op.pressure_solve_op(
        dimensions, mask_dimensions, active_mask, accessible_mask, laplace_matrix,
        divergence, p, r, z, guess, one_vector, input_advection_influence,
        staggered_dimensions, input_pressure_dirichlet_mask, accuracy, max_iterations)

    def grad(dp, di):
        grad_guess = tf.zeros_like(dp, dtype=tf.float32) + 5 + solve_count
        p = tf.zeros_like(dp, dtype=tf.float32) + 6 + solve_count
        r = tf.zeros_like(dp, dtype=tf.float32) + 7 + solve_count
        z = tf.zeros_like(dp, dtype=tf.float32) + 8 + solve_count
        divergence_grad, grad_iteration = pressure_op.pressure_solve_op(
            dimensions, mask_dimensions, active_mask, accessible_mask, laplace_matrix,
            dp, p, r, z, grad_guess, one_vector, input_advection_influence,
            staggered_dimensions, input_pressure_dirichlet_mask, accuracy, max_iterations)
        print(grad_iteration)
        return [None, None, None, None, None, divergence_grad, None, None, None, None, None, None, None, None, None, None, None]

    return [pressure, iteration], grad


class PisoPressureSolverCudaCustom(PoissonSolver):
    # performs a iterative CG linear solve based on a custom-storage laplacian
    def __init__(self, dx, accuracy=1e-5, max_iterations=2000, residual_reset=10, randomized_restarts=0, cast_to_double=True):
        PoissonSolver.__init__(self, 'CUDA Conjugate Gradient', supported_devices=('GPU',), supports_loop_counter=False, supports_guess=True, supports_continuous_masks=False)
        self.accuracy = accuracy
        self.max_iterations = max_iterations
        self.dx = dx
        self.scaling_field = None
        self.solve_count = 0.001
        self.laplace_rank_deficient = None
        self.residual_reset = residual_reset
        assert randomized_restarts >= 0
        self.randomized_restarts = randomized_restarts
        self.cast_to_double = cast_to_double

    def solve(self, scaling_field, divergence, guess, enable_backprop, simulation_physics: SimulationParameters, offset=0, unrolling_step=0):

        scaling_field = StaggeredGrid(scaling_field)

        datatype = tf.float32
        if self.cast_to_double:
            datatype = tf.float64
            divergence = tf.cast(divergence, tf.float64)

        # Setup
        dimensions = math.staticshape(divergence)[1:-1]
        dimensions = dimensions[::-1]  # the custom op needs it in the x,y,z order
        staggered_dimensions = math.flatten(np.array(
            [math.staticshape(scaling_field.data[i].data)[-2:0:-1] for i in range(len(scaling_field.data) - 1, -1, -1)], dtype=np.int32))
        dim_array = np.array(dimensions, dtype=np.int32)
        dim_product = np.prod(dimensions, dtype=np.int32)
        mask_dimensions = dim_array + 2
        laplace_matrix = tf.zeros(dim_product * (len(dimensions) * 2 + 1), dtype=datatype)

        input_advection_influence = flatten_staggered_data(scaling_field, coord_flip=False)

        # Helper variables for CG, make sure new memory is allocated for each variable.
        p = tf.zeros_like(divergence, dtype=datatype) + 1 + self.solve_count + offset
        z = tf.zeros_like(divergence, dtype=datatype) + 2 + self.solve_count + offset
        r = tf.zeros_like(divergence, dtype=datatype) + 3 + self.solve_count + offset
        if guess is None:
            guess = tf.zeros_like(divergence, dtype=datatype) + 4 + self.solve_count + offset
        else:
            if self.cast_to_double:
                guess = tf.cast(guess, tf.float64)
            else:
                guess = guess  # tf.zeros_like(guess, dtype=tf.float32) + 4 + self.solve_count

        if self.laplace_rank_deficient is None:
            prod = simulation_physics.accessible_mask * simulation_physics.active_mask + (1 - simulation_physics.accessible_mask) * (1 - simulation_physics.active_mask)
            prod = math.prod(prod[0, 0, 1:-1, 0]) * math.prod(prod[0, -1, 1:-1, 0]) * math.prod(prod[0, 1:-1, 0, 0]) * math.prod(prod[0, 1:-1, -1, 0])
            self.laplace_rank_deficient = tf.constant(prod, dtype=tf.bool, shape=[1, ])

        # Solve
        @tf.custom_gradient
        def psolve(divergence):
            pressure, iteration, lap = pressure_op.pressure_solve_op(
                dimensions, mask_dimensions, simulation_physics.active_mask, simulation_physics.accessible_mask, laplace_matrix,
                divergence, p, r, z, guess, input_advection_influence, staggered_dimensions,
                self.laplace_rank_deficient, self.accuracy, self.max_iterations, simulation_physics.bool_periodic[::-1], True, self.residual_reset, self.randomized_restarts, unrolling_step)

            def grad(dp, di, dl):
                grad_guess = tf.zeros_like(dp, dtype=datatype) + 5 + self.solve_count + offset
                p = tf.zeros_like(dp, dtype=datatype) + 6 + self.solve_count + offset
                r = tf.zeros_like(dp, dtype=datatype) + 7 + self.solve_count + offset
                z = tf.zeros_like(dp, dtype=datatype) + 8 + self.solve_count + offset
                divergence_grad, _, _ = pressure_op.pressure_solve_op(
                    dimensions, mask_dimensions, simulation_physics.active_mask, simulation_physics.accessible_mask, laplace_matrix,
                    tf.cast(dp, dtype=datatype), p, r, z, grad_guess, input_advection_influence,
                    staggered_dimensions, self.laplace_rank_deficient, self.accuracy, self.max_iterations, simulation_physics.bool_periodic[::-1], True, self.residual_reset, self.randomized_restarts,
                    100 + unrolling_step)
                return divergence_grad

            return [tf.cast(pressure, tf.float32), iteration, lap], grad

        pressure, iteration, lap = psolve(tf.cast(divergence, datatype))
        self.solve_count = self.solve_count + .001
        # print(self.solve_count)
        return pressure, iteration, lap

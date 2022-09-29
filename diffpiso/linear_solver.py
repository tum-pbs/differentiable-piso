from .piso_helpers import *
import scipy.sparse
import scipy.sparse.linalg

# LOAD CUSTOM CUDA OP
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path_bicg = os.path.join(current_dir, '../CUDAbuild/bicgstab_ilu_linear_solve_op.so')
kernel_path_multi = os.path.join(current_dir, '../CUDAbuild/multi_bicgstab_ilu_linear_solve_op.so')
if not os.path.isfile(kernel_path_multi) or not os.path.isfile(kernel_path_bicg):
    raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path_bicg)
bicg_op = tf.load_op_library(kernel_path_bicg)
multi_bicg_op = tf.load_op_library(kernel_path_multi)


class LinearSolver(object):

    def __init__(self, name, supported_devices, supports_guess, supports_batch, solver_type, input_format):
        self.name = name
        self.supported_devices = supported_devices
        self.supports_guess = supports_guess
        self.supports_batch = supports_batch
        self.solver_type = solver_type
        self.input_format = input_format

    def solve(self, *args):
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        """representation = name"""
        return self.name


class LinearSolverScipy(LinearSolver):
    def __init__(self):
        LinearSolver.__init__(self, 'Scipy Solver for sparse matrix', supported_devices='CPU', supports_guess=False,
                              supports_batch=False, solver_type='-', input_format='csr')

    def solve(self, matrix_values, row_ptr, col_indices, rhs, transpose=False):
        def scipy_solve(mv, ci, rp, rhs, transpose):
            csr_matrix = scipy.sparse.csr_matrix((mv, ci, rp))
            if transpose:
                csr_matrix = csr_matrix.transpose()
            solution = scipy.sparse.linalg.spsolve(csr_matrix, rhs)
            return solution

        @tf.custom_gradient
        def solve_call(rhs):
            sol = tf.py_function(scipy_solve, [matrix_values, col_indices, row_ptr, math.flatten(rhs), transpose], Tout=tf.float32)

            def grad(ds):
                df = tf.py_function(scipy_solve, [matrix_values, col_indices, row_ptr, ds, not transpose], Tout=tf.float32)
                return df

            return sol, grad

        solution = solve_call(rhs)
        return solution


class LinearSolverCudaBicgstabILU(LinearSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000):
        LinearSolver.__init__(self, 'CUDA iLU-preconditioned BiCGStab solve', supported_devices=('GPU',), supports_guess=True,
                              supports_batch=False, solver_type='iterative', input_format='csr')
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    def solve(self, matrix_values, row_ptr, col_indices, rhs, initial_guess=None, offset=0, transpose=False):

        flat_values = math.flatten(matrix_values)
        flat_row_ptr = math.flatten(row_ptr)
        flat_col_indices = math.flatten(col_indices)
        flat_rhs = math.flatten(rhs)

        nnz = matrix_values.shape[0]
        matrix_shape = row_ptr.shape[0] - 1

        if initial_guess is None:
            flat_x = tf.zeros(matrix_shape) + offset
            flat_x = flat_x - offset
        else:
            flat_x = tf.identity(math.flatten(initial_guess))

        s = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 1
        shat = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 2
        p = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 3
        phat = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 4
        r = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 5
        rhat = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 6
        v = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 7
        t = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 8
        z = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 9

        x_buffer = tf.zeros(matrix_shape, dtype=tf.float32) + offset + 11

        @tf.custom_gradient
        def solve_call(flat_rhs):
            sol = bicg_op.bicgstab_ilu_linear_solve(flat_values, flat_row_ptr, flat_col_indices, flat_rhs, flat_x,
                                                    s, shat, p, phat, r, rhat, v, t, z, x_buffer,
                                                    int(nnz), int(1), int(matrix_shape), self.accuracy, self.max_iterations, transpose)

            def grad(ds):
                df = bicg_op.bicgstab_ilu_linear_solve(flat_values, flat_row_ptr, flat_col_indices, ds, flat_x,
                                                       s, shat, p, phat, r, rhat, v, t, z, x_buffer,
                                                       int(nnz), int(1), int(matrix_shape), self.accuracy, self.max_iterations, not transpose)
                return df[3]

            return sol[3], grad

        solution = solve_call(flat_rhs)
        return solution


class LinearSolverCudaMultiBicgstabILU(LinearSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000, cast_to_double=False):
        LinearSolver.__init__(self, 'CUDA dual iLU-preconditioned BiCGStab solve', supported_devices=('GPU',), supports_guess=True,
                              supports_batch=False, solver_type='iterative', input_format='csr')

        self.max_iterations = max_iterations
        self.cast_to_double = cast_to_double
        if tf.is_tensor(accuracy):
            self.accuracy = accuracy
        else:
            self.accuracy = tf.constant(accuracy)

    def solve(self, matrix_values, row_ptr, col_indices, rhs, staggered_shape, initial_guess=None, offset=0,
              transpose=False, unrolling_step=0, warn=tf.zeros(shape=(1,), dtype=tf.bool)):
        datatype = tf.float32
        if self.cast_to_double:
            matrix_values = tf.cast(matrix_values, tf.float64)
            rhs = tf.cast(rhs, tf.float64)
            datatype = tf.float64

        flat_values = math.flatten(matrix_values)
        flat_row_ptr = math.flatten(row_ptr)
        flat_col_indices = math.flatten(col_indices)
        flat_rhs = math.flatten(rhs)

        shape_deduction = [np.array([1 if i != j else 0 for i in range(staggered_shape[-1])]) for j in range(staggered_shape[-1])]
        matrix_sizes = [np.prod(staggered_shape[1:-1] - sd) for sd in shape_deduction[::-1]]

        tot_mat_shape = np.sum(matrix_sizes)

        if initial_guess is None:
            flat_x = tf.zeros(tot_mat_shape, dtype=datatype) + offset
            flat_x = flat_x - offset
        else:
            flat_x = tf.identity(math.flatten(tf.cast(initial_guess, datatype)))

        s = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 1
        shat = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 2
        p = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 3
        phat = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 4
        r = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 5
        rhat = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 6
        v = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 7
        t = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 8
        z = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 9

        x_buffer = tf.zeros(tot_mat_shape, dtype=datatype) + offset + 11

        @tf.custom_gradient
        def solve_call(flat_rhs):
            sol = multi_bicg_op.multi_bicgstab_ilu_linear_solve(flat_values, flat_row_ptr, flat_col_indices, flat_rhs, flat_x,
                                                                s, shat, p, phat, r, rhat, v, t, z, x_buffer, warn,
                                                                tf.constant(matrix_sizes, tf.int32), self.accuracy, int(1), self.max_iterations, transpose, unrolling_step)

            def grad(ds, dw):
                df = multi_bicg_op.multi_bicgstab_ilu_linear_solve(flat_values, flat_row_ptr, flat_col_indices, ds, flat_x,
                                                                   s, shat, p, phat, r, rhat, v, t, z, x_buffer, warn,
                                                                   tf.constant(matrix_sizes, tf.int32), self.accuracy, int(1), self.max_iterations, not transpose, 100 + unrolling_step)
                return df[3] * (tf.constant(1.0) - tf.cast(df[4][0], dtype=tf.float32))

            return [tf.cast(sol[3], tf.float32), tf.cast(sol[4], tf.float32)], grad

        solution = solve_call(flat_rhs)
        return solution


def mat_vec_mul_csr(matrix_values, row_pointers, column_indices, staggered_field, staggered_shape):
    dim_prod = [staggered_shape[2] * (staggered_shape[1] - 1), staggered_shape[1] * (staggered_shape[2] - 1)]
    matrix_nnz = [row_pointers[dim_prod[0]], row_pointers[-1]]
    column_indices_offset = column_indices + math.concat([tf.zeros(shape=(matrix_nnz[0],), dtype=tf.int32),
                                                          dim_prod[0] * tf.ones(shape=(matrix_nnz[1],), dtype=tf.int32)], axis=0)
    data_flat = flatten_staggered_data(staggered_field, coord_flip=True)
    factor_gathered = tf.gather(data_flat, column_indices_offset)
    product = factor_gathered * matrix_values
    rptrs_offset = math.concat([row_pointers[:dim_prod[0] + 1],
                                row_pointers[dim_prod[0] + 2:] + tf.ones(shape=(dim_prod[1],), dtype=tf.int32) * row_pointers[dim_prod[0]]],
                               axis=0)
    segment_sum_ind = tf.searchsorted(rptrs_offset, tf.range(rptrs_offset[-1]), side='right')
    product_flat = tf.segment_sum(product, segment_sum_ind - 1)

    return stagger_flattened_data(product_flat, staggered_shape, coord_flip=True)


def print_residual(matrix_values, row_pointers, column_indices, staggered_field, staggered_shape, rhs):
    def printing(residual):
        print('linsolve residual', np.sum(np.abs(residual)))
        return residual

    residual = mat_vec_mul_csr(matrix_values, row_pointers, column_indices, staggered_field, staggered_shape)
    residual = flatten_staggered_data(residual) - rhs
    residual = tf.py_func(printing, [residual], Tout=[tf.float32])
    return residual

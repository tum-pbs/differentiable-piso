from .linear_solver import *

directory = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(directory, '../CUDAbuild/central_difference_csr_op.so')
if not os.path.isfile(kernel_path):
    raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)

cd_csr_op = tf.load_op_library(kernel_path)


def piso_step(velocity, pressure, pressure_inc1, pressure_inc2, dt, simulation_physics, dirichlet_values, viscosity_field=None, forcing_term=None, unrolling_step=0,
              warn=tf.zeros(shape=(1,), dtype=tf.bool),
              full_output=False, **kwargs):
    staggered_shape = velocity.staggered_tensor().shape

    def pressure_solve(field, A_0, guess, unrolling_step):
        # self.pressure_solver.set_A_0(A_0)
        res, _, L = simulation_physics.pressure_solver.solve(A_0, field, guess, False, simulation_physics, unrolling_step=unrolling_step)
        return res, L

    if viscosity_field is None:
        viscosity = tf.constant(simulation_physics.viscosity, dtype=tf.float32, shape=(1,))
    else:
        viscosity = viscosity_field

    beta = np.prod(velocity.dx) / dt

    # ADVECTION MATRICES
    matrix_values, row_pointers, column_indices, A, matrix_nnz, Aflat = \
        advection_matrix_cuda(velocity, flatten_staggered_data(simulation_physics.dirichlet_mask, True),
                              viscosity, beta=beta, no_slip_wall_mask=simulation_physics.no_slip_mask, bool_periodic=simulation_physics.bool_periodic,
                              active_mask=simulation_physics.active_mask, accessible_mask=simulation_physics.accessible_mask,
                              unrolling_step=unrolling_step)

    # Predictor step
    implicit_rhs = velocity.staggered_tensor() * beta - finite_volume_gradient_tensor(pressure, simulation_physics)
    if forcing_term is not None:
        implicit_rhs += forcing_term * np.prod(velocity.dx)
    implicit_rhs = arrange_rhs_term_tf(implicit_rhs, simulation_physics.dirichlet_mask, dirichlet_values, beta,
                                       coord_flip=True)

    sol = simulation_physics.linear_solver.solve(-matrix_values, row_pointers, column_indices, implicit_rhs,
                                                 staggered_shape, flatten_staggered_data(velocity, True), offset=1, transpose=False, unrolling_step=unrolling_step, warn=warn)
    warn = sol[1]
    sol = stagger_flattened_data(sol[0], staggered_shape, coord_flip=True)

    velocity_star = StaggeredGrid(sol, box=velocity.box, extrapolation=velocity.extrapolation)

    # Corrector step 1
    # This implicitly assumes dx=dy, since no FV scaling for the laplacian was taken
    v1div = finite_volume_divergence(velocity_star)
    # dx factor expalanation: for integrating d^2p/dx^2: FV appraoch dictates integradion over volume; divide this by dx for integrating out derivative, and second dx for compute derivative at cell face
    dx_factor = np.prod(velocity.dx) / (velocity.dx[0] ** 2)
    pressure_inc_data, Lap1 = pressure_solve(v1div, 1 / (beta - A) * dx_factor, guess=pressure_inc1.data, unrolling_step=unrolling_step)
    pressure_inc1 = CenteredGrid(pressure_inc_data, box=pressure_inc1.box, extrapolation=pressure_inc1.extrapolation)

    # gradient needs scaling from FV integral to cell values
    velocity_s2 = velocity_star.staggered_tensor() - finite_volume_gradient_tensor(pressure_inc1, sim_physics=simulation_physics) / (beta - A) / np.prod(velocity.dx)

    # Corrector step 2
    H_contribution = explicit_H_csr(matrix_values, row_pointers, column_indices,
                                    StaggeredGrid(velocity_s2 - velocity_star.staggered_tensor()),
                                    staggered_shape, A, beta)

    # This implicitly assumes dx=dy, since no FV scaling for the laplacian was taken
    H_div = finite_volume_divergence(StaggeredGrid(H_contribution / (beta - A), box=velocity.box, extrapolation=velocity.extrapolation))
    pressure_inc2_data, Lap2 = pressure_solve(H_div, 1 / (beta - A) * dx_factor, guess=pressure_inc2.data, unrolling_step=1000 + unrolling_step)
    pressure_inc2 = CenteredGrid(pressure_inc2_data, box=pressure_inc2.box, extrapolation=pressure_inc2.extrapolation)

    # gradient needs scaling from FV integral to cell values (H_contribution already includes this due to matrix)
    velocity_s3_data = velocity_s2 + (H_contribution - finite_volume_gradient_tensor(pressure_inc2, sim_physics=simulation_physics) / np.prod(velocity.dx)) / (
            beta - A)
    velocity_s3 = StaggeredGrid(velocity_s3_data, box=velocity.box, extrapolation=velocity.extrapolation)

    pressure = pressure + pressure_inc1 + pressure_inc2

    if full_output:
        return velocity_s3, pressure, pressure_inc1, pressure_inc2, matrix_values, column_indices, row_pointers, \
               velocity_star.staggered_tensor(), velocity_s2, Aflat, implicit_rhs, sol, velocity_s3_data, v1div, Lap1, Lap2, warn
    else:
        return velocity_s3, pressure, warn



def advection_matrix_cuda(velocity: StaggeredGrid, dirichlet_mask_flat, viscosity, beta=0, no_slip_wall_mask=None, bool_periodic=None, active_mask=None, accessible_mask=None, unrolling_step=0):
    dim_size = len(velocity.data)
    if bool_periodic is None:
        bool_periodic = math.zeros((dim_size,), dtype=np.bool)
    bool_periodic = bool_periodic[::-1]  # reorder Y-X to X-Y

    cuda_op = cd_csr_op.central_difference_matrix_csr

    velocity_padded = flatten_staggered_data(custom_padded(velocity, 1).staggered_tensor(), True)

    pad_depth = np.ones((dim_size * 2,), dtype=np.int32)
    grid_spacing = velocity.dx[-1::-1].astype(np.float32)
    cell_area = np.prod(velocity.dx) / velocity.dx[::-1].astype(np.float32)

    dimensions = math.flatten([math.staticshape(velocity.data[i].data)[-2:0:-1] for i in range(dim_size - 1, -1, -1)]).astype(np.int32)
    dim_product = np.array([np.int(np.prod(dimensions[i * dim_size:i * dim_size + dim_size])) for i in range(dim_size)], dtype=np.int32)

    matrix_nnz = []
    for i in range(dim_size):
        matrix_nnz.append(dim_product[i] * (dim_size * 2 + 1) -
                          2 * np.sum(dim_product[i] / dimensions[dim_size * i:dim_size * (i + 1)] * (1 - np.array(bool_periodic))))
    matrix_nnz = np.array(matrix_nnz)

    csr_val = tf.zeros(shape=(np.sum(matrix_nnz)), dtype=tf.float32)
    csr_col_ind = tf.zeros(shape=(np.sum(matrix_nnz)), dtype=tf.int32)
    csr_row_ptr = tf.zeros(shape=(np.sum(dim_product) + dim_size), dtype=tf.int32)
    diag_comp = tf.zeros(shape=(np.sum(dim_product)), dtype=tf.float32)

    if no_slip_wall_mask is None:
        no_slip_wall_mask = tf.zeros_like(diag_comp, dtype=tf.bool)

    @tf.custom_gradient
    def computation_call(velocity_padded, dimensions, pad_depth, grid_spacing, cell_area,
                         dirichlet_mask_flat, csr_val, csr_col_ind, csr_row_ptr, diag_comp,
                         viscosity, no_slip_wall_mask, boolPeriodic):
        out = cuda_op(velocity_padded, csr_val, csr_col_ind, csr_row_ptr, diag_comp, dirichlet_mask_flat,
                      math.flatten(active_mask), math.flatten(accessible_mask),
                      viscosity, dimensions, pad_depth, cell_area, grid_spacing, no_slip_wall_mask,
                      boolPeriodic, beta, unrolling_step)

        def grad(dfou_out_0, dfou_out_1, dfou_out_2, dfou_out_3):
            return [None, None, None, None, None, None, None, None, None, None, None, None, None]

        return out, grad

    out = computation_call(velocity_padded, dimensions, pad_depth, grid_spacing, cell_area, dirichlet_mask_flat,
                           csr_val, csr_col_ind, csr_row_ptr, diag_comp, viscosity, no_slip_wall_mask, bool_periodic)
    matrix_values = out[0]
    column_indices = out[1]
    row_pointers = out[2]
    A_flat = out[3]
    return matrix_values, row_pointers, column_indices, \
           stagger_flattened_data(A_flat, velocity.staggered_tensor().shape, coord_flip=True), matrix_nnz, A_flat


def pressure_extrapolation(boundaries):
    if not boundaries:
        return None

    if isinstance(boundaries, tuple):
        element = boundaries[0]
        boundaries = boundaries[1:]
        if isinstance(element, tuple):
            if len(boundaries)>0:
                val = (pressure_extrapolation(element),) + pressure_extrapolation(boundaries)
                return val
            else:
                val = (pressure_extrapolation(element),)
                return val
        else:
            if len(boundaries)>0:
                val= (element.accessible_extrapolation_mode,) + pressure_extrapolation(boundaries)
                return val
            else:
                val=(element.accessible_extrapolation_mode,)
                return val
    else:
        return boundaries.accessible_extrapolation_mode


class SimulationParameters(Physics):

    def __init__(self, dirichlet_mask, dirichlet_values, active_mask, accessible_mask, bool_periodic=None,
                 no_slip_mask=None, viscosity=0., linear_solver=None, pressure_solver=None):
        Physics.__init__(self)
        self.pressure_solver = pressure_solver
        self.linear_solver = linear_solver

        self.dirichlet_mask = dirichlet_mask
        self.dirichlet_values = dirichlet_values

        self.active_mask = active_mask
        self.accessible_mask = accessible_mask

        self.no_slip_mask = no_slip_mask
        self.bool_periodic = bool_periodic

        self.viscosity = viscosity

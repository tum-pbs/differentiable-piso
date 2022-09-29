import tensorflow as tf
import sys
import scipy

from phi.tf.flow import *
from phi.physics.field.staggered_grid import *
from phi.physics.field.grid import *

'''current_dir = os.path.dirname(os.path.realpath(__file__))
mv_kernel_path = os.path.join(current_dir, '../CUDAbuild/csr_mat_vec_prod_op.so')
if not os.path.isfile(mv_kernel_path):
    raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % mv_kernel_path)
csrMatVec_op = tf.load_op_library(mv_kernel_path)'''


@mappable()
def _custom_pad_mode_str(extrapolation):
    """
    Converts an extrapolation string (or struct of strings) to a string that can be passed to math functions like math.pad or math.resample.
    :param extrapolation: field extrapolation
    :return: padding mode, same type as extrapolation
    """
    return {'periodic': 'circular',
            'boundary': 'replicate',
            'constant': 'symmetric'}[extrapolation]

def _custom_pad_mode(extrapolation):
    """ Inserts 'constant' padding for batch dimension and channel dimension. """
    if isinstance(extrapolation, six.string_types):
        return _custom_pad_mode_str(extrapolation)
    else:
        return _custom_pad_mode_str(['constant'] + list(extrapolation) + ['constant'])


def custom_padded(staggered_field, widths):
    if isinstance(widths, int):
        widths = [[widths, widths]] * staggered_field.rank
    widths_in = widths
    pad_modes =  _custom_pad_mode(staggered_field.extrapolation)

    new_data = []
    for i in range(staggered_field.rank):
        widths = np.array([[0, 0]] + widths_in + [[0, 0]])
        data = staggered_field.data[i].data
        if isinstance(pad_modes, six.string_types):
            pad_modes = ['symmetric'] + [pad_modes] * len(math.spatial_dimensions(data)) + ['symmetric']
        for dim in math.spatial_dimensions(data):
            if (pad_modes[dim]=='circular') and (i==dim-1):
                data = math.split(data,[int(data.shape[dim])-1,1], axis=dim)[0]
                widths[dim,1] += 1
        new_data.append(math.pad(data, widths.tolist(),pad_modes))

    w_lower, w_upper = np.transpose(widths_in)
    box = AABox(staggered_field.box.lower - w_lower * staggered_field.dx, staggered_field.box.upper + w_upper * staggered_field.dx)
    return staggered_field.copied_with(data=new_data, box=box)


def update_dirichlet_values(dirichlet_values, update_bool, dirichlet_array):
    # updates dirichelt values for temporally changing dirichlet conditions
    values_unstacked = unstack_staggered_tensor(dirichlet_values)
    if update_bool[0][0]:
        values_unstacked[0] = math.concat([dirichlet_array[0][0][...,1:-1,:], values_unstacked[0][:,1:,...]],1)
    if update_bool[0][1]:
        values_unstacked[0] = math.concat([values_unstacked[0][:,:-1,...], dirichlet_array[0][1][...,1:-1,:]],1)
    if update_bool[1][0]:
        values_unstacked[1] = math.concat([dirichlet_array[1][0][:, 1:-1, ...], values_unstacked[1][..., 1:, :]], 2)
    if update_bool[1][1]:
        values_unstacked[1] = math.concat([values_unstacked[1][..., :-1, :], dirichlet_array[1][1][:, 1:-1, ...]], 2)

    return stack_staggered_components(values_unstacked)


def compute_mixingLayer_masks(staggered_shape, dirichlet_bool, dirichlet_array, dtype = np.float32):
    staggered_shape = np.array(staggered_shape)
    mask = [math.zeros(staggered_shape - np.array([0, 2, 1, 1])), math.zeros(staggered_shape - np.array([0, 1, 2, 1]))]
    neumann = [math.zeros(staggered_shape - np.array([0, 2, 1, 1])), math.zeros(staggered_shape - np.array([0, 1, 2, 1]))]
    values = [math.zeros(staggered_shape - np.array([0, 2, 1, 1])), math.zeros(staggered_shape - np.array([0, 1, 2, 1]))]

    concat_values =[]
    concat_masks = []
    concat_neumann = []
    concat_shapes = (staggered_shape - np.array([0, staggered_shape[1]-1, 1, 1]), staggered_shape - np.array([0, 1, staggered_shape[2]-1, 1]))
    if dirichlet_bool[0][0]:
        concat_values.append(dirichlet_array[0][0][...,1:-1,:])
        concat_masks.append(math.ones(concat_shapes[0], dtype=dtype))
        concat_neumann.append(math.zeros(concat_shapes[0], dtype=dtype))
    else:
        concat_values.append(math.zeros(concat_shapes[0], dtype=dtype))
        concat_masks.append(math.zeros(concat_shapes[0], dtype=dtype))
        concat_neumann.append(math.ones(concat_shapes[0], dtype=dtype))

    if dirichlet_bool[0][1]:
        concat_values.append(dirichlet_array[0][1][...,1:-1,:])
        concat_masks.append(math.ones(concat_shapes[0], dtype=dtype))
        concat_neumann.append(math.zeros(concat_shapes[0], dtype=dtype))
    else:
        concat_values.append(math.zeros(concat_shapes[0], dtype=dtype))
        concat_masks.append(math.zeros(concat_shapes[0], dtype=dtype))
        concat_neumann.append(math.ones(concat_shapes[0], dtype=dtype) * 2)

    if dirichlet_bool[1][0]:
        concat_values.append(dirichlet_array[1][0][:,1:-1,...])
        concat_masks.append(math.ones(concat_shapes[1], dtype=dtype))
        concat_neumann.append(math.zeros(concat_shapes[1], dtype=dtype))
    else:
        concat_values.append(math.zeros(concat_shapes[1], dtype=dtype))
        concat_masks.append(math.zeros(concat_shapes[1], dtype=dtype))
        concat_neumann.append(math.ones(concat_shapes[1], dtype=dtype))

    if dirichlet_bool[1][1]:
        concat_values.append(dirichlet_array[1][1][:,1:-1,...])
        concat_masks.append(math.ones(concat_shapes[1], dtype=dtype))
        concat_neumann.append(math.zeros(concat_shapes[1], dtype=dtype))
    else:
        concat_values.append(math.zeros(concat_shapes[1], dtype=dtype))
        concat_masks.append(math.zeros(concat_shapes[1], dtype=dtype))
        concat_neumann.append(math.ones(concat_shapes[1], dtype=dtype) * 2)

    for i in range(2):
        mask[i] = math.concat([concat_masks[i*2], mask[i], concat_masks[i*2+1]],i+1)
        neumann[i] = math.concat([concat_neumann[i*2], neumann[i], concat_neumann[i*2+1]],i+1)
        values[i] = math.concat([concat_values[i*2], values[i], concat_values[i*2+1]],i+1)

    accessible_mask = math.ones((staggered_shape[1] + 1, staggered_shape[2] + 1))
    accessible_mask[:, 0] = 0
    accessible_mask[0, :] = 0
    accessible_mask[-1, :] = 0
    accessible_mask = math.expand_dims(accessible_mask, axis=(0, -1))

    active_mask = math.pad(math.ones((staggered_shape[1] - 1, staggered_shape[2] - 1)), ((1, 1), (1, 1)), "constant")
    active_mask = math.expand_dims(active_mask, axis=(0, -1))

    return stack_staggered_components(mask), stack_staggered_components(values), stack_staggered_components(neumann), active_mask, accessible_mask


def temporal_mixing_layer_masks(staggered_shape, dirichlet_bool, dirichlet_array, dtype=np.float32):
    assert dirichlet_bool == ((True,True),(False,False))
    staggered_shape = np.array(staggered_shape)
    concat_shapes = (staggered_shape - np.array([0, staggered_shape[1] - 1, 1, 1]),
                     staggered_shape - np.array([0, 1, staggered_shape[2] - 1, 1]))

    mask = [math.concat([math.ones(concat_shapes[0]),
                         math.zeros(staggered_shape - np.array([0, 2, 1, 1])),
                         math.ones(concat_shapes[0])], 1),
            math.zeros(staggered_shape - np.array([0, 1, 0, 1]))]
    values = [math.concat([dirichlet_array[0][0][...,1:-1,:],
                           math.zeros(staggered_shape - np.array([0, 2, 1, 1])),
                           dirichlet_array[0][1][...,1:-1,:]], 1),
              math.zeros(staggered_shape - np.array([0, 1, 0, 1]))]

    boundary_bool_x = np.zeros([1, staggered_shape[1] - 1, staggered_shape[2], 4], dtype=np.bool)
    boundary_bool_x[:,0,:,2] = True
    boundary_bool_x[:,-1,:,3] = True

    boundary_bool_y = np.zeros([1, staggered_shape[1], staggered_shape[2] - 1, 4], dtype=np.bool)
    boundary_bool_y[:,0,:,2] = True
    boundary_bool_y[:,-1,:3] = True

    accessible_mask = math.concat([math.zeros((1,staggered_shape[2]+1)),
                                   math.ones((staggered_shape[1]-1,staggered_shape[2]+1)),
                                   math.zeros((1,staggered_shape[2]+1))], axis = 0)
    accessible_mask = math.expand_dims(accessible_mask, axis=(0,-1))
    active_mask = accessible_mask

    return stack_staggered_components(mask), stack_staggered_components(values),[boundary_bool_x, boundary_bool_y],\
           active_mask, accessible_mask


def arrange_rhs_term_tf(rhs, dirichlet_mask, dirichlet_values, beta, coord_flip=False, bool_periodic=None):
    rhs_out = (1 - dirichlet_mask) * rhs + dirichlet_mask * dirichlet_values * -1
    rhs_out = flatten_staggered_data(rhs_out, coord_flip=coord_flip)
    return rhs_out


def flatten_staggered_data(data, coord_flip=False):
    if isinstance(data, StaggeredGrid):
        data_grid = data
    else:
        data_grid = StaggeredGrid(data)

    if coord_flip:
        data_flat = math.concat([math.flatten(data_grid.data[i].data) for i in range(len(data_grid.data) - 1, -1, -1)], axis=0)
    else:
        data_flat = math.concat([math.flatten(data_grid.data[i].data) for i in range(len(data_grid.data))], axis=0)
    return data_flat


def stagger_flattened_data(flat_data, staggered_shape, coord_flip=False):
    if coord_flip:
        dim_prod = 0
        components = []
        for i in range(staggered_shape[-1]-1,-1,-1):
            local_shape = [1,]+[staggered_shape[j] if j-1==i else staggered_shape[j]-1 for j in range(1,staggered_shape[-1]+1)]+[1,]
            dim_add = np.prod(local_shape)
            components.append(math.reshape(flat_data[dim_prod:dim_prod+dim_add], local_shape))
            dim_prod += dim_add
        return stack_staggered_components(components[::-1])
    else:
        dim_prod = 0
        components = []
        for i in range(staggered_shape[-1]):
            local_shape = [1,]+[staggered_shape[j] if j-1==i else staggered_shape[j]-1 for j in range(1,staggered_shape[-1]+1)]+[1,]
            dim_add = np.prod(local_shape)
            components.append(math.reshape(flat_data[dim_prod:dim_prod+dim_add], local_shape))
            dim_prod += dim_add
        return stack_staggered_components(components)


def explicit_H_csr(matrix_values, row_pointers, column_indices, velocity, staggered_shape, A, beta=0):
    # computes the matrix-vector product of the H matrix in the second PISO corrector step
    # matrix_values, row_pointers, column_indices store the (u,v,w) Advection-diffusion matrices M, H=M-A
    dim_prod = [math.prod(math.staticshape(d.data)[1:-1]) for d in velocity.data[::-1]]
    matrix_nnz = [row_pointers[math.sum(dim_prod[:i+1])+i] for i in range(len(dim_prod))]
    column_indices_offset = column_indices + math.concat([tf.ones(shape = (matrix_nnz[i]), dtype=tf.int32) * math.sum(dim_prod[:i]) for i in range(len(dim_prod))], axis =0)
    velocity_flat = flatten_staggered_data(velocity, coord_flip=True)
    factor_gathered = tf.gather(velocity_flat, column_indices_offset)
    product = factor_gathered * matrix_values
    rptrs_offset = math.concat([row_pointers[:dim_prod[0]+1]] +
                               [row_pointers[np.sum(dim_prod[:i])+i+1 : np.sum(dim_prod[:i+1])+i+1] + np.sum(matrix_nnz[:i])
                                for i in range(1,len(dim_prod))], axis=0)
    segment_sum_ind = tf.searchsorted(rptrs_offset, tf.range(rptrs_offset[-1]), side='right')
    product_flat = tf.segment_sum(product, segment_sum_ind - 1)
    return stagger_flattened_data(product_flat, staggered_shape, coord_flip=True) - (A - beta) * velocity.staggered_tensor()


@tf.custom_gradient
def circular_padded_gradient(centered_data, dim):
    result = centered_data - math.roll(centered_data,1,dim)
    result = math.concat([result, math.split(result, [1, -1],axis=dim)[0]], axis=dim)
    def grad(staggered_data):
        gradient = math.split(staggered_data, [-1,1], dim)[0] - math.split(staggered_data, [1,-1], dim)[-1]
        return gradient,None
    return result, grad


def finite_volume_gradient_tensor(centered_field,sim_physics=None):
    # computes the pressure-gradient influence on the staggered grid
    assert isinstance(centered_field, CenteredGrid)
    data = centered_field.data
    if data.shape[-1] != 1:
        raise ValueError('input must be a scalar field')
    tensors = []
    pad_modes = _custom_pad_mode(centered_field.extrapolation)
    if isinstance(pad_modes,str):
        pad_modes = ['symmetric'] + [pad_modes] * len(math.spatial_dimensions(data)) + ['symmetric']

    for dim in math.spatial_dimensions(data):
        if pad_modes[dim]=='circular':
            spatial_gradient_tensor = circular_padded_gradient(data, dim)
            tensors.append(spatial_gradient_tensor* np.prod(centered_field.dx) / centered_field.dx[dim - 1])
        else:
            upper = centered_field.padded([[0, 1] if d == dim else [0, 0] for d in math.spatial_dimensions(data)]).data
            lower = centered_field.padded([[1, 0] if d == dim else [0, 0] for d in math.spatial_dimensions(data)]).data
            tensors.append((upper - lower) * np.prod(centered_field.dx) / centered_field.dx[dim - 1])
    if sim_physics is not None:
        zero_gradient_mask = []
        for dim in math.spatial_dimensions(data):
            upper = np.split(sim_physics.accessible_mask, [1], axis=dim)[1]
            lower = np.split(sim_physics.accessible_mask, [-1], axis=dim )[0]
            for d in math.spatial_dimensions(data):
                if d != dim:
                    upper = np.split(upper, [1, -1], axis=d)[1]
                    lower = np.split(lower, [1, -1], axis=d)[1]
            zero_gradient_mask.append(math.minimum(upper,lower))
        result = stack_staggered_components(tensors) * stack_staggered_components(zero_gradient_mask)
    else:
        result = stack_staggered_components(tensors)

    def grad(staggered_tensor):
        grad_vals = - staggered_tensor[:,1:,:-1,0] + staggered_tensor[:,:-1,:-1,0] \
                    - staggered_tensor[:,:-1,1:,1] +staggered_tensor[:,:-1,:-1,1]
        grad_vals = math.expand_dims(grad_vals, axis=-1)
        return CenteredGrid(grad_vals, box=centered_field.box, extrapolation=centered_field.extrapolation), None, None
    return result


def finite_volume_divergence(staggered_field):
    assert isinstance(staggered_field, StaggeredGrid)
    staggered_data = staggered_field.staggered_tensor()
    dx_prod = np.prod(staggered_field.dx)
    pad_modes =  _custom_pad_mode(staggered_field.extrapolation)
    if isinstance(pad_modes, six.string_types):
        pad_modes = ['symmetric'] + [pad_modes] * len(math.spatial_dimensions(staggered_data)) + ['symmetric']

    @tf.custom_gradient
    def custom_divergence(staggered_tensor):
        staggered_data_list = unstack_staggered_tensor(staggered_tensor)
        centered_data = math.sum([math.axis_gradient(staggered_data_list[i], i) * dx_prod / staggered_field.dx[i]
                                  for i in range(len(staggered_data_list))],axis=0)

        def grad(dcentered_data):
            result = []
            for i in range(staggered_field.rank):
                data_shape = list(dcentered_data.shape.as_list())
                spatial_dim = math.spatial_dimensions(dcentered_data)[i]
                data_shape[spatial_dim] = 1
                if pad_modes[spatial_dim] =='circular':
                    upper_slices = tuple([(slice(-2,-1) if j == i else slice(None)) for j in range(staggered_field.rank)])
                    lower_slices = tuple([(slice(0, 1) if j == i else slice(None)) for j in range(staggered_field.rank)])
                    result.append(- math.concat([dcentered_data, dcentered_data[(slice(None),) + lower_slices + (slice(None),)]], axis=spatial_dim) * dx_prod / staggered_field.dx[i]+
                                    math.concat([dcentered_data[(slice(None),) + upper_slices + (slice(None),)], dcentered_data], axis=spatial_dim) * dx_prod / staggered_field.dx[i])
                else:
                    result.append(-math.concat([dcentered_data, math.zeros(data_shape)], axis=spatial_dim) * dx_prod / staggered_field.dx[i] +
                                  math.concat([math.zeros(data_shape), dcentered_data], axis=spatial_dim) * dx_prod / staggered_field.dx[i])
            return stack_staggered_components(result)
        return centered_data, grad

    centered_data = custom_divergence(staggered_data)

    return centered_data


def vorticity(velocity: StaggeredGrid):
    central_data = velocity.at_centers().data
    gradients = [math.gradient(math.expand_dims(central_data[...,d], axis=-1), velocity.dx[0], 'central') for d in range(central_data.shape[-1])]
    if central_data.shape[-1]==2:
        vorticity = math.expand_dims(gradients[0][..., 1] - gradients[1][..., 0], axis=-1)
    else:
        vorticity = [gradients[0][...,1]-gradients[1][...,0],
                     gradients[2][...,0]-gradients[0][...,2],
                     gradients[1][...,2]-gradients[2][...,1]]
        vorticity = math.concat([math.expand_dims(v, axis =-1) for v in vorticity], axis=-1)
    return vorticity


def convert_to_scipy_csr(matrix_values, column_indices, row_pointers, staggered_shape):
    # converts CSR matrices to Scipy
    # matrix_values, column_indices, row_pointers all are concatenated to contain (u,v,w) matrices
    shape_deduction = [np.array([1 if i != j else 0 for i in range(staggered_shape[-1])]) for j in range(staggered_shape[-1])]
    matrix_sizes = [np.prod(staggered_shape[1:-1] - sd) for sd in shape_deduction[::-1]]
    csr_matrices = []
    rp_offset = 0
    mv_offset = 0
    for d in range(staggered_shape[-1]):
        rp = row_pointers[rp_offset+d:matrix_sizes[d]+rp_offset+d+1]
        rp_offset = rp_offset + matrix_sizes[d]

        mv = matrix_values[mv_offset:mv_offset+rp[-1]]
        ci = column_indices[mv_offset:mv_offset+rp[-1]]
        mv_offset = mv_offset + rp[-1]

        csr_matrices.append(scipy.sparse.csr_matrix((mv,ci,rp)))
    return csr_matrices


def calculate_staggered_shape(batch_size, resolution):
    return math.concat([[batch_size],
                       np.array(resolution) + np.array([1,]*resolution.shape[0]),
                      [resolution.shape[0]]], axis=0)


def calculate_centered_shape(batch_size, resolution):
    return math.concat([[batch_size], np.array(resolution), [1]], axis=0)
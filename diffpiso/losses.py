from phi.tf.flow import *
from diffpiso.piso_tf import stack_staggered_components
from diffpiso.piso_helpers import custom_padded
from diffpiso.evaluation_tools import EK_spectrum_1D_tf, EK_spectrum_2D_tf

def L2_field_loss(loss, fields, ground_truths, step_range, buffer_width, loss_factor, sponge_start, box=None, sum_steps= True,loss_influence_range=None, **kwargs):
    if not isinstance(step_range, list):
        step_range = [0, step_range]
    if not isinstance(loss_factor, list):
        loss_factor = [loss_factor for i in range(step_range[1])]
    loss_contrib = [[] for i in range(step_range[1]-step_range[0])]
    for i in range(len(fields)):
        for s in range(step_range[0],step_range[1]):
            loss_target_vel = StaggeredGrid(ground_truths[i][:, s, ...])#, box)
            if buffer_width is not None:
                stag_data = fields[i][s].staggered_tensor()
                target_stag_data = loss_target_vel.staggered_tensor()
                stag_shape = stag_data.shape
                if sponge_start==0:
                    sponge_start=stag_data.shape[2]
                loss_contrib[s-step_range[0]].append(
                    loss_factor[s] * tf.nn.l2_loss(stag_data
                    [:, buffer_width[0][0]:np.int32(stag_shape[1])-buffer_width[0][1], buffer_width[1][0]:np.int32(sponge_start) - buffer_width[1][1],:]
                    - target_stag_data
                    [:, buffer_width[0][0]:np.int32(stag_shape[1])-buffer_width[0][1], buffer_width[1][0]:np.int32(sponge_start) - buffer_width[1][1],:]))
            else:
                loss_contrib[s-step_range[0]].append(loss_factor[s] * tf.nn.l2_loss(fields[i][s].staggered_tensor()- loss_target_vel.staggered_tensor()))
    if sum_steps==True:
        loss_contrib = tf.reduce_sum(loss_contrib)
        return loss+loss_contrib, loss_contrib
    else:
        loss_contrib = [tf.reduce_sum(loss_contrib[i*loss_influence_range:min((i+1)*loss_influence_range, len(loss_contrib))])
                        for i in range((len(loss_contrib)-1)//loss_influence_range+1)]
        print(loss_contrib)
        return [loss[i]+loss_contrib[i//loss_influence_range] for i in range(step_range[1]-step_range[0])], loss_contrib


def spectral_energy_loss(loss, velocity_fields, ground_truths, step_range, buffer_width=[[0,0],[0,0]], loss_factor=1, sponge_start = 0,
                         log_distance = True, start_wavenumber=0, sum_steps=True,loss_influence_range=None, **kwargs):
    if not isinstance(step_range, list):
        step_range = [0, step_range]
    if not isinstance(loss_factor, list):
        loss_factor = [loss_factor for i in range(step_range[1])]
    loss_contrib = []
    for s in range(step_range[0], step_range[1]):
        central_data = velocity_fields[0][s].at_centers().data
        cen_shape = central_data.shape
        if sponge_start == 0:
            sponge_start = central_data.shape[2]
        central_data = central_data[:, buffer_width[0][0]:np.int32(cen_shape[1])-buffer_width[0][1], buffer_width[1][0]:np.int32(sponge_start) - buffer_width[1][1],:]
        central_data = tf.cast(central_data, tf.complex64)
        e_spectrum = EK_spectrum_2D_tf(central_data[0,...])
        central_gt_data = StaggeredGrid(ground_truths[0][:,s,...]).at_centers().data
        central_gt_data = central_gt_data[:, buffer_width[0][0]:np.int32(cen_shape[1])-buffer_width[0][1], buffer_width[1][0]:np.int32(sponge_start) - buffer_width[1][1],:]
        central_gt_data = tf.cast(central_gt_data, tf.complex64)
        gt_spectrum = EK_spectrum_2D_tf(central_gt_data[0,...])
        if log_distance:
            distance = tf.log(gt_spectrum[:e_spectrum.shape[0]]/e_spectrum)**2
            loss_contrib.append(tf.sqrt(tf.reduce_sum(distance[1+start_wavenumber:]))*loss_factor[s])
        else:
            loss_contrib.append(tf.reduce_sum(tf.abs(gt_spectrum[:e_spectrum.shape[0]] - e_spectrum)[1:])*loss_factor[s])
    if sum_steps:
        return loss+tf.reduce_sum(loss_contrib), tf.reduce_sum(loss_contrib)
    else:
        return [loss[i] + math.sum(loss_contrib[i:min(i+loss_influence_range,len(loss_contrib))]) for i in range(step_range[1]-step_range[0])], loss_contrib

def strain_rate_loss(loss, velocity_fields, ground_truths, step_range, buffer_width, loss_factor=1, sponge_start=0, box=None, sum_steps=True,loss_influence_range=None, **kwargs):
    if not isinstance(step_range, list):
        step_range = [0, step_range]
    if not isinstance(loss_factor, list):
        loss_factor = [loss_factor for i in range(step_range[1])]

    loss_contrib =[]
    for s in range(step_range[0], step_range[1]):
        velocity_padded = velocity_fields[0][s] #custom_padded( velocity_fields[0][s], 1)
        grads = [math.gradient(velocity_padded.data[i].data, velocity_padded.dx, 'forward') for i in range(len(velocity_padded.data))]
        strain = [grads[0][:, :-1, :, 0],
                  (grads[0][:, 1:-1, 0:-1, 1] + grads[1][:, 0:-1, 1:-1, 0]) / 2,
                  (grads[0][:, 1:-1, 0:-1, 1] + grads[1][:, 0:-1, 1:-1, 0]) / 2,
                  grads[1][:, :, :-1, 1]]

        gt_grid = StaggeredGrid(ground_truths[0][:,s,...], velocity_fields[0][s].box)
        gt_grads = [math.gradient(gt_grid.data[i].data, velocity_padded.dx, 'forward') for i in range(len(gt_grid.data))]
        gt_strain = [gt_grads[0][:, :-1, :, 0],
                  (gt_grads[0][:, 1:-1, 0:-1, 1] + gt_grads[1][:, 0:-1, 1:-1, 0]) / 2,
                  (gt_grads[0][:, 1:-1, 0:-1, 1] + gt_grads[1][:, 0:-1, 1:-1, 0]) / 2,
                  gt_grads[1][:, :, :-1, 1]]
        loss_contrib.append(sum([tf.reduce_sum(tf.abs(strain[i] - gt_strain[i]))
                                 for i in range(len(strain))]) * loss_factor[s])

    if sum_steps:
        return loss + tf.reduce_sum(loss_contrib), tf.reduce_sum(loss_contrib)
    else:
        return [loss[i] + math.sum(loss_contrib[i:min(i+loss_influence_range,len(loss_contrib))]) for i in range(step_range[1] - step_range[0])], loss_contrib

def multistep_averaging_loss(loss, velocity_fields, ground_truths, step_range, buffer_width, loss_factor=1, sponge_start=0, box=None,
                             sum_steps = True, loss_influence_range=None, **kwargs):
    if not isinstance(step_range, list):
        step_range = [0, step_range]

    data_u =[]
    averages_u =[]
    data_v =[]
    averages_v =[]
    data_u_gt = []
    averages_u_gt =[]
    data_v_gt = []
    averages_v_gt =[]

    for s in range(step_range[0],step_range[1]):
        shape_u = np.int32(velocity_fields[0][s].data[1].data.shape)
        shape_v = np.int32(velocity_fields[0][s].data[0].data.shape)
        data_u.append(velocity_fields[0][s].data[1].data[:,buffer_width[0][0]:shape_u[1]-buffer_width[0][1],
                                                            buffer_width[1][0]:shape_u[2]-buffer_width[1][1],0])
        data_v.append(velocity_fields[0][s].data[0].data[:,buffer_width[0][0]:shape_v[1]-buffer_width[0][1],
                                                            buffer_width[1][0]:shape_v[2]-buffer_width[1][1],0])
        gt_grid = StaggeredGrid(ground_truths[0][:, s, ...])
        data_u_gt.append(gt_grid.data[1].data[:,buffer_width[0][0]:shape_u[1]-buffer_width[0][1],buffer_width[1][0]:shape_u[2]-buffer_width[1][1],0])
        data_v_gt.append(gt_grid.data[0].data[:,buffer_width[0][0]:shape_v[1]-buffer_width[0][1],buffer_width[1][0]:shape_v[2]-buffer_width[1][1],0])

    if loss_influence_range is None:
        loss_influence_range = step_range[1]-step_range[0]

    data_u = math.concat(data_u, axis=0)
    data_v = math.concat(data_v, axis=0)
    data_u_gt = math.concat(data_u_gt, axis=0)
    data_v_gt = math.concat(data_v_gt, axis=0)

    for i in range(step_range[1]-step_range[0]-loss_influence_range+1):
        averages_u.append(math.mean(data_u[i:i+loss_influence_range], axis=(0)))
        averages_v.append(math.mean(data_v[i:i+loss_influence_range], axis=(0)))
        averages_u_gt.append(math.mean(data_u_gt[i:i+loss_influence_range], axis=(0)))
        averages_v_gt.append(math.mean(data_v_gt[i:i+loss_influence_range], axis=(0)))

    loss_contrib = []
    for i in range(step_range[1]-step_range[0]):
        if i<loss_influence_range//2:
            loss_contrib.append((math.sum(math.abs(averages_u[0]-averages_u_gt[0])) +math.sum(math.abs(averages_v[0]-averages_v_gt[0]))) * loss_factor)
        elif i>=(loss_influence_range//2+step_range[1]-step_range[0]-loss_influence_range):
            loss_contrib.append((math.sum(math.abs(averages_u[-1] - averages_u_gt[-1])) + math.sum(math.abs(averages_v[-1] - averages_v_gt[-1]))) * loss_factor)
        else:
            loss_contrib.append((math.sum(math.abs(averages_u[i-loss_influence_range//2] - averages_u_gt[i-loss_influence_range//2])) +
                                 math.sum(math.abs(averages_v[i-loss_influence_range//2] - averages_v_gt[i-loss_influence_range//2]))) * loss_factor)

    if sum_steps:
        return loss + tf.reduce_sum(loss_contrib), tf.reduce_sum(loss_contrib)
    else:
        return [loss[i] + loss_contrib[i] for i in range(step_range[1] - step_range[0])], loss_contrib

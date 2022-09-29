import sys
import time
import socket
import pickle
from diffpiso import *
from diffpiso.networks import initialise_fullyconv_network
from diffpiso.losses import *

base_path = '../learnedTurbulenceModelling_data/spatialMixingLayer/' # set base directory where dataset is located/ simulation will be stored

starting_frame = 0
timesteps = 2500

learning_dir = '' # set directory where model is stored
model_id = ''  # set model number  (id scheme in learning: EEEEEEiXXXXXX with E as epoch number and X as iteration number)

def neural_network_wrapper(neural_network, input, fluid, physical_parameters, simulation_parameters, loss_buffer_width, buffer_width):
    sponge_start = int(simulation_parameters['HRres'][1] * simulation_parameters['sponge_ratio']) // simulation_parameters['dx_ratio']
    NN_in = input[:, :, :sponge_start, :]
    NN_out = tf.pad(neural_network(NN_in), ((0, 0), (0, 0), (0, fluid.resolution[1] - sponge_start), (0, 0)))
    return NN_out

physical_parameters   = {'average_velocity':        1,
                         'velocity_difference':     1,
                         'inlet_profile_sharpness': 2,
                         'viscosity':               .002}

simulation_parameters = {'HRres':                   [64,64*4], # [512,512*4],
                         'dx_ratio':                1, # 8,
                         'differentiation_scheme':  'central_difference_new',
                         'dt':                      .05,
                         'dt_ratio':                1, # 8,
                         'box':                     box[0:64,0:64*4],
                         'sponge_ratio':            .875,
                         'relative_sponge_max':     20}

training_dict =         {'step_count':              1,
                         'grad_stop': 0,
                         'artificial_batch': 1,
                         'epochs':                  5,
                         'dataset':                 base_path+'/sml_HR_512-2048_dx8_dt8_pert0.082-0.018/',
                         'dataset_characteristics': [(0.082,0.018)],
                         'start_frame':             8010,
                         'frame_count_training':    27000,
                         'frame_count_validation':  4900,
                         'perturb_inlet':           True,
                         'pressure_included':       True,
                         'network_initialiser':     initialise_fullyconv_network,
                         'padding':                 'VALID',
                         'load_model_path':         base_path+learning_dir+'/model_epoch_'+model_id+'.ckpt',
                         'loss_functions':          [L2_field_loss],
                         'loss_factor':             [1],
                         'HR_buffer_width':         [[0, 0], [0, 0]],
                         'data_shuffling_seeds':    None,
                         'start_first_epoch_at':    0,
                         'learning_rate':           8e-6,
                         'lr_decay_fun':            lambda l: l*.8,
                         'store_interm_ckpts':      200,
                         'staggered_formulation':   False
                         }

buffer_width = [[i // simulation_parameters['dx_ratio'] for i in j] for j in training_dict['HR_buffer_width']]
sponge_start = int(simulation_parameters['HRres'][1] * simulation_parameters['sponge_ratio']) // simulation_parameters['dx_ratio']  # //2

solver_precision = 1e-8
domain, sim_physics, pressure_solver, velocity_placeholder, velocity, pressure_placeholder, pressure, viscosity_field, bc_placeholders, bcx= \
        spatialMixingLayer_setup(simulation_parameters, solver_precision, physical_parameters, 1)

# NN DEFINITION -------------------------------------------------------------------------------------------------
if (training_dict['load_model_path']is None):
    load_model_path = base_path + '/model_epoch_'+str(training_dict['epochs']-1).zfill(6)+'.ckpt'
else:
    load_model_path = training_dict['load_model_path']

print('LOAD MODEL PATH',load_model_path)
assert training_dict['network_initialiser'] is not None
neural_network, weights, loss_buffer_width = \
    training_dict['network_initialiser'](buffer_width=buffer_width, padding=training_dict['padding'], restore_shape=True)
saver = tf.train.Saver(weights)

dirichlet_placeholder_update = lambda dv, tf_pl: update_dirichlet_values(dv,((False, False), (True, False)),tf_pl)

velocity_all_steps, pressure_all_steps, nn_all_steps, velnew, pnew, NN_out,warn, velocity_all_arrays, pressure_all_arrays = \
    run_piso_steps(velocity, pressure, domain, physical_parameters, simulation_parameters, training_dict, neural_network,neural_network_wrapper,
                   sim_physics, viscosity_field, bcx, bc_placeholders,
                   dirichlet_placeholder_update=dirichlet_placeholder_update, loss_buffer_width=loss_buffer_width)
velnew_data = velnew.staggered_tensor()
pnew_data = pnew.data
residual_force_data = NN_out

def boundary_perturbation_fun_new(shape,time):
    return boundary_perturbation_fun(domain, physical_parameters['average_velocity'], shape, time, training_dict['dataset_characteristics'][0])

tf.Graph.finalize(tf.get_default_graph())

# SIMULATION RUN -------------------------------------------------------------------------------------------------
performance = []
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
with tf.Session(config=session_config) as sess:
    if load_model_path is not None and training_dict['network_initialiser'] is not None:
        print('LOAD MODEL PATH', load_model_path)
        saver.restore(sess, load_model_path.replace('//','/'))

    sub_path = create_base_dir(base_path+learning_dir, '/start_' + str(starting_frame).zfill(6) + '_'+model_id+
                               '_pert'+str(training_dict['dataset_characteristics'][0][0])+'-'+str(training_dict['dataset_characteristics'][0][1])+'_')
    os.mkdir(sub_path+'/plots')

    initial_vel = np.load(training_dict['dataset'] + 'velocity_' + str(starting_frame).zfill(6) + '.npz')['arr_0']
    initial_pre = np.load(training_dict['dataset']  + 'pressure_' + str(starting_frame).zfill(6) + '.npz')['arr_0']
    vel_np = StaggeredGrid(initial_vel, velocity.box).at(velocity)
    p_np = CenteredGrid(initial_pre, pressure.box).at(pressure)
    np.savez(sub_path + '/velocity_' + str(0).zfill(6), vel_np.staggered_tensor())
    np.savez(sub_path + '/pressure_' + str(0).zfill(6), p_np.data)
    if residual_force_data is not None:
        np.savez(sub_path + '/nn_forcing_' + str(0).zfill(6), np.zeros_like(vel_np.staggered_tensor()))

    for i in range(1,timesteps):
        # BOUNDARY CONDITION - PERTURBATION -----------------------------------------------------------------------
        if training_dict['perturb_inlet'] == True:
            boundary_perturbation = boundary_perturbation_fun_new(bc_placeholders.shape, simulation_parameters['dt']*starting_frame+
                                                                  simulation_parameters['dt']*simulation_parameters['dt_ratio']*i)
        else:
            boundary_perturbation = np.zeros(bc_placeholders.shape)

        s = time.time()
        vel_out, p_out, nn_out = sess.run([velnew_data, pnew_data, residual_force_data],
                                          feed_dict={velocity_placeholder: vel_np.staggered_tensor(),
                                                     pressure_placeholder: p_np.data,
                                                     bc_placeholders: boundary_perturbation})
        f = time.time()
        performance.append(f-s)
        np.savez(sub_path + '/velocity_' + str(i).zfill(6), vel_out)
        np.savez(sub_path + '/pressure_' + str(i).zfill(6), p_out)
        if residual_force_data is not None:
            np.savez(sub_path + '/nn_forcing_' + str(i).zfill(6), nn_out)

        if i%50==0:
            plt.figure(figsize=(8,12))
            plt.subplot(5,1,1)
            plt.title("v velocity")
            plt.imshow(vel_out[0,...,0])
            plt.colorbar()
            plt.subplot(5,1,2)
            plt.title("u velocity")
            plt.imshow(vel_out[0,...,1])
            plt.colorbar()
            plt.subplot(5,1,3)
            plt.title("p pressure")
            plt.imshow(p_out[0,...,0])
            plt.colorbar()
            plt.subplot(5,1,4)
            plt.title("nn forcing y")
            plt.imshow(nn_out[0,...,0])
            plt.colorbar()
            plt.subplot(5,1,5)
            plt.title("nn forcing x")
            plt.imshow(nn_out[0,...,1])
            plt.colorbar()
            plt.savefig(sub_path+'/plots/plt_'+str(i))
            plt.close()

        vel_np = StaggeredGrid(vel_out, velocity.box)
        p_np = CenteredGrid(p_out, pressure.box)

np.savez(sub_path+'/performance_'+socket.gethostname(), np.array(performance))
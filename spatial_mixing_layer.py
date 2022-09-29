from diffpiso import *
base_path = '../learnedTurbulenceModelling_data/' # set directory where simulation will be stored
# SIMULATION SETUP ----------------------------------------------------------------------------------------
physical_parameters   = {'average_velocity':        1,
                         'velocity_difference':     1,
                         'inlet_profile_sharpness': 2,
                         'viscosity':               .002}

simulation_parameters = {'HRres':                   [128,128*4],
                         'dx_ratio':                1,
                         'dt':                      .2,
                         'dt_ratio':                1,
                         'box':                     box[0:64,0:64*4],
                         'sponge_ratio':            .875,
                         'placeholder_update':      lambda dv, tf_pl: update_dirichlet_values(dv,((False, False), (True, False)),tf_pl),
                         'relative_sponge_max':     20}

domain, sim_physics, pressure_solver, velocity_placeholder, velocity, pressure_placeholder, pressure, viscosity_field, bc_placeholders, bcx = \
    spatialMixingLayer_setup(simulation_parameters, 1e-8, physical_parameters, 1)

perturbation_amp = [0.082,0.018]
length = 50000

def boundary_perturbation_fun(shape, time, perturbation_amplitudes): # According to J. Ko et al.: Sensitivity of two-dimensional spatial mixing layer
    y_disc = np.linspace(0, domain.box.size[0], domain.resolution[0] + 2) - domain.box.half_size[0]
    eps = [perturbation_amplitudes[0]*physical_parameters['average_velocity'],
           perturbation_amplitudes[1]*physical_parameters['average_velocity']]
    n = [.4*np.pi, .3*np.pi]
    omeg = [.22,.11]
    u_perturb = np.sum([eps[i]*np.cos(n[i]*y_disc)*(1-np.tanh(y_disc*2)**2)*np.sin(omeg[i]*time)
                          for i in range(len(eps))], axis=0)
    return math.reshape(u_perturb, shape)

save_path = create_base_dir(base_path,'/mixingLayer_HRdata_pert'+"{:.3f}".format(perturbation_amp[0])+'-'+"{:.3f}".format(perturbation_amp[1])+
                            '_'+str(domain.resolution[0])+'-'+str(domain.resolution[1])+'_')
save_source(__file__, save_path, '/src.py')

warn = tf.zeros(shape=(1,), dtype=tf.bool)
dirichlet_placeholder_update = lambda dv, tf_pl: update_dirichlet_values(dv,((False, False), (True, False)),tf_pl)
velocity_all_steps, pressure_all_steps, nn_all_steps, velnew, pnew, NN_out, warn, velocity_all_arrays, pressure_all_arrays = \
    run_piso_steps(velocity, pressure, domain, physical_parameters, simulation_parameters, None, None,None,
                   sim_physics, viscosity_field, bcx, bc_placeholders,
                   dirichlet_placeholder_update=dirichlet_placeholder_update)
velnew_data = velnew.staggered_tensor()
pnew_data = pnew.data

tf.Graph.finalize(tf.get_default_graph())
# SIMULATION RUN -------------------------------------------------------------------------------------------------
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
with tf.Session(config=session_config) as sess:
    os.mkdir(save_path+'/plots')
    counter = 0
    initial_vel_x = np.ones(velocity.data[1].data.shape)*bcx[:,1:-1,...]
    initial_vel_y = np.zeros(velocity.data[0].data.shape)
    initial_pre = np.zeros(pressure.data.shape)
    vel_np = StaggeredGrid(stack_staggered_components([initial_vel_y, initial_vel_x]),
                           box=velocity.box, extrapolation=velocity.extrapolation)
    p_np = CenteredGrid(initial_pre, box=pressure.box, extrapolation=pressure.extrapolation)
    np.savez(save_path + '/velocity_' + str(counter).zfill(6), vel_np.staggered_tensor())
    np.savez(save_path + '/pressure_' + str(counter).zfill(6), p_np.data)
    counter += 1

    for i in range(int(length*8/simulation_parameters['dt_ratio'])):

        # BOUNDARY CONDITION - PERTURBATION -----------------------------------------------------------------------
        boundary_perturbation = boundary_perturbation_fun(bc_placeholders.shape,
                                                          simulation_parameters['dt']*simulation_parameters['dt_ratio']*i,
                                                          perturbation_amp)
        vel_out, p_out = sess.run([velnew_data, pnew_data],
                                  feed_dict={velocity_placeholder: vel_np.staggered_tensor(),
                                             pressure_placeholder: p_np.data,
                                             bc_placeholders: boundary_perturbation})
        np.savez(save_path + '/velocity_' + str(counter).zfill(6), vel_out)
        np.savez(save_path + '/pressure_' + str(counter).zfill(6), p_out)

        if counter%50==0:
            plt.subplot(3, 1, 1)
            plt.imshow(vel_out[0, ..., 0])
            plt.colorbar()
            plt.subplot(3, 1, 2)
            plt.imshow(vel_out[0, ..., 1])
            plt.colorbar()
            plt.subplot(3, 1, 3)
            plt.imshow(p_out[0, ..., 0])
            plt.savefig(save_path+'/plots/plt_'+str(counter))
            plt.close()

        counter += 1
        vel_np = StaggeredGrid(vel_out, velocity.box)
        p_np = CenteredGrid(p_out, pressure.box)




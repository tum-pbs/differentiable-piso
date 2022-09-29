from . import *
#from .piso_tf import *
#from .datamanagement import *
#from .losses import *
#import matplotlib.pyplot as plt

def boundary_perturbation_fun(domain, average_velocity, shape, time, perturbation_amplitudes):  # According to J. Ko et al.: Sensitivity of two-dimensional spatial mixing layer
    y_disc = np.linspace(0, domain.box.size[0], domain.resolution[0] + 2) - domain.box.half_size[0]
    eps = [perturbation_amplitudes[0] * average_velocity, perturbation_amplitudes[1] * average_velocity]
    n = [.4 * np.pi, .3 * np.pi]
    omeg = [.22, .11]
    u_perturb = np.sum([eps[i] * np.cos(n[i] * y_disc) * (1 - np.tanh(y_disc / 2) ** 2) * np.sin(omeg[i] * time)
                        for i in range(len(eps))], axis=0)
    return math.reshape(u_perturb, shape)


def print_run_info(step_count, dt_ratio, high_resolution, resolution, trainable_variables):
    print('Differentiable Physics Learning through ' + str(step_count) + ' PISO step(s)')
    print('timestep-ratio ', dt_ratio)
    print('Intermediate steps', step_count)
    print('HR: ' + str(high_resolution[0]) + ',' + str(high_resolution[1]) +
          '  LR: ' + str(resolution[0]) + ',' + str(resolution[1]))
    print('Number of trainable parameters:      ',
          np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables]))


def training_run(base_dir, physical_parameters, simulation_parameters, training_dict, solver_precision=1e-10):
    save_source(__file__, base_dir, '/src_' + os.path.basename(__file__))

    buffer_width = [[i // simulation_parameters['dx_ratio'] for i in j] for j in training_dict['HR_buffer_width']]

    if 'sponge_ratio' in simulation_parameters.keys():
        sponge_start = int(simulation_parameters['HRres'][1] * simulation_parameters['sponge_ratio']) // simulation_parameters['dx_ratio']  # //2
    else:
        sponge_start = 0

    if 'perturb_inlet' in training_dict.keys():
        perturb_inlet = training_dict['perturb_inlet']
    else:
        perturb_inlet = False

    learning_rate = training_dict['learning_rate']

    domain, sim_physics, pressure_solver, \
    velocity_placeholder, velocity, pressure_placeholder, pressure, viscosity_field, bc_placeholders, bcx_tf = \
        simulation_parameters['setup_fun'](simulation_parameters, solver_precision, physical_parameters, training_dict['step_count'])

    target_velocity = tf.placeholder(dtype=tf.float32, shape=[1, training_dict['step_count'], domain.resolution[0] + 1, domain.resolution[1] + 1, 2])
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32)

    neural_network, weights, loss_buffer_width = training_dict['network_initialiser'](buffer_width=buffer_width, padding=training_dict['padding'])
    saver = tf.train.Saver(weights, max_to_keep=training_dict['epochs'] * (training_dict['store_interm_ckpts'] + 2))

    velocity_all_steps, pressure_all_steps, nn_all_steps, velnew, pnew, NN_out, warn, velocity_all_arrays, pressure_all_arrays = \
        run_piso_steps(velocity, pressure, domain, physical_parameters, simulation_parameters, training_dict, neural_network, training_dict['network_wrapper'],
                       sim_physics, viscosity_field, bcx_tf, bc_placeholders, simulation_parameters['placeholder_update'], loss_buffer_width)
    nn_all_arrays = nn_all_steps

    # LOSS DEFINITION ---------------------------------------------------------------------------------------------
    if training_dict['sum_steps']:
        loss = 0
    else:
        loss = [0 for i in range(training_dict['step_count'])]
    contributions = []
    for l in range(len(training_dict['loss_functions'])):
        loss_function = training_dict['loss_functions'][l]
        loss_factor = training_dict['loss_factor'][l]
        loss, contrib = loss_function(loss, [velocity_all_steps], [target_velocity], training_dict['step_count'], loss_buffer_width,
                                      loss_factor, sponge_start, sum_steps=training_dict['sum_steps'], loss_influence_range=training_dict['loss_influence_range'])
        contributions.append(tf.reduce_sum(contrib))
    total_loss = tf.reduce_sum(loss)
    contributions = tf.convert_to_tensor(contributions)

    # FILTER GRADIENTS WITHOUT TF.COND
    if not training_dict['sum_steps']:
        grads = [tf.gradients(loss[i], nn_all_steps[i]) for i in range(training_dict['step_count'])]
        grads = [tf.gradients(nn_all_steps[i], tf.trainable_variables(), grads[i]) for i in range(training_dict['step_count'])]
    else:
        grads = tf.gradients(loss, tf.trainable_variables())

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
    grads_pl = [tf.placeholder(dtype=tf.float32, shape=v.shape) for v in tf.trainable_variables()]
    train_op = optimizer.apply_gradients(zip(grads_pl, tf.trainable_variables()))
    weight_norm_sum = tf.add_n([tf.norm(v) for v in tf.trainable_variables()])

    # Output for plots after validation
    plt_outputs = [tf.gradients(loss, nn_all_arrays[0])[0], tf.gradients(loss, nn_all_arrays[-1])[0],
                   nn_all_arrays[0], nn_all_arrays[-1], velnew.staggered_tensor(), pnew.data]

    init = tf.global_variables_initializer()
    adam_reinit = tf.variables_initializer(optimizer.variables())
    bytesInUse = tf.contrib.memory_stats.BytesInUse()
    tf.Graph.finalize(tf.get_default_graph())

    # DATAMANAGEMENT ---------------------------------------------------------------------------------------------
    dataGraph = tf.Graph()
    with dataGraph.as_default() as g:
        with g.name_scope("dataG") as g_scope:
            start_frames = training_dict['start_frame']
            frame_count = training_dict['frame_count_training']
            frame_count_test = training_dict['frame_count_validation']
            if 'dataset_characteristics' in training_dict and training_dict['dataset_characteristics'] is not None:
                characteristics = []
                for f in range(len(frame_count)):
                    if 'perturbation_temporal_offset' in training_dict:
                        offset = training_dict['perturbation_temporal_offset'][f]
                    else:
                        offset = 0
                    characteristics.append([(i*simulation_parameters['dt']+offset,) + training_dict['dataset_characteristics'][f]
                                            for i in range(start_frames[f], start_frames[f] + frame_count[f])])
            else:
                characteristics = [range(start_frames[f], start_frames[f] + frame_count[f])
                                   for f in range(len(frame_count))]

            field_names = ['velocity', 'pressure']
            train_tuple = data_path_assembler(training_dict['dataset'], field_names, characteristics, start_frame=start_frames,
                                              frame_count=frame_count, step_count=[training_dict['step_count'] for i in range(len(start_frames))], dt_ratio=simulation_parameters['dt_ratio'])

            test_tuple = data_path_assembler(training_dict['dataset'], field_names, characteristics,
                                             start_frame=[start_frames[f] + frame_count[f] for f in range(len(frame_count))],
                                             frame_count=frame_count_test, step_count=[training_dict['step_count'] for i in range(len(start_frames))], dt_ratio=simulation_parameters['dt_ratio'])

            train_dataset = make_tf_dataset(train_tuple, load_function_wrapper, batch_size=1, shuffle=True, prefetch_size=2)
            train_iterator = train_dataset.make_initializable_iterator()
            train_init_op = train_iterator.initializer
            train_next = train_iterator.get_next()

            test_dataset = make_tf_dataset(test_tuple, load_function_wrapper, batch_size=1, shuffle=False, prefetch_size=2)
            test_iterator = test_dataset.make_initializable_iterator()
            test_init_op = test_iterator.initializer
            test_next = test_iterator.get_next()

    loss_log = open(base_dir + "/loss.log", 'w')
    old_stdout = sys.stdout
    #sys.stdout = loss_log

    print_run_info(training_dict['step_count'], simulation_parameters['dt_ratio'], simulation_parameters['HRres'], domain.resolution, tf.trainable_variables())

    loss_history = np.zeros(training_dict['epochs'] * (sum(frame_count) - training_dict['step_count']))
    loss_history_test = np.zeros(training_dict['epochs'] * (sum(frame_count_test) - training_dict['step_count']))
    model_l2_losses = []
    model_descriptors = []

    # LEARNING RUN -------------------------------------------------------------------------------------------------
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    restarted = False

    sess = tf.Session(graph=tf.get_default_graph(), config=session_config)
    dataSess = tf.Session(graph=dataGraph, config=session_config)
    sess.run(init)
    if training_dict['load_model_path'] is not None:
        saver.restore(sess, training_dict['load_model_path'])
        print('using loaded model ' + training_dict['load_model_path'])

    writer = tf.summary.FileWriter("output", sess.graph)
    writer.close()
    for e in range(training_dict['epochs']):
        # TRAINING RUN THROUGH DATASET ---------------------------------------------------------------------------------
        dataSess.run(train_init_op)
        if e == 0:
            iterations = range(training_dict['start_first_epoch_at'], sum(frame_count) - len(frame_count) * training_dict['step_count'] * simulation_parameters['dt_ratio'])
        else:
            iterations = range(0, sum(frame_count) - len(frame_count) * training_dict['step_count'] * simulation_parameters['dt_ratio'])

        for i in iterations:
            velocity_data, pressure_data, characs = dataSess.run(train_next)
            characs = characs[0]
            data_time = characs[0]
            coarse_velocity = StaggeredGrid(velocity_data[:, 0, ...], velocity.box).at(velocity)
            coarse_pressure = CenteredGrid(pressure_data[:, 0, ...], pressure.box).at(pressure)

            current_target_vel = [np.expand_dims(StaggeredGrid(velocity_data[:, s, ...], velocity.box).at(velocity).staggered_tensor(), 1)
                                  for s in range(1, training_dict['step_count'] + 1)]
            current_target_vel = np.concatenate(current_target_vel, 1)

            feed_dict = {velocity_placeholder: coarse_velocity.staggered_tensor(),
                         pressure_placeholder: coarse_pressure.data,
                         target_velocity: current_target_vel,
                         learning_rate_placeholder: learning_rate}

            if perturb_inlet:
                boundary_perturbation = np.array([boundary_perturbation_fun(domain, physical_parameters['average_velocity'], bc_placeholders.shape[1:],
                                                                            data_time + simulation_parameters['dt_ratio'] * t*simulation_parameters['dt'], characs[1:])
                                                  for t in range(training_dict['step_count'])])
                feed_dict[bc_placeholders] = boundary_perturbation

            grads_out, loss_out, BIU, contribs_out, linsolve_warning = sess.run([grads, total_loss, bytesInUse, contributions, warn], feed_dict=feed_dict)
            linsolve_warning = any([any(l) for l in linsolve_warning])
            if linsolve_warning == 0:
                restarted = False
                if i % 100 == 0:
                    saver.save(sess, base_dir + '/model_last_working')
                    np.savez(base_dir + '/training_loss_progression', loss_history)
                if not any([np.isnan(e).any() for e in grads_out]):
                    grad_dict = {learning_rate_placeholder: learning_rate}
                    grad_dict.update(dict(zip(grads_pl, grads_out)))
                    _, wns_out = sess.run([train_op, weight_norm_sum], feed_dict=grad_dict)
            else:
                if restarted:
                    sess.close()
                    tf.reset_default_graph()
                    domain, sim_physics, pressure_solver, \
                    velocity_placeholder, velocity, pressure_placeholder, pressure, viscosity_field, bc_placeholders, bcx_tf = \
                        simulation_parameters['setup_fun'](simulation_parameters, solver_precision, physical_parameters, training_dict['step_count'])

                    target_velocity = tf.placeholder(dtype=tf.float32, shape=[1, training_dict['step_count'], domain.resolution[0] + 1, domain.resolution[1] + 1, 2])
                    learning_rate_placeholder = tf.placeholder(dtype=tf.float32)

                    neural_network, weights, loss_buffer_width = training_dict['network_initialiser'](buffer_width=buffer_width, padding=training_dict['padding'])
                    saver = tf.train.Saver(weights, max_to_keep=training_dict['epochs'] * (training_dict['store_interm_ckpts'] + 2))

                    velocity_all_steps, pressure_all_steps, nn_all_steps, velnew, pnew, NN_out, warn, velocity_all_arrays, pressure_all_arrays = \
                        run_piso_steps(velocity, pressure, domain, physical_parameters, simulation_parameters, training_dict, neural_network, training_dict['network_wrapper'],
                                       sim_physics, viscosity_field, bcx_tf, bc_placeholders, simulation_parameters['placeholder_update'], loss_buffer_width)
                    # LOSS DEFINITION ---------------------------------------------------------------------------------------------
                    if training_dict['sum_steps']:
                        loss = 0
                    else:
                        loss = [0 for i in range(training_dict['step_count'])]
                    contributions = []
                    for l in range(len(training_dict['loss_functions'])):
                        loss_function = training_dict['loss_functions'][l]
                        loss_factor = training_dict['loss_factor'][l]
                        loss, contrib = loss_function(loss, [velocity_all_steps], [target_velocity], training_dict['step_count'], loss_buffer_width,
                                                      loss_factor, sponge_start, sum_steps=training_dict['sum_steps'],
                                                      loss_influence_range=training_dict['loss_influence_range'])  # , fluid.velocity.box)
                        contributions.append(tf.reduce_sum(contrib))
                    total_loss = tf.reduce_sum(loss)
                    contributions = tf.convert_to_tensor(contributions)

                    if not training_dict['sum_steps']:
                        grads = [tf.gradients(loss[i], nn_all_steps[i]) for i in range(training_dict['step_count'])]
                        grads = math.sum([tf.gradients(nn_all_steps[i], tf.trainable_variables(), grads[i]) for i in range(training_dict['step_count'])], axis=0)
                    else:
                        grads = tf.gradients(loss, tf.trainable_variables())
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
                    grads_pl = [tf.placeholder(dtype=tf.float32, shape=v.shape) for v in tf.trainable_variables()]
                    train_op = optimizer.apply_gradients(zip(grads_pl, tf.trainable_variables()))
                    weight_norm_sum = tf.add_n([tf.norm(v) for v in tf.trainable_variables()])

                    # SOME OUTPUT FOR PLOTS AFTER EACH VALIDATION RUN
                    plt_outputs = [tf.gradients(loss, nn_all_arrays[training_dict['grad_stop']])[0], tf.gradients(loss, nn_all_arrays[-1])[0],
                                   nn_all_arrays[0], nn_all_arrays[-1], velnew.staggered_tensor(), pnew.data]

                    init = tf.global_variables_initializer()
                    adam_reinit = tf.variables_initializer(optimizer.variables())
                    bytesInUse = tf.contrib.memory_stats.BytesInUse()
                    tf.Graph.finalize(tf.get_default_graph())
                    sess = tf.Session()
                    sess.run(init)
                    saver.restore(sess, base_dir + '/model_epoch_' + model_descriptors[-1] + '.ckpt')
                else:
                    print("RESTARTING FROM LAST WORKING")
                    saver.restore(sess, base_dir + '/model_last_working')
                    sess.run(adam_reinit)
                restarted = True
                loss_out = -1

            print('epoch ' + str(e) + '  iteration ' + str(i) + '  loss: ' + str(loss_out) + ' warn:' + str(linsolve_warning) + ' bytes: ' +
                  str(BIU) + ' wns: ' + str(wns_out) + '  loss_contribs', contribs_out)
            loss_history[(e * (sum(frame_count) - training_dict['step_count']) + i)] = loss_out
            if True & (i % (len(iterations) // training_dict['store_interm_ckpts']) == 0) & (i > 0):
                saver.save(sess, base_dir + '/model_epoch_' + str(e).zfill(6) + 'i' + str(i).zfill(6) + '.ckpt')

                starting_frame = training_dict['start_frame'][0]
                timesteps = training_dict['interm_forward_steps']
                initial_vel = np.load(training_dict['dataset'][0] + 'velocity_' + str(starting_frame).zfill(6) + '.npz')['arr_0']
                initial_pre = np.load(training_dict['dataset'][0] + 'pressure_' + str(starting_frame).zfill(6) + '.npz')['arr_0']
                vel_np = StaggeredGrid(initial_vel, velocity.box).at(velocity)
                p_np = CenteredGrid(initial_pre, pressure.box).at(pressure)

                target_vel = np.load(training_dict['dataset'][0] + 'velocity_' + str(timesteps * simulation_parameters['dx_ratio'] + starting_frame).zfill(6) + '.npz')['arr_0']
                target_vel = StaggeredGrid(target_vel, velocity.box).at(velocity).staggered_tensor()

                for c in range(timesteps):
                    feed_dict = {velocity_placeholder: vel_np.staggered_tensor(),
                                 pressure_placeholder: p_np.data}
                    if perturb_inlet:
                        time_c = starting_frame * simulation_parameters['dt'] + simulation_parameters['dt'] * simulation_parameters['dt_ratio'] * c
                        if 'perturbation_temporal_offset' in training_dict:
                            time_c += training_dict['perturbation_temporal_offset'][0]
                        boundary_perturbation = \
                            np.array([boundary_perturbation_fun(domain, physical_parameters['average_velocity'], bc_placeholders.shape[1:],
                                                                time_c + simulation_parameters['dt'] * simulation_parameters['dt_ratio'] * t,
                                                                training_dict['dataset_characteristics'][0]) for t in range(training_dict['step_count'])])
                        feed_dict[bc_placeholders] = boundary_perturbation

                    vel_out, p_out = sess.run([velocity_all_arrays[0], pressure_all_arrays[0]], feed_dict=feed_dict)
                    vel_np = StaggeredGrid(vel_out, velocity.box)
                    p_np = CenteredGrid(p_out, pressure.box)

                l2 = np.sum((target_vel - vel_np.staggered_tensor()) ** 2)
                model_l2_losses.append(l2)
                model_descriptors.append(str(e).zfill(6) + 'i' + str(i).zfill(6))
                plt.bar(model_descriptors, model_l2_losses)
                plt.title('Model comp after ' + str(timesteps) + ' timesteps')
                plt.xticks(rotation='vertical')
                plt.savefig(base_dir + '/modelComp_t' + str(timesteps) + '_' + model_descriptors[-1] + '.png')
                plt.close()
                if len(model_l2_losses) > 2:
                    if model_l2_losses[-1] > 20 * model_l2_losses[-2]:
                        saver.restore(sess, base_dir + '/model_epoch_' + model_descriptors[-2] + '.ckpt')

        # VALIDATION RUN  ----------------------------------------------------------------------------------------------
        dataSess.run(test_init_op)

        for i in range(sum(frame_count_test) - len(frame_count_test) * training_dict['step_count'] * simulation_parameters['dt_ratio']):
            velocity_data, pressure_data, characs = dataSess.run(test_next)
            characs = characs[0]
            data_time = characs[0]
            coarse_velocity = StaggeredGrid(velocity_data[:, 0, ...], velocity.box).at(velocity)
            coarse_pressure = CenteredGrid(pressure_data[:, 0, ...], pressure.box).at(pressure)

            current_target_vel = \
                [np.expand_dims(
                    StaggeredGrid(velocity_data[:, s, ...], velocity.box).at(velocity).staggered_tensor(),
                    1) for s in range(1, training_dict['step_count'] + 1)]
            current_target_vel = np.concatenate(current_target_vel, 1)

            feed_dict = {velocity_placeholder: coarse_velocity.staggered_tensor(),
                         pressure_placeholder: coarse_pressure.data,
                         target_velocity: current_target_vel,
                         learning_rate_placeholder: learning_rate}
            if perturb_inlet:
                boundary_perturbation = \
                    np.array([boundary_perturbation_fun(domain, physical_parameters['average_velocity'], bc_placeholders.shape[1:],
                                                        data_time + simulation_parameters['dt_ratio'] * t * simulation_parameters['dt'],
                                                        characs[1:]) for t in range(training_dict['step_count'])])
                feed_dict[bc_placeholders] = boundary_perturbation

            loss_out = sess.run([total_loss], feed_dict=feed_dict)

            print('epoch ' + str(e) + '  validation ' + str(i) + '  validation_loss: ' + str(loss_out))
            loss_history_test[(e * (sum(frame_count_test) - training_dict['step_count']) + i)] = loss_out[0]

            grad_out1, grad_out2, nn_for_start, nn_for_end, vel_out, p_out = sess.run(plt_outputs, feed_dict=feed_dict)
            fig = plt.figure(figsize=(6, 18))
            plt.title('Flow after epoch ' + str(e))
            plt.subplot('611')
            plt.title("NN force s=0")
            plt.imshow(nn_for_start[0, ..., 0] ** 2 + nn_for_start[0, ..., 1] ** 2)
            plt.colorbar()
            plt.subplot('612')
            plt.title("NN force s=-1")
            plt.imshow(nn_for_end[0, ..., 0] ** 2 + nn_for_end[0, ..., 1] ** 2)
            plt.colorbar()
            plt.subplot('613')
            plt.title("v velocity s=0")
            plt.imshow(vel_out[0, ..., 0])
            plt.colorbar()
            plt.subplot('614')
            plt.title("u velocity s=0")
            plt.imshow(vel_out[0, ..., 1])
            plt.colorbar()
            plt.subplot('615')
            plt.title("gradient s=0")
            plt.imshow(grad_out1[0, ..., 0] ** 2 + grad_out1[0, ..., 1] ** 2)
            plt.colorbar()
            plt.subplot('616')
            plt.title("gradient s=-1")
            plt.imshow(grad_out2[0, ..., 0] ** 2 + grad_out2[0, ..., 1] ** 2)
            plt.colorbar()

            plt.savefig(base_dir + '/plot_iteration_' + str(e).zfill(6))
            plt.close()

        # Now saving
        saver.save(sess, base_dir + '/model_epoch_' + str(e).zfill(6) + '.ckpt')
        if training_dict['lr_decay_fun'] is not None:
            learning_rate = training_dict['lr_decay_fun'](learning_rate)

    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('loss_value')
    plt.savefig(base_dir + '/loss_plot.png')
    np.savez(base_dir + '/training_loss_progression', loss_history)
    np.savez(base_dir + '/validation_loss_progression', loss_history_test)

    tf.reset_default_graph()
    dataSess.close()
    sess.close()


## Workaround function, can be used instead of stop_gradient, prevents parallel execution of gradients in sub-ranges (which are problematic for linear solves)
@tf.custom_gradient
def zero_gradient_op(centered_data):
    def grad(dc):
        return dc * 0

    return centered_data, grad


def run_piso_steps(velocity, pressure, domain, physical_parameters, simulation_parameters, training_dict, neural_network, neural_network_wrapper,
                   sim_physics, viscosity_field, bcx, bc_placeholders, dirichlet_placeholder_update=None, loss_buffer_width=None):
    # Do first step
    if neural_network is not None:
        buffer_width = [[i // simulation_parameters['dx_ratio'] for i in j] for j in training_dict['HR_buffer_width']]
        NN_in = velocity.at_centers().data
        if training_dict['pressure_included']:
            NN_in = tf.concat([NN_in, pressure.gradient().data], -1)

        NN_out = neural_network_wrapper(neural_network, NN_in, domain, physical_parameters, simulation_parameters, loss_buffer_width, buffer_width)

        residual_force = StaggeredGrid(
            [CenteredGrid(math.expand_dims(NN_out[..., 0], -1), velocity.box).at(velocity.data[0]).data,
             CenteredGrid(math.expand_dims(NN_out[..., 1], -1), velocity.box).at(velocity.data[1]).data],
            velocity.box).staggered_tensor()
        nn_all_steps = [NN_out]

    else:
        residual_force = None

    step_count = training_dict['step_count'] if training_dict is not None else 1
    warn = [None] * step_count
    dt = simulation_parameters['dt'] * simulation_parameters['dt_ratio']
    dirichlet_values = sim_physics.dirichlet_values
    pressure_inc1 = CenteredGrid(tf.zeros_like(pressure.data) + 5e-13, pressure.box)
    pressure_inc2 = CenteredGrid(tf.zeros_like(pressure.data) + 1e-12, pressure.box)
    vel_piso, p_piso, warn[0] = piso_step(velocity, pressure, pressure_inc1, pressure_inc2, dt, sim_physics, dirichlet_values,
                                          differentiation_scheme='central_difference_new', viscosity_field=viscosity_field,
                                          forcing_term=residual_force, unrolling_step=0, warn=tf.zeros(shape=(1,), dtype=tf.bool))
    velocity_all_steps = [vel_piso]
    pressure_all_steps = [p_piso]
    vel_tensor = vel_piso.staggered_tensor()
    p_tensor = p_piso.data
    velocity_all_arrays = [vel_tensor]
    pressure_all_arrays = [p_tensor]
    velnew = StaggeredGrid(velocity_all_arrays[0], vel_piso.box, vel_piso.extrapolation)
    pnew = CenteredGrid(pressure_all_arrays[0], p_piso.box, p_piso.extrapolation)

    # Remaining steps
    for i in range(step_count - 1):
        if (i + 1) % training_dict['loss_influence_range'] == 0:
            velnew = StaggeredGrid(tf.stop_gradient(velnew.staggered_tensor()), velnew.box, velnew.extrapolation)
            pnew = CenteredGrid(zero_gradient_op(pnew.data), pnew.box, pnew.extrapolation)

        if dirichlet_placeholder_update is not None:
            dirichlet_values = dirichlet_placeholder_update(sim_physics.dirichlet_values, (([], []), (tf.constant(bcx, dtype=bc_placeholders[i + 1].dtype) + bc_placeholders[i + 1], [])))

        if neural_network is not None:
            NN_in = velnew.at_centers().data
            if training_dict['pressure_included']:
                NN_in = tf.concat([NN_in, pnew.gradient().data], -1)

            NN_out = neural_network_wrapper(neural_network, NN_in, domain, physical_parameters, simulation_parameters, loss_buffer_width, buffer_width)

            residual_force = StaggeredGrid(
                [CenteredGrid(math.expand_dims(NN_out[..., 0], -1), velocity.box).at(velocity.data[0]).data,
                 CenteredGrid(math.expand_dims(NN_out[..., 1], -1), velocity.box).at(velocity.data[1]).data],
                velocity.box).staggered_tensor()
            nn_all_steps.append(NN_out)
        else:
            residual_force = None

        pressure_inc1 = CenteredGrid(tf.zeros_like(pressure.data) + 5e-13, pressure.box)
        pressure_inc2 = CenteredGrid(tf.zeros_like(pressure.data) + 1e-12, pressure.box)

        vel_piso, p_piso, warn[i + 1] = piso_step(velnew, pnew, pressure_inc1, pressure_inc2, dt, sim_physics, dirichlet_values,
                                                  differentiation_scheme='central_difference_new', viscosity_field=viscosity_field,
                                                  forcing_term=residual_force, unrolling_step=i + 1, warn=tf.zeros(shape=(1,), dtype=tf.bool))
        velocity_all_steps.append(vel_piso)
        pressure_all_steps.append(p_piso)
        vel_tensor = vel_piso.staggered_tensor()
        p_tensor = p_piso.data
        velocity_all_arrays.append(vel_tensor)
        pressure_all_arrays.append(p_tensor)
        velnew = StaggeredGrid(velocity_all_arrays[i + 1], vel_piso.box, vel_piso.extrapolation)
        pnew = CenteredGrid(pressure_all_arrays[i + 1], p_piso.box, p_piso.extrapolation)

        velocity, pressure = velnew, pnew

    if neural_network is None:
        nn_all_steps = []
        NN_out = []
    return velocity_all_steps, pressure_all_steps, nn_all_steps, velnew, pnew, NN_out, warn, velocity_all_arrays, pressure_all_arrays


def spatialMixingLayer_setup(simulation_parameters, solver_precision, physical_parameters, step_count):
    HRres = simulation_parameters['HRres']
    dx_ratio = simulation_parameters['dx_ratio']
    box = simulation_parameters['box']
    boundary_bool = ((True, True), (True, False))

    pressure_solver = PisoPressureSolverCudaCustom(accuracy=solver_precision, max_iterations=10000, dx=[], residual_reset=1000, randomized_restarts=0, cast_to_double=True)
    linear_solver = LinearSolverCudaMultiBicgstabILU(accuracy=solver_precision, max_iterations=10000, cast_to_double=False)
    domain = Domain([int(HRres[0] / dx_ratio), int(HRres[1] / dx_ratio)], box=box, boundaries=((OPEN, OPEN), (OPEN, CLOSED)))

    average_velocity = physical_parameters['average_velocity']
    velocity_difference = physical_parameters['velocity_difference']
    sharpness = physical_parameters['inlet_profile_sharpness']
    sponge_start = int(HRres[1] * simulation_parameters['sponge_ratio'] / dx_ratio)  # //2

    sponge_max = physical_parameters['viscosity'] * simulation_parameters['relative_sponge_max']
    inlet_profile = velocity_difference / 2 * np.tanh(sharpness * (
            np.linspace(0, domain.box.size[0], domain.resolution[0] + 2) - domain.box.half_size[
        0])) + average_velocity

    bcx = np.reshape(inlet_profile, (1, domain.resolution[0] + 2, 1, 1))
    bc_placeholders = tf.placeholder(dtype=tf.float32, shape=(step_count,) + bcx.shape)
    bcx_tf = tf.constant(bcx, dtype=tf.float32)
    bcxout = []

    staggered_shape = domain.staggered_grid(1).staggered_tensor().shape
    centered_shape = domain.centered_grid(1).data.shape

    bcy = np.zeros((1, 1, domain.resolution[1] + 2, 1))
    boundary_array = ((bcy, bcy), (bcx_tf + bc_placeholders[0], bcxout))
    dirichlet_mask, dirichlet_values, neumann_mask, active_mask, accessible_mask = \
        compute_mixingLayer_masks(staggered_shape, boundary_bool, boundary_array)

    pressure_solver.dx = domain.dx[0]
    pressure_solver.neumann_BC = boundary_bool
    pressure_solver.active_mask = active_mask
    pressure_solver.accessible_mask = accessible_mask

    # PLACEHOLDER DEFINITION ---------------------------------------------------------------------------------------
    velocity_placeholder = placeholder(shape=staggered_shape, dtype=tf.float32,
                                       basename='velocity_placeholder')
    velocity = StaggeredGrid.sample(velocity_placeholder, domain=domain)
    pressure_placeholder = placeholder(shape=centered_shape, dtype=tf.float32, basename='pressure_placeholder')
    pressure = CenteredGrid(pressure_placeholder, box=domain.box, extrapolation=pressure_extrapolation(domain.boundaries))

    viscosity = np.ones(pressure.data.shape) * physical_parameters['viscosity']
    viscosity[:, :, sponge_start:, :] += \
        np.expand_dims(np.matmul(np.ones((domain.resolution[0], 1)),
                                 np.expand_dims(np.linspace(0, sponge_max, domain.resolution[1] - sponge_start), 0)),
                       (0, -1))
    viscosity = flatten_staggered_data(CenteredGrid(viscosity, pressure.box).at(velocity), coord_flip=True)
    viscosity_field = tf.constant(viscosity)

    sim_physics = SimulationParameters(dirichlet_mask=dirichlet_mask.astype(bool), dirichlet_values=dirichlet_values,
                                       active_mask=active_mask, accessible_mask=accessible_mask, bool_periodic=(False, False),
                                       no_slip_mask=np.zeros_like(dirichlet_mask, dtype=np.bool), viscosity=viscosity_field,
                                       linear_solver=linear_solver, pressure_solver=pressure_solver)
    return domain, sim_physics, pressure_solver, \
           velocity_placeholder, velocity, pressure_placeholder, pressure, viscosity_field, bc_placeholders, bcx

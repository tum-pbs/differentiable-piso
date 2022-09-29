import sys
from diffpiso import *

base_path = '../learnedTurbulenceModelling_data/spatialMixingLayer/'  # Set path to directories containing the dataset

def neural_network_wrapper(neural_network, input, fluid, physical_parameters, simulation_parameters, loss_buffer_width, buffer_width):
    sponge_start = int(simulation_parameters['HRres'][1] * simulation_parameters['sponge_ratio']) // simulation_parameters['dx_ratio']
    NN_in = input[:, :, :sponge_start, :]
    NN_out = tf.pad(neural_network(NN_in), ((0, 0), (0, 0), (0, fluid.resolution[1] - sponge_start), (0, 0)))
    return NN_out

initialiser = None # weight initializer can be changed, e.g. tf.random_normal_initializer(stddev=.01, mean=.0), default is Glorot initialisation

tf.random.set_random_seed(42)
physical_parameters   = {'average_velocity':        1,                                  # average mixing layer velocity
                         'velocity_difference':     1,                                  # velocity difference
                         'inlet_profile_sharpness': 2,                                  # sharpness of the tanh profile used at the inlet boundary
                         'viscosity':               .002}

simulation_parameters = {'HRres':                   [64,64*4],                          # resolution of the dataset (our uploaded dataset is already downsampled by 8 from 512 to 64)
                         'dx_ratio':                1,                                  # downsampling ratio w.r.t. dataset (uploaded dataset is already downsampled)
                         'dt':                      .05*8,                              # timestep of the dataset (our uploaded dataset is already temporally coarsed by 8)
                         'dt_ratio':                1,                                  # temporal coarsening w.r.t the dataset
                         'box':                     box[0:64,0:64*4],                   # physical box size (see PhiFlow)
                         'sponge_ratio':            .875,                               # spongelayer starting point relative to total size in x
                         'relative_sponge_max':     20,                                 # factor of songe layer viscosity
                         'placeholder_update':      lambda dv, tf_pl: update_dirichlet_values(dv,((False, False), (True, False)),tf_pl),
                         'setup_fun':               spatialMixingLayer_setup}

training_dict =         {'step_count':              10,                                 # number of unrolled forward steps
                         'epochs':                  2,                                  # number of epochs
                         'dataset':                 [base_path + '/sml_HR_512-2048_dx8_dt8_pert0.050-0.050/',
                                                     base_path + '/sml_HR_512-2048_dx8_dt8_pert0.075-0.025/',
                                                     base_path + '/sml_HR_512-2048_dx8_dt8_pert0.025-0.075/',
                                                     base_path + '/sml_HR_512-2048_dx8_dt8_pert0.040-0.060/',
                                                     base_path + '/sml_HR_512-2048_dx8_dt8_pert0.060-0.040/',
                                                     ],
                         'start_frame':             [0,0,0,0,0],                        # first timeframe used in training
                         'frame_count_training':    [200,200,200,200,200],              # number of timeframes used in training
                         'frame_count_validation':  [100,100,100,100,100],              # number of timeframes used for validation (at the end of dataset)
                         'dataset_characteristics': [(0.05,0.05),(0.075,0.025),         # characteristic perturbation frequencies of the datasets
                                                     (0.025,0.075),(0.040,0.060),
                                                     (0.060,0.040)],
                         'perturb_inlet':           True,
                         'perturbation_temporal_offset': [11001 * .05  for i in range(5)], # temporal offset corresponds to start of downsampled data
                         'pressure_included':       True,
                         'network_initialiser':     lambda buffer_width, padding: initialise_fullyconv_network(buffer_width, padding, restore_shape=True, initialiser=initialiser),
                         'network_wrapper':         neural_network_wrapper,
                         'padding':                 'VALID',
                         'load_model_path':         None,
                         'loss_functions':          [L2_field_loss, spectral_energy_loss, strain_rate_loss, multistep_averaging_loss],   # list of loss functions
                         'loss_factor':             [50,0.5,2,0.5],                     # weighting of loss functions
                         'HR_buffer_width':         [[0, 0], [0, 0]],
                         'data_shuffling_seeds':    None,
                         'start_first_epoch_at':    0,
                         'learning_rate':           1e-5,
                         'lr_decay_fun':            lambda l: l*.4,
                         'store_interm_ckpts':      10,                                 # number of intermeadiate checkpoints stored per epoch
                         'interm_forward_steps':    100,                                # number of forward steps used in intermediate model evaluations
                         'sum_steps':               True,                               # loss function summation, recommended: True
                         'loss_influence_range':    10                                  # length of backpropagation range (set between 1 and step_count)
                         }


if training_dict['data_shuffling_seeds'] is None: training_dict['data_shuffling_seeds']=[None for e in range(training_dict['epochs'])]
if len(training_dict['data_shuffling_seeds']) < training_dict['epochs']: training_dict['data_shuffling_seeds'].append([None for e in range(training_dict['epochs'])])

name_add = '_'
if L2_field_loss in training_dict['loss_functions']:
    name_add +="L2"
if strain_rate_loss in training_dict['loss_functions']:
    name_add +="SR"
if spectral_energy_loss in training_dict['loss_functions']:
    name_add +="SE"
if multistep_averaging_loss in training_dict['loss_functions']:
    name_add += "MS"

name_add+= '_'+'-'.join([str(ls) for ls in training_dict['loss_factor']])

if (isinstance(initialiser, tf.random_normal_initializer) and training_dict['load_model_path'] is None) or\
        (training_dict['load_model_path'] is not None and 'normInit' in training_dict['load_model_path']):
    name_add += '_normInit'
elif training_dict['load_model_path'] is None:
    name_add += '_glorotInit'

save_path = create_base_dir(base_path, '/diffPhy_integrated_'+str(simulation_parameters['dx_ratio'])+'x_' + str(training_dict['step_count']) +
                            'step_LR_'+str(simulation_parameters['HRres'][0]//simulation_parameters['dx_ratio'])+'-'+
                            str(simulation_parameters['HRres'][1]//simulation_parameters['dx_ratio'])+name_add+'_')
save_source(__file__, save_path, '/src_' + os.path.basename(__file__))
training_run(save_path, physical_parameters, simulation_parameters, training_dict, solver_precision=1e-6)


import sys
sys.path.append('..')
from diffpiso import *

simulation_data_directory = '../lidDrivenCavity/'  #"SPECIFY YOUR TARGET DIRECTORY"

RE = 1000
N = 128
# SIMULATION SETUP ----------------------------------------------------------------------------------------
pressure_solver = PisoPressureSolverCudaCustom(accuracy=1e-8, max_iterations=1000, dx=[], cast_to_double=True)
pressure_solver.laplace_rank_deficient = tf.constant(True, dtype=tf.bool, shape=[1, ])
accuracy_placeholder = tf.placeholder(dtype=tf.float32)
linear_solver = LinearSolverCudaMultiBicgstabILU(accuracy=accuracy_placeholder, max_iterations=100, cast_to_double=False)

domain = Domain([N+1, N], box=box[0:1+1/N,0:1], boundaries=OPEN)
staggered_shape = calculate_staggered_shape(1, domain.resolution)
centered_shape = calculate_centered_shape(1, domain.resolution)

periodic_bool = (False, False)
dirichlet_mask_y = math.zeros(staggered_shape - np.array([0, 0, 1, 1]))
dirichlet_mask_y[:,0,...] = 1
dirichlet_mask_y[:,-2:,...] = 1
dirichlet_mask_x = math.zeros(staggered_shape - np.array([0, 1, 0, 1]))
dirichlet_mask_x[...,0,:] = 1
dirichlet_mask_x[...,-1,:] = 1
dirichlet_mask_x[:,-1,...] = 1
dirichlet_mask = stack_staggered_components([dirichlet_mask_y, dirichlet_mask_x])
dirichlet_values_y =math.zeros(staggered_shape - np.array([0, 0, 1, 1]))
dirichlet_values_x = math.zeros(staggered_shape - np.array([0, 1, 0, 1]))
dirichlet_values_x[:,-1,...]=1
dirichlet_values = stack_staggered_components([dirichlet_values_y, dirichlet_values_x])

accessible_mask = math.pad(math.ones(staggered_shape + np.array([0, -1, -1, -1])),(0,1,1,0),"constant")
accessible_mask[0,-2,...] = 0
active_mask = math.pad(math.ones(staggered_shape + np.array([0, -1, -1, -1])),(0,1,1,0),"constant")
active_mask[0,-2,...] = 0

no_slip_bool = np.zeros([1,staggered_shape[1]+1, staggered_shape[2]+1,1], dtype=np.bool)
no_slip_bool[0,0,:,0] = 1
no_slip_bool[0,-2:,:,0] = 1
no_slip_bool[0,:,0,0] = 1
no_slip_bool[0,:,-1,0] = 1
no_slip_bool = math.flatten(no_slip_bool)

sim_physics = SimulationParameters(dirichlet_mask=dirichlet_mask.astype(bool), dirichlet_values=dirichlet_values, active_mask= active_mask,
                                   accessible_mask= accessible_mask, bool_periodic= periodic_bool, no_slip_mask=no_slip_bool,
                                   viscosity=1/RE, linear_solver=linear_solver, pressure_solver=pressure_solver)

# PLACEHOLDER DEFINITION ---------------------------------------------------------------------------------------
dt = 0.01
velocity_placeholder = placeholder(shape=staggered_shape, dtype=tf.float32,basename='velocity_placeholder')
velocity = StaggeredGrid.sample(velocity_placeholder,domain=domain)

pressure_placeholder = placeholder(shape=centered_shape, dtype=tf.float32, basename='pressure_placeholder')
pressure = CenteredGrid(pressure_placeholder, box=domain.box, extrapolation=pressure_extrapolation(domain.boundaries))

pressure_inc1 = CenteredGrid(tf.zeros_like(pressure.data), pressure.box, pressure.extrapolation)
pressure_inc2 = CenteredGrid(tf.zeros_like(pressure.data) + 1e-12, pressure.box, pressure.extrapolation)

# SIMULATION STEP
vel_piso, pnew,_ = piso_step(velocity, pressure, pressure_inc1, pressure_inc2, dt, sim_physics, sim_physics.dirichlet_values)
pnew = pnew.data
velnew = vel_piso.staggered_tensor()

# INITIAL CONDITION ---------------------------------------------------------------------------------------
vel_np = StaggeredGrid(np.zeros(staggered_shape), velocity.box)
p_np = CenteredGrid(np.zeros(pressure.data.shape), pressure.box)


np_accuracy=1e-3
save_path = create_base_dir(simulation_data_directory,'/LDC_Re'+str(RE)+'_'+str(N)+'x'+str(N)+'_')
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)
for i in range(int(25 // dt)):
    feed_dict = {velocity_placeholder: vel_np.staggered_tensor(),
                 pressure_placeholder: p_np.data,
                 accuracy_placeholder: np_accuracy}

    vel_out, p_out = sess.run([velnew, pnew], feed_dict=feed_dict)
    vel_np = StaggeredGrid(vel_out, velocity.box)
    p_np = CenteredGrid(p_out, pressure.box)

    if i%100==0:
        rows = 2
        columns = 2
        f = plt.figure(figsize=(10,10))
        plt.subplot(rows, columns, 1)
        plt.title(r'$u$')
        plt.imshow(vel_out[0, ..., 0])
        plt.colorbar()
        plt.subplot(rows, columns, 2)
        plt.title(r'$v$')
        plt.imshow(vel_out[0, ..., 1])
        plt.colorbar()
        plt.subplot(rows, columns, 3)
        plt.title(r'$\omega$')
        plt.imshow(vorticity(vel_np)[0, :-1,:, 0])
        plt.colorbar()
        plt.subplot(rows, columns, 4)
        plt.title(r'p')
        plt.imshow(p_out[0, ..., 0])
        plt.colorbar()
        plt.savefig(save_path+'/plot_'+str(i))
        plt.close()
        np.savez(save_path+'/velocity_'+str(i).zfill(6)+'.npz', vel_out)
        np.savez(save_path+'/pressure_'+str(i).zfill(6)+'.npz', p_out)
    print('step',i)

    if i==5:
        np_accuracy = 1e-8


np.savez(save_path+'/velocity_'+str(i).zfill(6)+'.npz', vel_out)
np.savez(save_path+'/pressure_'+str(i).zfill(6)+'.npz', p_out)
print('done')
from phi.tf.flow import *
from .piso_helpers import custom_padded

def strain_tensor(velocity: StaggeredGrid):
    velocity_padded = custom_padded(velocity, 1)
    grads = [math.gradient(velocity_padded.data[i].data, velocity_padded.dx, 'forward') for i in range(len(velocity.data))]
    strain = [(grads[0][:,:-2,:-1,0] + grads[0][:,1:-1,1:,0]) / 2,
              (grads[0][:,1:-2,1:-1,1] + grads[1][:,1:-1,1:-2,0]) / 2,
              (grads[0][:,1:-2,1:-1,1] + grads[1][:,1:-1,1:-2,0]) / 2,
              (grads[1][:,:-1,:-2,1] + grads[1][:,1:, 1:-1, 1])/2]
    return strain

def strain_tensor_centered(velocity: StaggeredGrid):
    velocity_padded = custom_padded(velocity, 1)
    grads = [math.gradient(velocity_padded.data[i].data, velocity_padded.dx, 'forward') for i in range(len(velocity.data))]
    corner_box =AABox(velocity.box.lower - .5*velocity.dx,
                      velocity.box.upper + .5*velocity.dx)
    corner_data = math.expand_dims(grads[0][:, 1:-1, :-1, 1] + grads[1][:, :-1, 1:-1, 0], -1) / 2

    corner_val = CenteredGrid(corner_data,
                              box=corner_box)
    strain = [grads[0][:,1:-2,1:-1,0],
              corner_val.at(velocity.center_points).data[...,0],
              corner_val.at(velocity.center_points).data[...,0],
              grads[1][:,1:-1,1:-2,1]]
    return strain

def smagorinsky_eddy_viscosity(velocity: StaggeredGrid, smagorinsky_constant: float):
    strain_centered = strain_tensor_centered(velocity)
    strain_norm = (2 * math.sum([strain_centered[i]**2 for i in range(len(strain_centered))],0))**.5
    eddy_viscosity = (smagorinsky_constant * velocity.dx[0]**2) * strain_norm
    return math.expand_dims(eddy_viscosity,axis=-1)

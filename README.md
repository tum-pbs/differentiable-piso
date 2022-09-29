# Code repository for "Learned Turbulence Modelling with Differentiable Fluid Solvers"

This repository contains the source code for 
["Learned Turbulence Modelling with Differentiable Fluid Solvers"](http://arxiv.org/abs/2202.06988) by Bjoern List, Liwei Chen, and Nils Thuerey.

![Teaser-image](resources/learned-piso1-teaser.jpeg)

## Abstract:

In this paper, we train turbulence models based on convolutional neural networks. These learned turbulence models improve under-resolved low resolution solutions to the incompressible Navier-Stokes equations at simulation time. Our study involves the development of a differentiable numerical solver that supports the propagation of optimisation gradients through multiple solver steps. The significance of this property is demonstrated by the superior stability and accuracy of those models that unroll more solver steps during training. Furthermore, we introduce loss terms based on turbulence physics that further improve the model accuracy. This approach is applied to three two-dimensional turbulence flow scenarios, a homogeneous decaying turbulence case, a temporally evolving mixing layer, and a spatially evolving mixing layer. Our models achieve significant improvements of long-term a-posteriori statistics when compared to no-model simulations, without requiring these statistics to be directly included in the learning targets. At inference time, our proposed method also gains substantial performance improvements over similarly accurate, purely numerical methods.

[TUM](https://ge.in.tum.de/)


# Differentiable PISO solver

This implementation of the PISO method solves the incompressible Navier-Stokes equations to second order accuracy.
 Differentiability of the solver is achieved by utilising the machine learning frameworks [TensorFlow](https://www.tensorflow.org/) and [Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow).

## Installation
Required packages/software:

- CUDA 10.0

Conda:
- python=3.6
- numpy>=1.17
- matplotlib>=3.3.4
- scipy>=1.5.2

Pip:
- tensorflow_gpu==1.14

A forked and modified version of phiflow==1.4.0 is included in this repository. To install this version, run the following command in the project directory:
```
pip install ./PhiFlow
```

The CUDA custom operations (PISO and PhiFlow) can be compiled from source by calling
```python setup.py tf_cuda```.
This requires installations of CUDA's ```nvcc``` compiler and ```gcc4.8```. For convenience, we also provide the compiled binaries in this repository.

## Simulation setup

The solver supports two-dimensional flow simulations on staggered, uniform cartesian grids. As in the PISO method, a simulation step is split into a prediction step solving the implicit advection-diffusion system, followed by two pressure correction steps.

It is recommended to follow this documentation along with one the code-verification file `lidDrivenCavity_2D.py`.

Include all python function by importing
```
from diffpiso import *
```

### Simulation Parameters

The simulation parameters are stored inside an instance of `SimulationParameters`. This instance thus needs information about the boundary conditions (encoded by boundary masks), the kinematic viscosity, and the numerical solvers used for the linear systems in the implicit advection and pressure-poisson systems:

- `dirichlet_mask`: staggered shape, `bool`: `True` if velocity cell is a Dirichlet boundary
- `dirichlet_values`: staggered shape, `float32`: Value of the Dirichlet boundary
- `active_mask`: centered shape padded by 1, `float32`: 1.0 if cell lies within computed domain (i.e. 0.0 for cells outside of a Dirichlet or Neumann boundary; 1.0 for periodic)
- `accessible_mask`: centered shape padded by 1, `float32`: 1.0 if cell is part of the fluid body (i.e. 0.0 for cells outside of a Dirichlet boundary, 1.0 for Neumann or periodic)
- `no_slip_mask`: staggered shape, `bool`: `True` for velocity cells tangential to a no-slip wall boundary; the wall must also be encoded in the `active_mask` (1 inside, 0 wall), the `accessible_mask` (1 inside, 0 wall), and the `dirichlet_mask` (`True` for wall-normal component)
- `bool_periodic`: shape=(2), `bool`: `True` if (y,x)-dimension is periodic

- `viscosity`: kinematic viscosity
- `linear_solver`: solver for the linear system arising from advection-diffusion, recommended: `LinearSolverCudaMultiBicgstabILU`
- `pressure_solver`: solver for the linear system arising from the pressure correction, recommended: `PisoPressureSolverCudaCustom`

If temporally changing `dirichlet_values` or temporally changing and local `viscosity` are needed, these should be replaced by `tf.placeholder()` of relevant shape and type.

### Simulated fields

The simulated *velocity* and *pressure* fields are represented by a `StaggeredGrid` and `CenteredGrid` from Φ<sub>Flow</sub>. These grids take a `box` and `extrapolation` encoding the simulation domain and boundaries. Both the `box` and `extrapolation` can be inferred from an instance of Φ<sub>Flow</sub>'s `Domain`. More information can be found in the documentation of branch `1.15` of Φ<sub>Flow</sub>.

Since the extrapolation for a *pressure* field is different from the *velocity*, the pressure extrapolation should be calculated by `pressure_extrapolation(domain.boundaries)`.

Alongside the *pressure* and *velocity* fields, we need to initialize two additional `CenteredGrid`s for the incremental pressure calculation in the two correction steps.

The data tensors of these grids should be set to initialized with placeholder tensors, which can later be fed with data.

```
velocity_placeholder = placeholder(shape=staggered_shape, dtype=tf.float32, basename='velocity_placeholder')
velocity = StaggeredGrid.sample(velocity_placeholder,domain=domain)

pressure_placeholder = placeholder(shape=centered_shape, dtype=tf.float32, basename='pressure_placeholder')
pressure = CenteredGrid(pressure_placeholder, box=domain.box, extrapolation=pressure_extrapolation(domain.boundaries))

pressure_inc1 = CenteredGrid(tf.zeros_like(pressure.data), pressure.box, pressure.extrapolation)
pressure_inc2 = CenteredGrid(tf.zeros_like(pressure.data) + 1e-12, pressure.box, pressure.extrapolation)
```

### Simulation step

A PISO simulation step is performed by `piso_step()`.

```
velocity_new, pressure_new = piso_step(velocity, pressure, pressure_inc1, pressure_inc2, dt, sim_parameters)
```
Additionally, a flag can be set `full_output=True` to expose the substeps of the PISO algorithm.

### Running a simulation

A simulation loop is performed inside a tensorflow `Session` by iteratively processing the updated simulation fields.

```
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
with tf.Session(config=session_config) as sess:
  for i in range(timesteps):
    feed_dict = {velocity_placeholder: velocity_numpy, pressure_placeholder: pressure_numpy}
    velocity_numpy, pressure_numpy = sess.run([velocity_new, pressure_new], feed_dict=feed_dict)
```
In the first step, `velocity_numpy` and `pressure_numpy` need to be set to the initial condition.


## Setups from "Learned Turbulence Modelling" (2022)

### Spatial Mixing Layer
 
For convenience, we provide files that automate the setup of simulation of spatial mixing layers, as well as training and inference of the network models. 
 
#### Forward simulations

```spatial_mixing_layer.py```

This script runs a forward (DNS) simulation of the spatial mixing layer. The physical and simulation parameters are passed via dictionaries defined within this file.

#### Model training
 
```spatial_mixing_layer_differentiable_training.py```

This script runs a differentiable-solver training setup. Training parameters, such as the loss functions and their weighting, the number of unrolled steps, the backpropagation subrange, and the learning rate can be set in this script. 

#### Inference mode 

```spatial_mixing_layer_inference.py```

Performs a long inference simulation with the trained turbulence model.

## Dataset

An accompanying dataset can be downloaded [here](https://mediatum.ub.tum.de/1687392) (~ 2 GB).

The dataset consists of spatially and temporally coarsened DNS snapshots. In line with the coarsening used in our trained models, the ratio was set to 8 for each spatial and the temporal dimension. 

Our data contains 2 simulations of the isotropic decaying turbulence case, 3 setups of temporally developing mixing layers and 6 simulations of spatially developing mixing layers. 

The training script is setup to directly process coarsened data, and one simulation in the dataset is always left as an extrapolative test case.
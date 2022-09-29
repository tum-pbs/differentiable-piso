from phi.tf.flow import math, StaggeredGrid
import glob as gb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools


# TIME ANALYSIS
def spectral_analysis_time(velocity, tstart, yMin, yMax, xMin, xMax, averaging, sample_spacing):
    monitoring_range = velocity[tstart:, yMin:yMax, xMin:xMax, :]
    ux_in = monitoring_range[..., 1] - averaging * np.average(monitoring_range[..., 1], axis=0)
    uy_in = monitoring_range[..., 0] - averaging * np.average(monitoring_range[..., 0], axis=0)

    # number of samplepoints
    N = uy_in.shape[0]

    uy_dft = np.fft.fft(uy_in, N, axis=0)
    ux_dft = np.fft.fft(ux_in, N, axis=0)

    x = np.linspace(0.0, N * sample_spacing, N)

    freq = np.arange(0, N - 1) * (1. / sample_spacing / N)
    freq = freq[freq < 1. / sample_spacing / 2]

    Ek = np.abs(ux_dft[:N // 2]) ** 2 + np.abs(uy_dft[:N // 2]) ** 2

    return freq, uy_dft, ux_dft, Ek


def spectral_analysis_1Dspace(velocity, tStart, tFin, tEval, yCoord, xRange, grid_spacing, averaging):
    monitoring_range = velocity[tStart:tFin, yCoord, xRange[0]:xRange[1]]

    ux_in = monitoring_range[tEval[0] - tStart:tEval[1] - tStart, ..., 0] - averaging * np.average(
        monitoring_range[..., 0], axis=0)
    uy_in = monitoring_range[tEval[0] - tStart:tEval[1] - tStart, ..., 1] - averaging * np.average(
        monitoring_range[..., 1], axis=0)

    uy_dft = np.fft.fft(uy_in, axis=-1)
    ux_dft = np.fft.fft(ux_in, axis=-1)
    N = np.abs(xRange[1] - xRange[0])

    dkm = 2 * np.pi / (N * grid_spacing)
    km = np.arange(0, np.pi / grid_spacing, dkm)

    Ekm = grid_spacing / (2 * np.pi * N) * (ux_dft * np.conj(ux_dft) + uy_dft * np.conj(uy_dft))

    return km, Ekm

def vorticity_structure(velocity: StaggeredGrid):
    velocity_tensor = velocity.padded(1).staggered_tensor()
    vorticity = (velocity_tensor[:,1:-1,1:-1,0] - velocity_tensor[:,1:-1,:-2,0])/velocity.dx[0] -\
                (velocity_tensor[:,1:-1,1:-1,1] - velocity_tensor[:,:-2,1:-1,1])/velocity.dx[0]
    domain_shape = vorticity.shape[1:]
    vort_cen = vorticity[0,domain_shape[0]//2, domain_shape[1]//2]

    x_dist = (np.matmul(math.expand_dims(math.range(domain_shape[0], dtype=np.float32),-1),math.ones((1,domain_shape[1]))) - domain_shape[0]/2)**2
    y_dist = (np.matmul(math.ones((domain_shape[0],1)), math.expand_dims(math.range(domain_shape[1], dtype=np.float32),0)) - domain_shape[1]/2)**2
    r = np.round(np.sqrt(x_dist + y_dist)).astype(np.int)
    data = vorticity - vort_cen
    max = np.ceil(np.sqrt((domain_shape[0]//2)**2 + (domain_shape[1]//2)**2)+1).astype(int)
    vort_struct = np.zeros((max,))
    num_of_entries = np.zeros((max,))
    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            num_of_entries[r[i,j]] += 1
            vort_struct[r[i,j]] += data[0,i,j]
    vort_struct[num_of_entries>0] /= num_of_entries[num_of_entries>0]
    return vort_struct

def vorticity_correlation(velocity: StaggeredGrid):
    velocity_tensor = velocity.padded(1).staggered_tensor()
    vorticity = (velocity_tensor[:,1:-1,1:-1,0] - velocity_tensor[:,1:-1,:-2,0])/velocity.dx[0] -\
                (velocity_tensor[:,1:-1,1:-1,1] - velocity_tensor[:,:-2,1:-1,1])/velocity.dx[0]
    domain_shape = vorticity.shape[1:]
    vort_cen = vorticity[0,domain_shape[0]//2, domain_shape[1]//2]

    x_dist = (np.matmul(math.expand_dims(math.range(domain_shape[0], dtype=np.float32),-1),math.ones((1,domain_shape[1]))) - domain_shape[0]/2)**2
    y_dist = (np.matmul(math.ones((domain_shape[0],1)), math.expand_dims(math.range(domain_shape[1], dtype=np.float32),0)) - domain_shape[1]/2)**2
    r = np.round(np.sqrt(x_dist + y_dist)).astype(np.int)
    data = vorticity * vort_cen
    max = np.ceil(np.sqrt((domain_shape[0]//2)**2 + (domain_shape[1]//2)**2)+1).astype(int)
    vort_struct = np.zeros((max,))
    num_of_entries = np.zeros((max,))
    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            num_of_entries[r[i,j]] += 1
            vort_struct[r[i,j]] += data[0,i,j]
    vort_struct[num_of_entries>0] /= num_of_entries[num_of_entries>0]
    return vort_struct/vort_cen/vort_cen

def EK_spectrum_2D(velocity_centered, domain_size):
    N = velocity_centered.shape[1]
    cutoff = N//2
    u = velocity_centered[...,1]
    v = velocity_centered[...,0]
    small = 1e-20
    u_fft = np.fft.fft2(u) / u.size
    v_fft = np.fft.fft2(v) / v.size
    e_u = np.abs(u_fft*np.conj(u_fft))
    e_v = np.abs(v_fft*np.conj(v_fft))
    e_u_shift = np.fft.fftshift(e_u)
    e_v_shift = np.fft.fftshift(e_v)

    domain_shape = e_u_shift.shape
    sample_radius = int(np.ceil(((domain_shape[0]**2 + domain_shape[1]**2)**.5  *.5)) +1)
    e_sampled = np.zeros(sample_radius,)+small

    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            wavenum = int(np.round(np.sqrt((i - domain_shape[0]/2) ** 2 + (j - domain_shape[1]/2) ** 2)))
            e_sampled[wavenum] += (e_u_shift[i,j] + e_v_shift[i,j])*.5
    return np.arange(e_sampled.size, dtype=np.float)[:cutoff], e_sampled[:cutoff]

def EK_spectrum_3D(velocity_centered, domain_size):
    N = velocity_centered.shape[1]
    cutoff = N//2
    u = velocity_centered[0,...,2]
    v = velocity_centered[0,...,1]
    w = velocity_centered[0,...,0]
    small = 1e-20

    u_fft = np.fft.fftn(u) / u.size
    v_fft = np.fft.fftn(v) / v.size
    w_fft = np.fft.fftn(w) / w.size
    e_u = np.abs(u_fft*np.conj(u_fft))
    e_v = np.abs(v_fft*np.conj(v_fft))
    e_w = np.abs(w_fft*np.conj(w_fft))
    e_u_shift = np.fft.fftshift(e_u)
    e_v_shift = np.fft.fftshift(e_v)
    e_w_shift = np.fft.fftshift(e_w)

    domain_shape = e_u_shift.shape
    sample_radius = int(np.ceil(((domain_shape[0]**2 + domain_shape[1]**2 + domain_shape[2]**2)**.5  *.5)) +1)
    e_sampled = np.zeros(sample_radius,)+small

    for i in range(domain_shape[0]):
        for j in range(domain_shape[1]):
            for k in range(domain_shape[2]):
                wavenum = int(np.round(np.sqrt((i - domain_shape[0]/2) ** 2 +
                                               (j - domain_shape[1]/2) ** 2 +
                                               (k - domain_shape[2]/2) ** 2)))
                e_sampled[wavenum] += (e_u_shift[i,j,k] + e_v_shift[i,j,k] + e_w_shift[i,j,k])*.5
    return np.arange(e_sampled.size, dtype=np.float)[:cutoff], e_sampled[:cutoff]

def EK_spectrum_avg_vorticity(path, start_step, steps, timestep_ratio, dx):
    end = start_step+steps*timestep_ratio
    data = [np.load(path + 'velocity_'+ str(s // 8).zfill(6) + '.npz')['arr_0'] for s in range(start_step, end, timestep_ratio)]
    vorticity = [(data[i][0, 1:-1, 1:-1, 0] - data[i][0, 1:-1, :-2, 0]) / dx - \
                   (data[i][0, 1:-1, 1:-1, 1] - data[i][0, :-2, 1:-1, 1]) / dx for i in range(steps)]
    data_cen = [np.concatenate([np.expand_dims(data[i][0, 1:, :-1, 0] + data[i][0, :-1, :-1, 0], -1) / 2,
                                np.expand_dims(data[i][0, :-1, 1:, 1] + data[i][0, :-1, :-1, 1], -1) / 2], axis=-1) for i in range(steps)]
    NN_spectrum = [EK_spectrum_2D(data_cen[i], [2 * np.pi, 2 * np.pi]) for i in range(steps)]

    return NN_spectrum[0][0], np.average([NN_spectrum[i][1] for i in range(steps)], axis =0), vorticity

def tf_fftshift(fft_2D):
    hshape = [l//2 for l in fft_2D.shape.as_list()]
    result_high = tf.concat([fft_2D[hshape[0]:,:hshape[1]], fft_2D[:hshape[0], :hshape[1]]], axis=0)
    result_low =  tf.concat([fft_2D[hshape[0]:,hshape[1]:], fft_2D[:hshape[0], hshape[1]:]], axis=0)
    return tf.concat([result_low, result_high], axis=1)

def EK_spectrum_2D_tf(velocity_centered):
    N = velocity_centered.shape[1]
    u = velocity_centered[...,1]
    v = velocity_centered[...,0]
    u_fft = tf.fft2d(u)
    v_fft = tf.fft2d(v)
    e_u = tf.abs(u_fft*tf.conj(u_fft))
    e_v = tf.abs(v_fft*tf.conj(v_fft))
    e_u_shift = tf_fftshift(e_u)
    e_v_shift = tf_fftshift(e_v)

    domain_shape = e_u_shift.shape.as_list()
    wvn_i = (tf.matmul(tf.expand_dims(tf.range(domain_shape[0], dtype=tf.float32),-1),tf.ones((1,domain_shape[1]))) - domain_shape[0]/2)**2
    wvn_j = (tf.matmul(tf.ones((domain_shape[0],1)), tf.expand_dims(tf.range(domain_shape[1], dtype=tf.float32),0)) - domain_shape[1]/2)**2
    wvn = tf.cast(tf.round(tf.sqrt(wvn_i+wvn_j)), dtype=tf.int32)
    flat_wvn = tf.reshape(wvn, [-1])
    flat_data = tf.reshape(e_u_shift+e_v_shift, [-1])
    sorted_wvn_ids = tf.argsort(flat_wvn)
    vel_shape = velocity_centered.shape.as_list()
    cutoff =  math.min(vel_shape[:2])//2 #int(np.ceil(((vel_shape[0]//2)**2+(vel_shape[1]//2)**2)**.5))
    esum = tf.math.segment_sum(tf.gather(flat_data, sorted_wvn_ids), tf.gather(flat_wvn, sorted_wvn_ids))*.5
    esum = esum[:cutoff]/np.prod(u.shape.as_list())/np.prod(v.shape.as_list())
    esum.set_shape([cutoff,])
    return esum

def EK_spectrum_1D_tf(velocity_centered, axis):
    print('espec vel shape: ', velocity_centered.shape)
    N = velocity_centered.shape[1]
    u = velocity_centered[..., 1]
    v = velocity_centered[..., 0]
    axes = np.delete(np.arange(len(u.shape)), axis).tolist()+[axis,]
    u = tf.transpose(u, axes)
    v = tf.transpose(v, axes)
    u_fft = tf.fft(u)
    v_fft = tf.fft(v)
    e_u = tf.abs(u_fft * tf.conj(u_fft))
    e_v = tf.abs(v_fft * tf.conj(v_fft))
    esum = tf.reduce_sum(e_u, axis=np.arange(len(u.shape))[:-1]) +\
           tf.reduce_sum(e_v, axis=np.arange(len(u.shape))[:-1])
    return esum[:N//2+1]


def plot_spectra(wavenumbers,spectra, title, legend, figsize=(8,6), helper_line_exponents = [-3.,-5/3,-5.]):
    fig = plt.figure(figsize=figsize)
    [plt.loglog(wavenumbers[i],spectra[i]) for i in range(len(spectra))]
    plt.xlabel(r'Wavenumber $\kappa$', fontsize=15)
    plt.ylabel(r'TKE $E(\kappa)$', fontsize=15)
    plt.title(title)
    wvn = wavenumbers[0]
    styles = ['dashed','solid','dashdot']
    for i in range(len(helper_line_exponents)):
        plt.loglog(wvn[10:], wvn[10:]**helper_line_exponents[i] , linewidth=1, linestyle=styles[i], color='k')
    plt.legend(legend)
    ylims = plt.gca().get_ylim()
    plt.vlines(np.max(wvn),ylims[0], ylims[1])
    plt.grid()
    return fig


def spectral_analysis_2Dspace(velocity, tStart, tFin, tEval, frame, grid_spacing, averaging):
    monitoring_range = velocity[tStart:tFin,
                       frame[0][0]:frame[0][1],
                       frame[1][0]:frame[1][1]]

    # fluctuating velocities
    ux_in = monitoring_range[[tEval - tStart], ..., 0] - averaging * np.average(monitoring_range[..., 0], axis=0)
    uy_in = monitoring_range[[tEval - tStart], ..., 1] - averaging * np.average(monitoring_range[..., 1], axis=0)

    uy_dft = np.fft.fft2(uy_in, axes=(-2, -1))
    ux_dft = np.fft.fft2(ux_in, axes=(-2, -1))
    Ny = np.abs(frame[0][1] - frame[0][0])
    Nx = np.abs(frame[1][1] - frame[1][0])

    dkx = 2 * np.pi / (Nx * grid_spacing)
    dky = 2 * np.pi / (Ny * grid_spacing)

    kx = np.arange(0, np.pi / grid_spacing, dkx)
    ky = np.arange(0, np.pi / grid_spacing, dky)
    kp = np.zeros(int(np.sqrt(2) * max(Nx / 2, Ny / 2)) // 1)
    Ekp = np.zeros(int(np.sqrt(2) * max(Nx / 2, Ny / 2)) // 1)
    num_lm = np.zeros(int(np.sqrt(2) * max(Nx / 2, Ny / 2)) // 1)

    for p in range(kp.shape[0]):
        kp[p] = p * max(dkx, dky)
        Ekp[p] = np.sum(grid_spacing ** 2 * min(dkx, dky) / (8 * np.pi ** 2 * Nx * Ny) *
                        np.array([ux_dft[:, m, l] * np.conj(ux_dft[:, m, l]) + uy_dft[:, m, l] * np.conj(uy_dft[:, m, l])
                                  for (m, l) in itertools.product(range(ky.shape[0]), range(kx.shape[0]))
                                  if np.abs((kx[l] ** 2 + ky[m] ** 2) ** .5 - kp[p]) < max(dkx, dky) / 2]))
        num_lm[p] = len([0 for (m, l) in itertools.product(range(ky.shape[0]), range(kx.shape[0]))
                         if np.abs((kx[l] ** 2 + ky[m] ** 2) ** .5 - kp[p]) < max(dkx, dky) / 2])

    return kp, Ekp, num_lm, kx, ky

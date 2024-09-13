import sys
sys.path.append('C:\\UWMadisonResearch\\Conditional_Score_FNO_ErrorTerm')

import torch
import math
import numpy as np
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.signal import convolve2d
from Data_Generation.random_forcing import GaussianRF, get_twod_bj, get_twod_dW


def local_avg(tensor, kernel):
    B, H, W = tensor.shape
    filtered_tensor = cp.zeros((B, H, W))
    for b in range(B):
        filtered_slice = convolve2d(tensor[b, :, :], kernel, mode='same', boundary='wrap')
        filtered_tensor[b, :, :] = filtered_slice
    return filtered_tensor

def smagorinsky(q_h, v_h, k_x, k_y, Cs, space=1/32):
    N1 = N2 = 32

    q_x = q_h.clone()
    temp = q_x[..., 0].clone()
    q_x[..., 0] = -2 * math.pi * k_x * q_x[..., 1]
    q_x[..., 1] = 2 * math.pi * k_x * temp
    q_x = torch.fft.ifftn(torch.view_as_complex(q_x), dim=[1, 2], s=(N1, N2)).real

    q_y = q_h.clone()
    temp = q_y[..., 0].clone()
    q_y[..., 0] = -2 * math.pi * k_y * q_y[..., 1]
    q_y[..., 1] = 2 * math.pi * k_y * temp
    q_y = torch.fft.ifftn(torch.view_as_complex(q_y), dim=[1, 2], s=(N1, N2)).real

    v_x = v_h.clone()
    temp = v_x[..., 0].clone()
    v_x[..., 0] = -2 * math.pi * k_x * v_x[..., 1]
    v_x[..., 1] = 2 * math.pi * k_x * temp
    v_x = torch.fft.ifftn(torch.view_as_complex(v_x), dim=[1, 2], s=(N1, N2)).real

    v_y = v_h.clone()
    temp = v_y[..., 0].clone()
    v_y[..., 0] = -2 * math.pi * k_y * v_y[..., 1]
    v_y[..., 1] = 2 * math.pi * k_y * temp
    v_y = torch.fft.ifftn(torch.view_as_complex(v_y), dim=[1, 2], s=(N1, N2)).real

    # Strain rate tensor components
    S11 = q_x
    S22 = v_y
    S12 = 0.5 * (q_y + v_x)

    # Magnitude of strain rate tensor
    S_mag = torch.sqrt(2.0 * (S11 ** 2 + S22 ** 2 + 2 * S12 ** 2))

    nu_t = (Cs * space) ** 2 * S_mag

    return nu_t

def navier_stokes_2d_filtered(a, w0, f, visc, T, kernel, downscale, delta_t=1e-3, record_steps=1, stochastic_forcing=None):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)
    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)

    # If stochastic forcing
    if stochastic_forcing is not None:
        # initialise noise
        bj = get_twod_bj(delta_t, [N1, N2], a, stochastic_forcing['alpha'], w_h.device)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1).transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 / a[0] ** 2 + k_y ** 2 / a[1] ** 2)

    # lap_ = lap.clone()
    lap[0, 0] = 1.0

    lap = torch.tensor(lap).to(w0.device)

    w_0_filtered = local_avg(cp.from_dlpack(w0), kernel)
    w_0_filtered = torch.tensor(w_0_filtered).to(w0.device)
    w_0_filtered = w_0_filtered[:, ::downscale, ::downscale]
    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    filtered_sol = torch.zeros(*w_0_filtered.size(), record_steps, device=w0.device)
    nonlinear_diff = torch.zeros(*w_0_filtered.size(), record_steps, device=w0.device)
    diffusion_diff = torch.zeros(*w_0_filtered.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi * k_x * v[..., 1]
        v[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y / a[1]), dim=[1, 2], s=(N1, N2)).real


        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real
        w_filtered = local_avg(cp.from_dlpack(w), kernel)
        w_filtered = torch.tensor(w_filtered).to(w0.device)

        w_h_filtered = torch.fft.fftn(w_filtered, dim=[1, 2])
        w_h_filtered = torch.stack([w_h_filtered.real, w_h_filtered.imag], dim=-1)

        # Stream function in Fourier space: solve Poisson equation
        psi_h_filtered = w_h_filtered.clone()
        psi_h_filtered[..., 0] = psi_h_filtered[..., 0] / lap
        psi_h_filtered[..., 1] = psi_h_filtered[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q_filtered = psi_h_filtered.clone()
        temp = q_filtered[..., 0].clone()
        q_filtered[..., 0] = -2 * math.pi * k_y * q_filtered[..., 1]
        q_filtered[..., 1] = 2 * math.pi * k_y * temp
        q_filtered = torch.fft.ifftn(torch.view_as_complex(q_filtered / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v_filtered = psi_h_filtered.clone()
        temp = v_filtered[..., 0].clone()
        v_filtered[..., 0] = 2 * math.pi * k_x * v_filtered[..., 1]
        v_filtered[..., 1] = -2 * math.pi * k_x * temp
        v_filtered = torch.fft.ifftn(torch.view_as_complex(v_filtered / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x_filtered = w_h_filtered.clone()
        temp = w_x_filtered[..., 0].clone()
        w_x_filtered[..., 0] = -2 * math.pi * k_x * w_x_filtered[..., 1]
        w_x_filtered[..., 1] = 2 * math.pi * k_x * temp
        w_x_filtered = torch.fft.ifftn(torch.view_as_complex(w_x_filtered / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y_filtered = w_h_filtered.clone()
        temp = w_y_filtered[..., 0].clone()
        w_y_filtered[..., 0] = -2 * math.pi * k_y * w_y_filtered[..., 1]
        w_y_filtered[..., 1] = 2 * math.pi * k_y * temp
        w_y_filtered = torch.fft.ifftn(torch.view_as_complex(w_y_filtered / a[1]), dim=[1, 2], s=(N1, N2)).real


        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        nonlinear_term = q * w_x + v * w_y
        filtered_nonlinear_term = local_avg(cp.from_dlpack(nonlinear_term), kernel)
        filtered_nonlinear_term = torch.tensor(filtered_nonlinear_term).to(w0.device)
        nonlinear_term_filtered = q_filtered * w_x_filtered + v_filtered * w_y_filtered
        nonlinear_diff_term = filtered_nonlinear_term - nonlinear_term_filtered

        F_h = torch.fft.fftn(nonlinear_term, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        #Dealias
        # F_h[..., 0] = dealias * F_h[..., 0]
        # F_h[..., 1] = dealias * F_h[..., 1]

        # Diffusion term
        diffusion_term_h = torch.zeros_like(w_h)
        diffusion_term_h[..., 0] = visc * lap * w_h[..., 0]
        diffusion_term_h[..., 1] = visc * lap * w_h[..., 1]
        diffusion_term = torch.fft.ifftn(torch.view_as_complex(diffusion_term_h), dim=[1, 2], s=(N1, N2)).real
        filtered_diffusion_term = local_avg(cp.from_dlpack(diffusion_term), kernel)
        filtered_diffusion_term = torch.tensor(filtered_diffusion_term).to(w0.device)

        diffusion_term_filtered = torch.zeros_like(w_h_filtered)
        diffusion_term_filtered[..., 0] = visc * lap * w_h_filtered[..., 0]
        diffusion_term_filtered[..., 1] = visc * lap * w_h_filtered[..., 1]
        diffusion_term_filtered = torch.fft.ifftn(torch.view_as_complex(diffusion_term_filtered), dim=[1, 2],
                                                  s=(N1, N2)).real

        diffusion_diff_term = filtered_diffusion_term - diffusion_term_filtered

        if stochastic_forcing:
          dW, dW2 = get_twod_dW(bj, stochastic_forcing['kappa'], w_h.shape[0], w_h.device)
          gudWh = torch.fft.fft2(stochastic_forcing['sigma']*dW, dim=[-2,-1])
          gudWh = torch.stack([gudWh.real, gudWh.imag],dim=-1)
        else:
          gudWh = torch.zeros_like(f_h)


        # Cranck-Nicholson update
        w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + gudWh[..., 0] +
                       w_h[..., 0] - 0.5 * delta_t * diffusion_term_h[..., 0]) / (1.0 + 0.5 * delta_t * visc * lap)
        w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + gudWh[..., 1] +
                       w_h[..., 1] - 0.5 * delta_t * diffusion_term_h[..., 1]) / (1.0 + 0.5 * delta_t * visc * lap)

        w_filtered = w_filtered[:, ::downscale, ::downscale]
        nonlinear_diff_term = nonlinear_diff_term[:, ::downscale, ::downscale]
        diffusion_diff_term = diffusion_diff_term[:, ::downscale, ::downscale]

        if j == 0:
            sol[..., 0] = w
            filtered_sol[..., 0] = w_filtered
            nonlinear_diff[..., 0] = nonlinear_diff_term
            diffusion_diff[..., 0] = diffusion_diff_term
            sol_t[0] = 0

            c += 1

        if j != 0 and (j) % record_time == 0:
            # Record solution and time
            sol[..., c] = w
            filtered_sol[..., c] = w_filtered
            nonlinear_diff[..., c] = nonlinear_diff_term
            diffusion_diff[..., c] = diffusion_diff_term
            sol_t[c] = t

            c += 1

        t += delta_t

    return sol,  filtered_sol, nonlinear_diff, diffusion_diff, sol_t

def navier_stokes_2d_orig(a, w0, f, visc, T, delta_t=1e-4, record_steps=1):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1).transpose(0, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 / a[0] ** 2 + k_y ** 2 / a[1] ** 2)
    # lap_ = lap.clone()
    lap[0, 0] = 1.0

#     dealias = torch.unsqueeze(
#         torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    nonlinear = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)
    diffusion = torch.zeros(*w0.size(), record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi * k_x * v[..., 1]
        v[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        nonlinear_term = q * w_x + v * w_y
        F_h = torch.fft.fftn(nonlinear_term, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        #Dealias
#         F_h[..., 0] = dealias * F_h[..., 0]
#         F_h[..., 1] = dealias * F_h[..., 1]

        diffusion_h = torch.zeros_like(w_h)
        diffusion_h[..., 0] = visc * lap * w_h[..., 0]
        diffusion_h[..., 1] = visc * lap * w_h[..., 1]
        diffusion_term = torch.fft.ifftn(torch.view_as_complex(diffusion_h), dim=[1, 2], s=(N1, N2)).real

        # Cranck-Nicholson update
        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real
        w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + (
                    1.0 - 0.5 * delta_t * visc * lap) * w_h[..., 0]) / (1.0 + 0.5 * delta_t * visc * lap)
        w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + (
                    1.0 - 0.5 * delta_t * visc * lap) * w_h[..., 1]) / (1.0 + 0.5 * delta_t * visc * lap)

        # Update real time (used only for recording)
        if j == 0:
            sol[..., 0] = w
            nonlinear[..., 0] = nonlinear_term
            sol_t[0] = 0
            diffusion[..., 0] = diffusion_term

            c += 1

        if j != 0 and (j) % record_time == 0:
            # Record solution and time
            sol[..., c] = w
            nonlinear[..., c] = nonlinear_term
            sol_t[c] = t
            diffusion[..., c] = diffusion_term

            c += 1

        t += delta_t

    return sol, nonlinear, diffusion, sol_t

def navier_stokes_2d_smag(a, w0, f, visc, T, delta_t=1e-4, record_steps=1):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1).transpose(0, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 / a[0] ** 2 + k_y ** 2 / a[1] ** 2)
    # lap_ = lap.clone()
    lap[0, 0] = 1.0

#     dealias = torch.unsqueeze(
#         torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)
    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    nonlinear_cor_col = torch.zeros(*w0.size(), record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q_h = psi_h.clone()
        temp = q_h[..., 0].clone()
        q_h[..., 0] = -2 * math.pi * k_y * q_h[..., 1]
        q_h[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q_h / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v_h = psi_h.clone()
        temp = v_h[..., 0].clone()
        v_h[..., 0] = 2 * math.pi * k_x * v_h[..., 1]
        v_h[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v_h / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y / a[1]), dim=[1, 2], s=(N1, N2)).real

        nu_t = smagorinsky(q_h, v_h, k_x, k_y, 0.16, space=1 / 32)
        correction_term_h = torch.zeros_like(w_h)
        correction_term_h[..., 0] = nu_t * lap * w_h[..., 0]
        correction_term_h[..., 1] = nu_t * lap * w_h[..., 1]
        correction_term = torch.fft.ifftn(torch.view_as_complex(correction_term_h), dim=[1, 2], s=(N1, N2)).real

        # visc_effective = visc + nu_t
        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        nonlinear_term = q * w_x + v * w_y + correction_term
        F_h = torch.fft.fftn(nonlinear_term, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        #Dealias
#         F_h[..., 0] = dealias * F_h[..., 0]
#         F_h[..., 1] = dealias * F_h[..., 1]

        diffusion_h = torch.zeros_like(w_h)
        diffusion_h[..., 0] = visc * lap * w_h[..., 0]
        diffusion_h[..., 1] = visc * lap * w_h[..., 1]

        # Cranck-Nicholson update
        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real
        w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] +
                       w_h[..., 0] - 0.5 * delta_t * diffusion_h[..., 0]) / (1.0 + 0.5 * delta_t * visc * lap)
        w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] +
                       w_h[..., 1] - 0.5 * delta_t * diffusion_h[..., 1]) / (1.0 + 0.5 * delta_t * visc * lap)

        # Update real time (used only for recording)
        if j == 0:
            sol[..., 0] = w
            nonlinear_cor_col[..., 0] = correction_term
            c += 1

        if j != 0 and (j) % record_time == 0:
            # Record solution and time
            sol[..., c] = w
            nonlinear_cor_col[..., c] = correction_term
            c += 1

        t += delta_t

    return sol, nonlinear_cor_col

def navier_stokes_2d_model(a, w0, f, visc, T, condition, sampler, delta_t=1e-4, record_steps=1, eva_steps=10):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)

    # Record solution every this number of steps
    record_time = math.floor(steps / record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1).transpose(0, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 / a[0] ** 2 + k_y ** 2 / a[1] ** 2)
    # lap_ = lap.clone()
    lap[0, 0] = 1.0

#     dealias = torch.unsqueeze(
#         torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2, torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q_h = psi_h.clone()
        temp = q_h[..., 0].clone()
        q_h[..., 0] = -2 * math.pi * k_y * q_h[..., 1]
        q_h[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q_h / a[1]), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v_h = psi_h.clone()
        temp = v_h[..., 0].clone()
        v_h[..., 0] = 2 * math.pi * k_x * v_h[..., 1]
        v_h[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v_h / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x / a[0]), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y / a[1]), dim=[1, 2], s=(N1, N2)).real

        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real

        if j % eva_steps == 0:
            correction_term = sampler(condition[..., j], w)
        else:
            correction_term = correction_term

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        nonlinear_term = q * w_x + v * w_y + correction_term
        F_h = torch.fft.fftn(nonlinear_term, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        #Dealias
#         F_h[..., 0] = dealias * F_h[..., 0]
#         F_h[..., 1] = dealias * F_h[..., 1]
        diffusion_h = torch.zeros_like(w_h)
        diffusion_h[..., 0] = visc * lap * w_h[..., 0]
        diffusion_h[..., 1] = visc * lap * w_h[..., 1]

        # Cranck-Nicholson update
        w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] +
                       w_h[..., 0] - 0.5 * delta_t * diffusion_h[..., 0]) / (1.0 + 0.5 * delta_t * visc * lap)
        w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t * f_h[..., 1] +
                       w_h[..., 1] - 0.5 * delta_t * diffusion_h[..., 1]) / (1.0 + 0.5 * delta_t * visc * lap)

        # Update real time (used only for recording)
        if j == 0:
            sol[..., 0] = w
            c += 1

        if j != 0 and (j) % record_time == 0:
            # Record solution and time
            sol[..., c] = w
            c += 1

        t += delta_t

    return sol
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def Diff_mat_1D(Nx, device='cpu', partially_observed=None):
    # I tried to change the implementation to torch dense matrix here, but kept the original implem.
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0, 1]] = [-3, 4]  # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-2, Nx-1]] = [-4, 3]  # this is 2nd order backward difference (2*dx division is required)

    # Second derivative
    D2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0, 1, 2]] = [2, -5, 4] # this is 2nd order forward difference. division by dx^2 required. 
    D2_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    
    if partially_observed is not None:
        if partially_observed == 'front_rear_quarter':

            # # note that Nx == 130 here
            D_1d.data[(Nx - 2) // 4] = [0., 0.]
            D_1d.data[(Nx - 2) // 4 + 1] = [0., 0.]
            D_1d.data[(Nx - 2) // 4 + 2] = [0., 0.]
            D_1d.data[((Nx - 2) * 3) // 4 + 1] = [0., 0.]
            D_1d.data[((Nx - 2) * 3) // 4] = [0., 0.]
            # D_1d.data[((Nx - 2) * 3) // 4 - 1] = [0., 0.]

            D2_1d.data[(Nx - 2) // 4] = [0., 0., 0.]
            D2_1d.data[(Nx - 2) // 4 + 1] = [0., 0., 0.]
            D2_1d.data[(Nx - 2) // 4 + 2] = [0., 0., 0.]
            # D2_1d.data[(Nx - 2) // 4 - 2] = [0., 0., 0.]
            # D2_1d.data[((Nx - 2) * 3) // 4] = [0., 0., 0.]
            D2_1d.data[((Nx - 2) * 3) // 4 + 1] = [0., 0., 0.]
            D2_1d.data[((Nx - 2) * 3) // 4] = [0., 0., 0.]
            # D2_1d.data[((Nx - 2) * 3) // 4 - 1] = [0., 0., 0.]

        else:
            raise NotImplementedError
    return D_1d, D2_1d


def one_step_solver_u(u, f, dt=0.1, visc=0.01, mode='mean', partially_observed=None):
    '''
    Calculates u following the solver using coarse time step, based on which the 
    PINN loss will be evaluated.
    Note that this is an approximated version -- only 10 time stamps in u.

    Arguments:
        u: (B, 11, 128)
        f: (B, 10, 128)
    '''
    s = u.size(-1)
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax - xmin) / (s + 1)
    D_1d, D2_1d = Diff_mat_1D(s + 2, device=u.device, partially_observed=partially_observed)
    dx_idx = list(D_1d.rows)
    dx = torch.FloatTensor(np.stack(D_1d.data)/(2*delta_x)).to(u.device)
    dxx_idx = list(D2_1d.rows)
    dxx = torch.FloatTensor(np.stack(D2_1d.data)/delta_x**2).to(u.device)

    # pad to 130 spatially
    u = F.pad(u, (1, 1))
    f = F.pad(f, (1, 1))
    # Dirichlet BC
    u = u[...,1: -1]
    u = F.pad(u, (1, 1)) # u: 11 * 130

    dx_u = torch.einsum('btsi,si->bts', u[..., dx_idx], dx)
    dxx_u =  torch.einsum('btsi,si->bts', u[..., dxx_idx], dxx)
    
    # using u_{i+2} and u_i to approximate u_{i+1} (somewhat resembling Crank-Nicolson method)
    u_next = u[..., :-1, :] + dt * (-u[..., :-1, :] * dx_u[..., :-1, :] + visc * dxx_u[..., :-1, :] + f)
    u_prev = u[..., 1:, :] - dt * (-u[..., 1:, :] * dx_u[..., 1:, :] + visc * dxx_u[..., 1:, :] + f)
    
    u_pde = torch.zeros_like(u)
    if mode == 'mean':
        # NOTE: only time from 1 to 9 can be used to evaluate the PINN loss (init and last are not calculated).
        u_pde[..., 1:, :] = u_next / 2
        u_pde[..., :-1, :] += u_prev / 2
    elif mode == 'forward':
        # NOTE: only time from 1 to 10 can be used to evaluate the PINN loss (init and last are not calculated).
        # So, fill init u to u_pde
        u_pde[..., 1:, :] = u_next
        u_pde[..., 0, :] = u[..., 0, :]
    elif mode == 'backward':
        # NOTE: only time from 0 to 9 can be used to evaluate the PINN loss (init and last are not calculated).
        # So, fill final u to u_pde
        u_pde[..., :-1, :] = u_prev
        u_pde[..., -1, :] = u[..., -1, :]

    # Only 128 x can be used. Remove padding.
    u_pde = u_pde[..., 1: -1]
    
    return u_pde

def pinn_loss(u, f, mode='mean', partially_observed=None):
    '''
    Calculates the PINN loss given u and f.
    Note that this is an approximated version -- only 10 time stamps in u.
    
    Arguments:
        u (B, 11, 128)
        f (B, 10, 128)
    '''
    u_pde = one_step_solver_u(u, f, mode=mode, partially_observed=partially_observed)
    # NOTE: for different modes, the loss will be different by whether filling the init/final u
    if partially_observed:
        Nx = u_pde.size(-1)
        u_pde[..., Nx // 4: (Nx * 3) // 4] = u[..., Nx // 4: (Nx * 3) // 4]
    loss = (u_pde - u).square().mean()
    return loss

def get_pinn_loss_2dconv(mode='mean', partially_observed=None):
    '''
    Returns the PINN loss function.

    Arguments:
        x: (B, 2, 16, 128)
    '''
    def pinn_loss_2dconv(x):
        u, f = x[..., 0, :11, :], x[..., 0, :10, :]
        return pinn_loss(u, f, mode=mode, partially_observed=partially_observed)
    return pinn_loss_2dconv

def residual_gradient(x, mode='mean', partially_observed=None):
    loss_fn = get_pinn_loss_2dconv(mode=mode, partially_observed=partially_observed)
    x.requires_grad_(True)
    J = loss_fn(x) # vec of size of batch
    grad = torch.autograd.grad(J, x, grad_outputs=torch.ones_like(J), retain_graph = True, create_graph=True, allow_unused=True)[0]
    return grad.detach()
    
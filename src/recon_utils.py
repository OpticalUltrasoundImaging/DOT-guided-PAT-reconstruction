import os
import sys
from pathlib import Path
import numpy as np
from .load_data_utils import LinearSystemParam
from .fluence_utils import compute_phi_heterogeneous, compute_phi_and_grad
from .us_utils import pe_das_linear, nakagami_linear
from .pat_utils import pa_das_linear, pa_inverse_recon
from .recon_iq_utils import PATInverseSolver

from typing import Optional, Tuple
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, identity, vstack, csr_matrix, kron
from scipy.sparse.linalg import lsqr

N_ELEMENTS = 128  # default number of elements for linear array
Z_MAX_CM = 4.0  # cm, default max depth for reconstruction
DY = 0.1 # cm, default pixel size along y direction (elevation)
NY = 11

def _enforce_mean(map_arr: np.ndarray,
                  target_mean: float,
                  eps: float = 1e-12)-> np.ndarray:
    curr_mean = np.mean(map_arr)
    if curr_mean == 0:
        return np.ones_like(map_arr) * target_mean
    return map_arr * (target_mean / (curr_mean + eps))

def _initialize_mus(input_dir: str,
                    info_US: LinearSystemParam,
                    mu_s_mean_cm: float = 8.0,
                    dB_US: int=75,
                    depth_max_cm: float = Z_MAX_CM*1e-2) -> tuple[np.ndarray, np.ndarray]:
    _, RF_env_raw, US_img = pe_das_linear(input_dir, info_US, dB_US, 'hann', 'gsf')
    naka_img, _ = nakagami_linear(RF_env_raw, info_US)
    zmax_idx = int(depth_max_cm / np.max(info_US.d_sample) * naka_img.shape[0])
    naka_img = naka_img[0:zmax_idx,:]
    naka_img = gaussian_filter(naka_img, sigma = 2.0)
    mus_raw = _enforce_mean(naka_img, mu_s_mean_cm)
    US_img = US_img[0:zmax_idx,:]
    return US_img, mus_raw

def _tukey_weights(y_coords: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    L = y_coords.max() - y_coords.min()
    if L == 0: return np.ones_like(y_coords)
    u = (y_coords - y_coords.min()) / L
    w = np.ones_like(u)
    a = alpha / 2.0
    # rising taper
    left = u < a
    if np.any(left):
        w[left] = 0.5 * (1.0 + np.cos(np.pi * (2*u[left]/alpha - 1.0)))
    # falling taper
    right = u > 1 - a
    if np.any(right):
        w[right] = 0.5 * (1.0 + np.cos(np.pi * (2*(u[right]-1)/alpha + 1.0)))
    return w

def _estimate_fluence(mu_a_map_2d,
                      mu_s_map_2d,
                      grid_size: Tuple[int,int],
                      pixel_size_cm: Tuple[float,float],
                      compute_grad: bool = False,
                      adjoint_rhs: Optional[np.ndarray] = None):
    '''
    input maps are 1d flattened arrays
    grid size is (Nz, Nx)
    mu_map: (Nz, Ny, Nx)
    returns a flattened phi
    '''
    Nz, Nx = grid_size
    dz, dx = pixel_size_cm
    mu_a_map_2d = mu_a_map_2d.reshape(grid_size, order='C')
    mu_s_map_2d = mu_s_map_2d.reshape(grid_size, order='C')
    mu_a_map = np.repeat(mu_a_map_2d[:, np.newaxis, :], NY, axis=1)
    mu_sp_map = np.repeat(mu_s_map_2d[:, np.newaxis, :], NY, axis=1)
    voxel_size_cm = (dx, DY, dz)

    y_coords = np.linspace(-DY*(NY-1)/2, DY*(NY-1)/2, NY)
    w = _tukey_weights(y_coords, alpha=0.2)
    w /= (w.sum() + 1e-12)

    adj3d = None
    if adjoint_rhs is not None:
        adj = np.asarray(adjoint_rhs)
        # flattened 2D case: expand and apply weights
        if adj.ndim == 1 and adj.size == Nz * Nx:
            adj2d = adj.reshape((Nz, Nx), order='C')
            # adj3d[z,y,x] = adj2d[z,x] * w[y]
            # create shape (Nz, NY, Nx)
            adj3d = (adj2d[:, np.newaxis, :] * w[np.newaxis, :, np.newaxis])
        elif adj.ndim == 2 and adj.shape == (Nz, Nx):
            adj2d = adj
            adj3d = (adj2d[:, np.newaxis, :] * w[np.newaxis, :, np.newaxis])
        elif adj.ndim == 3:
            # possible orders: (Nz, NY, Nx) or (NY, Nx, Nz) or (Ny, Nx, Nz) or (Ny, Nz, Nx)
            if adj.shape == (Nz, NY, Nx):
                # multiply along y (axis=1) by w
                adj3d = adj * w[np.newaxis, :, np.newaxis]
            elif adj.shape == (NY, Nx, Nz):
                # convert to (Nz, NY, Nx) first
                adj_tmp = np.transpose(adj, (2, 0, 1))  # (Nz, NY, Nx)
                adj3d = adj_tmp * w[np.newaxis, :, np.newaxis]
            elif adj.shape == (NY, Nx, Nz):
                adj_tmp = np.transpose(adj, (2, 0, 1))
                adj3d = adj_tmp * w[np.newaxis, :, np.newaxis]
            elif adj.shape == (Ny := adj.shape[0], Nx := adj.shape[1], Nz := adj.shape[2]):  # fallback pattern
                # try to detect common other layouts; last-resort transpose
                adj_tmp = np.transpose(adj, (2, 0, 1))
                if adj_tmp.shape == (Nz, NY, Nx):
                    adj3d = adj_tmp * w[np.newaxis, :, np.newaxis]
                else:
                    raise ValueError("Unrecognized 3D adjoint_rhs shape.")
            else:
                raise ValueError("Unrecognized 3D adjoint_rhs shape; expected (Nz,NY,Nx) or transposable variants.")
        else:
            raise ValueError("adjoint_rhs shape not recognized. Provide flattened (Nz*Nx), (Nz,Nx), or 3D array.")


    phi3d , info , phi_grad3d = compute_phi_and_grad(mu_a_cm = mu_a_map,
                                                   mu_sp_cm = mu_sp_map,
                                                   voxel_size_cm = voxel_size_cm,
                                                   compute_grad = compute_grad,
                                                   adjoint_rhs = adj3d,
                                                   verbose = False)
    
    phi2d = np.tensordot(w, phi3d, axes=([0], [0]))   # shape (Nx, Nz)
    phi2d = phi2d.T   # shape (Nz, Nx)
    phi2d_flat = phi2d.flatten(order='C')
    phi_grad2d_flat = None
    if compute_grad:
        if phi_grad3d is None:
            raise RuntimeError("compute_grad=True but no gradient is returned")
        phi_grad2d = np.tensordot(w, phi_grad3d, axes=([0], [0]))   # shape (Nx, Nz)
        phi_grad2d = phi_grad2d.T
        phi_grad2d_flat = phi_grad2d.flatten(order='C')
    return phi2d_flat , info , phi_grad2d_flat

def _initialize_mua(input_dir: str,
                    info_PA: LinearSystemParam,
                    mu_a_mean_cm: float = 0.03,
                    depth_max_cm: float = Z_MAX_CM*1e-2,
                    dB_PA: int = 25) -> tuple[np.ndarray, np.ndarray]:
    _, RF_env_raw = pa_das_linear(input_dir, info_PA, 'hann', 'cf')
    min_dB = 10 ** (-dB_PA / 20.0)
    RF_env_norm = RF_env_raw / np.max(RF_env_raw) if np.max(RF_env_raw) != 0 else RF_env_raw
    with np.errstate(divide='ignore', invalid='ignore'):
        RF_log = (20.0 / dB_PA) * np.log10(np.maximum(RF_env_norm, 1e-20)) + 1.0
    RF_log[RF_env_norm < min_dB] = 0.0
        
    # Compute target image size
    z_range = np.max(info_PA.d_sample)
    xz_ratio = info_PA.FOV / z_range
    Lx = int(4 * info_PA.N_sc * 2.0/3.0)
    Lz = int(round(4 * info_PA.N_sc / xz_ratio)) if xz_ratio != 0 else RF_log.shape[0]
    zoom_z = Lz / RF_log.shape[0] if RF_log.shape[0] > 0 else 1.0
    zoom_x = Lx / RF_log.shape[1] if RF_log.shape[1] > 0 else 1.0
    RF_log_resized = zoom(RF_log, (zoom_z, zoom_x), order=1)
    idx_z = int(depth_max_cm / z_range * RF_log_resized.shape[0])
    PA_img = RF_log_resized[0:idx_z, :]
    mua_raw = gaussian_filter(PA_img, sigma=2.0)
    mua_raw = _enforce_mean(mua_raw, mu_a_mean_cm)
    return PA_img, mua_raw

def _convert_initialization_to_initial_maps(mu_a_init_2d: np.ndarray,
                                            mu_s_init_2d: np.ndarray,
                                            info_US: LinearSystemParam,
                                            info_PA: LinearSystemParam,
                                            bbox_cm: list[float],
                                            pixel_size_cm: list[float],
                                            verbose = True) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, zmin, zmax = bbox_cm
    dz, dx = pixel_size_cm
    fov_us = info_US.FOV
    fov_us *= 100 # in cm
    fov_pa = info_PA.FOV
    fov_pa *= 100 # in cm
    x_init_grid_us , x_init_grid_pa = np.linspace(-fov_us/2.0, fov_us/2.0, num = mu_s_init_2d.shape[1]) , np.linspace(-fov_pa/2.0, fov_pa/2.0, num = mu_a_init_2d.shape[1])
    z_init_grid_us , z_init_grid_pa = np.linspace(0.0, Z_MAX_CM, num = mu_s_init_2d.shape[0]) , np.linspace(0.0, Z_MAX_CM, num = mu_a_init_2d.shape[0])
    
    xq_grid , zq_grid = np.arange(xmin, xmax, dx) , np.arange(zmin, zmax, dz)
    Xq, Zq = np.meshgrid(xq_grid , zq_grid, indexing = 'xy')
    q_pts = np.column_stack([Zq.ravel() , Xq.ravel()])
    interp_mu_a = RegularGridInterpolator((z_init_grid_pa, x_init_grid_pa), mu_a_init_2d, bounds_error=False, fill_value=np.nan)
    mu_a_recon_map = interp_mu_a(q_pts)
    mu_a_recon_map = mu_a_recon_map.reshape(Zq.shape)
    interp_mu_s = RegularGridInterpolator((z_init_grid_us, x_init_grid_us), mu_s_init_2d, bounds_error=False, fill_value=np.nan)
    mu_s_recon_map = interp_mu_s(q_pts)
    mu_s_recon_map = mu_s_recon_map.reshape(Zq.shape)
    if verbose:
        print("mu maps initialized.")
    return mu_a_recon_map , mu_s_recon_map

def _update_mua(G: np.ndarray,
                phi_2d: np.ndarray,
                RF_env: np.ndarray,
                reg_lambda: float = 1e-3,
                maxiter: int = 100):
    '''
    Make sure G and phi_2d are on the same grid!!!
    '''
    Dphi = diags(phi_2d)
    A_sparse = G @ Dphi
    n = A_sparse.shape[1]
    if reg_lambda is None or reg_lambda == 0:
        sol = lsqr(A_sparse , RF_env , iter_lim = maxiter)
        x = sol[0]
    else:
        sqrt_l = np.sqrt(reg_lambda)
        I = identity(n, format=A_sparse.format)
        A_aug = vstack([A_sparse , sqrt_l*I])
        env_aug = np.concatenate([RF_env , np.zeros(n)])
        sol = lsqr(A_aug , env_aug , iter_lim = maxiter)
        x = sol[0]
    return x

def _estimate_phi_sensitivity(mu_a: np.ndarray,
                              mu_s: np.ndarray,
                              Ny: int = 7,
                              dmus_perturb: float = 1e-6,
                              subsample_ratio: float = 0.5):
    n = mu_s.size
    phi0 , _ = _estimate_fluence(mu_a , mu_s , Ny)
    if subsample_ratio is None or subsample_ratio >= 1.0:
        subsample_indices = np.arange(n)
    else:
        k = max(1 , int(n*subsample_ratio))
        subsample_indices = np.linspace(0, n-1, k, dtype=int)
    dphi_diag = np.zeros_like(phi0)
    for j in subsample_indices:
        mu_s[j] += dmus_perturb
        phi_p = _estimate_fluence(mu_a, mu_s)
        mu_s[j] -= dmus_perturb
        dphi_diag[j] = (phi_p[j] - phi0[j]) / dmus_perturb
    
    # fill missing indices
    if subsample_ratio is not None and subsample_ratio < 1.0:
        comp_idx = np.sort(subsample_indices)
        missing = np.setdiff1d(np.arange(n), comp_idx)
        for mi in missing:
            nearest = comp_idx[np.argmin(np.abs(comp_idx - mi))]
            dphi_diag[mi] = dphi_diag[nearest]
    return dphi_diag, phi0

def _make_2d_laplacian(nx: int, nz: int, dx: float, dz: float) -> csr_matrix:
    '''
    Return sparse 2D Laplacian (nz*nx by nz*nx) using central second differences.
    Grid ordering: flattened with row-major (z major) i.e., index = iz * nx + ix.
    '''
    def second_diff_1d(n, step):
        e = np.ones(n)
        diags_list = [ -2.0 * e, 1.0 * e[:-1], 1.0 * e[:-1] ]
        offsets = [0, 1, -1]
        return diags(diags_list, offsets, shape=(n, n), format='csr') / (step**2)
    D2x = second_diff_1d(nx, dx)  # acts on x
    D2z = second_diff_1d(nz, dz)  # acts on z
    Ix = identity(nx, format='csr')
    Iz = identity(nz, format='csr')
    Lap = kron(Iz, D2x, format='csr') + kron(D2z, Ix, format='csr')
    return Lap.tocsr()

def _fista_l1_lsqr(A, y, lam, n_iter=100, x0=None, L=None, nonneg=True, verbose=False):
    """
    FISTA for min_x 0.5||A x - y||^2 + lam * ||x||_1
    A: sparse matrix (m x n) supporting @ and .T @
    y: (m,)
    lam: scalar
    L: Lipschitz constant (||A||_2^2). If None estimate with power iteration.
    """
    m, n = A.shape
    if L is None:
        # power iteration on A^T A
        v = np.random.randn(n)
        v /= np.linalg.norm(v) + 1e-16
        for _ in range(20):
            Av = A @ v
            AtAv = A.T @ Av
            norm = np.linalg.norm(AtAv)
            if norm == 0:
                break
            v = AtAv / (norm + 1e-16)
        Av = A @ v
        L = (np.linalg.norm(Av)**2) / (np.linalg.norm(v)**2 + 1e-16)
        L = max(L, 1e-8)
    t_step = 1.0 / L

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    yk = x.copy()
    tk = 1.0

    def shrink(u, thresh):
        return np.sign(u) * np.maximum(np.abs(u) - thresh, 0.0)

    for k in range(n_iter):
        grad = A.T @ (A @ yk - y)  # n,
        x_new = shrink(yk - t_step * grad, lam * t_step)
        if nonneg:
            x_new = np.clip(x_new, 0.0, None)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * (tk**2))) / 2.0
        yk = x_new + ((tk - 1.0) / t_new) * (x_new - x)
        x = x_new
        tk = t_new
        if verbose and (k % 50 == 0 or k == n_iter - 1):
            obj = 0.5 * np.linalg.norm(A @ x - y)**2 + lam * np.linalg.norm(x, 1)
            print(f"FISTA iter {k:4d} obj {obj:.4e}")
    return x

def generate_mu_init(input_dir: str,
                     info_US: LinearSystemParam,
                     info_PA: LinearSystemParam,
                     mu_a_mean_cm: float,
                     mu_s_mean_cm: float,
                     bbox_cm: list[float],
                     pixel_size_cm: list[float],
                     ):
    US_img, mus_raw = _initialize_mus(input_dir , info_US , mu_s_mean_cm)
    PA_img, mua_raw = _initialize_mua(input_dir , info_PA , mu_a_mean_cm)
    mu_a_init , mu_s_init = _convert_initialization_to_initial_maps(mua_raw , mus_raw , info_US, info_PA, bbox_cm , pixel_size_cm)
    res = {"US image": US_img,
           "PAT image": PA_img,
           "mua0": mu_a_init,
           "mus0": mu_s_init}
    return res

def optimize_mu_maps(RF_data,
                     G,
                     mu_a_init,
                     mu_s_init,
                     global_mu_a_avg,
                     global_mu_s_avg,
                     pixel_size_cm,
                     grid_size,
                     n_iters,
                     reg_lambda_mu_a = 1e-3,
                     mu_s_step = 1e-2,
                     reg_lambda_mu_s = 1e-3,
                     dmus_perturb = 1e-3,
                     subsample_ratio: Optional[float] = 0.05,
                     verbose = True,
                     ):
    mu_a = mu_a_init.astype(np.float32).copy()
    mu_s = mu_s_init.astype(np.float32).copy()
    Nz,Nx = grid_size

    history = {'residual norm':[],
               'mu_a_mean':[],
               'mu_s_mean':[]}
    for it in range(1 , n_iters + 1):
        phi , _ , _ = _estimate_fluence(mu_a , mu_s , grid_size=grid_size, pixel_size_cm=pixel_size_cm)
        mu_a = _update_mua(G , phi , RF_data , reg_lambda=reg_lambda_mu_a)
        mu_a = np.clip(mu_a , 0.0 , None)
        mu_a = _enforce_mean(mu_a , global_mu_a_avg)
        phi , _ , _ = _estimate_fluence(mu_a , mu_s , grid_size=grid_size , pixel_size_cm=pixel_size_cm) # phi -> (Nz, Nx)
        
        residual = G @ (mu_a.ravel()*phi.ravel()) - RF_data.ravel()
        resnorm = 0.5 * np.linalg.norm(residual) ** 2
        history['residual_norm'].append(resnorm)
        history['mu_a_mean'].append(np.mean(mu_a))
        history['mu_s_mean'].append(np.mean(mu_s))

        if verbose:
            print(f"Iter {it:3d} | residual = {resnorm:.6e} | mean(mu_a)={np.mean(mu_a):.6e} mean(mu_s)={np.mean(mu_s):.6e}")

        adjoint_rhs_flat = -mu_a.ravel() * (G.T.dot(residual))
        adjoint_rhs = adjoint_rhs_flat.reshape(phi.shape , order='C') # (Nz, Nx)
        adjoint_rhs = np.repeat(adjoint_rhs[:,np.newaxis,:], NY, axis=1) # (Nz, Ny, Nx)
        phi , _ , grad_mu_s = _estimate_fluence(mu_a , mu_s , grid_size = grid_size , pixel_size_cm=pixel_size_cm , compute_grad=True, adjoint_rhs=adjoint_rhs)

        # approximate gradient wrt mu_s: grad_j ≈ df_dphi_j * dphi_diag_j + reg term
        grad_mu_s += reg_lambda_mu_s * mu_s

        # gradient step
        mu_s = mu_s - mu_s_step * grad_mu_s
        mu_s = np.clip(mu_s, 0.0, None)
        mu_s = _enforce_mean(mu_s, global_mu_s_avg)

    return mu_a , mu_s , history

def optimize_mu_maps_regularize(
    RF_data,           # y, shape (m,)
    G,                 # system matrix (m x n) sparse
    mu_a_init,         # (n,) initial
    mu_s_init,         # (n,) initial
    global_mu_a_avg,   # scalar
    global_mu_s_avg,   # scalar
    grid_shape: Tuple[int,int], #(Nz, Nx)
    grid_spacing: Tuple[float,float], # (dz, dx)
    n_iters: int = 20,
    # FISTA / mu_a params
    lam_mu_a: float = 1e-3,
    fista_iters: int = 10,
    fista_estimate_L: Optional[float] = None,
    # mu_s params
    mu_s_step: float = 1e-3,
    mu_s_reg_lambda: float = 1e-2,
    # Laplacian (either pass prebuilt Lap or pass grid params)
    Lap: Optional[csr_matrix] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    
    mu_a = mu_a_init.astype(np.float32).copy()
    mu_s = mu_s_init.astype(np.float32).copy()
    mu_a_prev = mu_a.copy()
    mu_s_prev = mu_s.copy()
    n = mu_a.size
    assert mu_s.size == n, "mu_s must match mu_a size"

    Nz , Nx = grid_shape
    dz , dx = grid_spacing
    if Nz*Nx != n:
        raise ValueError("grid shape does not match length of mu map")

    # Build Laplacian if not provided
    if Lap is None:
        Lap = _make_2d_laplacian(nx=Nx, nz=Nz, dx=dx, dz=dz)

    history = {'residual norm': [],
               'mua hist': [],
               'mus hist': [],
               'mua_frac_change':[],
               'mus_frac_change':[],
               'mua sparsity': []}

    RF_data = np.asarray(RF_data, dtype=np.float32).ravel()
    m = RF_data.size
    if G.shape[1] != n or G.shape[0] != m:
        raise ValueError(f"G should have dimensions (m,n) where n={n}, m={m}. But the acutual shape is {G.shape}")

    # Normalize G
    G_norms = np.linalg.norm(G, axis=0)
    G_norms[G_norms == 0] = 1.0
    G /= G_norms

    for it in range(1, n_iters + 1):
        phi, _, _ = _estimate_fluence(mu_a, mu_s, grid_size = grid_shape, pixel_size_cm = grid_spacing, compute_grad = False, adjoint_rhs = None)  # (n,)
        if verbose:
            print(f"Iter {it:3d}, first pass fluence solver completed")
        # Update mu_a: minimize 0.5||A x - y||^2 + lam||x||1, A = G @ diag(phi)
        A = G*phi[np.newaxis,:]
        # Run FISTA with warm-start and estimated L optional
        # mu_a = _fista_l1_lsqr(A, RF_data, lam=lam_mu_a, n_iter=fista_iters, x0=mu_a, L=fista_estimate_L, nonneg=True, verbose=False)
        solver = PATInverseSolver(A, RF_data)
        mu_a = solver.reconstruct(method='l1', lambda_reg = lam_mu_a , x0 = mu_a, verbose=False)
        mu_a *= G_norms
        mu_a = np.clip(mu_a, 0.0, None)
        mu_a = _enforce_mean(mu_a, global_mu_a_avg)
        if verbose:
            print(f"Iter {it:3d}, mua updated")
        
        # Recompute phi after mu_a update, and output mu_s gradient this time
        RF_pred = G.dot(mu_a * phi)
        residual = RF_pred - RF_data
        data_fidelity_norm = 0.5 * np.linalg.norm(residual)**2
        adjoint_rhs_flat = mu_a * (G.T.dot(residual))
        adjoint_rhs_2d = adjoint_rhs_flat.reshape(grid_shape , order='C') # (Nz, Nx)
        if verbose:
            print(f"Iter {it:3d}, source adjoint term computed")
        phi , _ , grad_mu_s_data = _estimate_fluence(mu_a , mu_s , grid_size = grid_shape , pixel_size_cm = grid_spacing , compute_grad = True, adjoint_rhs = adjoint_rhs_2d)

        if verbose:
            print(f"Iter {it:3d}, second pass fluence solver completed")
        if grad_mu_s_data is None:
            raise RuntimeError("No mu_s gradient is returned")
        grad_mu_s_data = np.asarray(grad_mu_s_data, dtype=np.float32).ravel()

        # Update mu_s with Laplacian regularization
        grad_smooth = mu_s_reg_lambda * (Lap @ (Lap @ mu_s))  # (n,)
        grad_mu_s = grad_mu_s_data + grad_smooth
        mu_s = mu_s - mu_s_step * grad_mu_s
        if verbose:
            print(f"Iter {it:3d}, mus updated")
        mu_s = np.clip(mu_s, 0.0, None)
        mu_s = _enforce_mean(mu_s, global_mu_s_avg)
        
        fractional_change_in_mu_a = np.linalg.norm(mu_a - mu_a_prev) / (np.linalg.norm(mu_a_prev) + 1e-14)
        fractinoal_change_in_mu_s = np.linalg.norm(mu_s - mu_s_prev) / (np.linalg.norm(mu_s_prev) + 1e-12)
        history['residual norm'].append(data_fidelity_norm)
        history['mua hist'].append(mu_a)
        history['mus hist'].append(mu_s)
        history['mua_frac_change'].append(fractional_change_in_mu_a)
        history['mus_frac_change'].append(fractinoal_change_in_mu_s)
        history['mua sparsity'].append(np.mean(mu_a == 0.0))
        mu_a_prev = mu_a.copy()
        mu_s_prev = mu_s.copy()
        if verbose:
            print(f"Iter {it:3d} | resobj {data_fidelity_norm:.6e} | Δμ_s={fractinoal_change_in_mu_s:.3e} | Δμ_a={fractional_change_in_mu_a:.4e} | sparsity(mu_a)={history['mu_a_sparsity'][-1]:.3f}")
    return mu_a, mu_s, history


import os
import sys
from pathlib import Path
import numpy as np
from .load_data_utils import LinearSystemParam
from .fluence_utils import compute_phi_heterogeneous
from .us_utils import pe_das_linear, nakagami_linear
from .pat_utils import pa_das_linear, pa_inverse_recon
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, identity, vstack
from scipy.sparse.linalg import lsqr

N_ELEMENTS = 128  # default number of elements for linear array
Z_MAX_CM = 4.0  # cm, default max depth for reconstruction
DY = 0.1 # cm, default pixel size along y direction (elevation)

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

def _estimate_fluence(mu_a_map_2d: np.ndarray,
                      mu_sp_map_2d: np.ndarray,
                      Ny: int = 7):
    '''
    mu_map_2d: (Nz, Nx)
    mu_map: (Nz, Ny, Nx)
    '''
    mu_a_map = np.repeat(mu_a_map_2d[:, np.newaxis, :], Ny, axis=1)
    mu_sp_map = np.repeat(mu_sp_map_2d[:, np.newaxis, :], Ny, axis=1)
    phi3d , info = compute_phi_heterogeneous(mu_a_cm=mu_a_map, mu_sp_cm=mu_sp_map, fiber_sigma_cm=0.03, verbose=True)
    ry = (Ny-1)//2 * DY
    y_coords = np.arange(-ry, ry, DY)
    w = _tukey_weights(y_coords, alpha=0.2)
    w /= (w.sum() + 1e-12)
    phi2d = np.tensordot(w, phi3d, axes=([0], [0]))   # shape (Nx, Nz)
    phi2d = phi2d.T   # shape (Nz, Nx)
    return phi2d , info

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
                     n_iters = 20,
                     reg_lambda_mu_a = 1e-3,
                     mu_s_step = 1e-2,
                     reg_lambda_mu_s = 1e-3,
                     dmus_perturb = 1e-6,
                     verbose = True,
                     ):
    mu_a = mu_a_init.astype(np.float32).copy()
    mu_s = mu_s_init.astype(np.float32).copy()
    history = {'residual norm':[],
               'mu_a_mean':[],
               'mu_s_mean':[]}
    for it in range(1 , n_iters + 1):
        phi , _ = _estimate_fluence(mu_a , mu_s)
        mu_a = _update_mua(G , phi , RF_data , reg_lambda=reg_lambda_mu_a)
        mu_a = np.clip(mu_a , 0.0 , None)
        mu_a = _enforce_mean(mu_a , global_mu_a_avg)
        phi = _estimate_fluence(mu_a , mu_s)
        
        modeled = (G @ (mu_a * phi))
        residual = modeled - RF_data
        resnorm = 0.5 * np.linalg.norm(residual) ** 2
        history['residual_norm'].append(resnorm)
        history['mu_a_mean'].append(np.mean(mu_a))
        history['mu_s_mean'].append(np.mean(mu_s))

        if verbose:
            print(f"Iter {it:3d} | residual = {resnorm:.6e} | mean(mu_a)={np.mean(mu_a):.6e} mean(mu_s)={np.mean(mu_s):.6e}")

        GTr = G.T @ residual  # shape (n,)
        df_dphi = GTr * mu_a
        
        dphi_diag, phi0 = _estimate_phi_sensitivity(mu_a, mu_s, eps=dmus_perturb)

        # approximate gradient wrt mu_s: grad_j ≈ df_dphi_j * dphi_diag_j + reg term
        grad_mu_s = df_dphi * dphi_diag
        grad_mu_s += reg_lambda_mu_s * mu_s

        # gradient step
        mu_s = mu_s - mu_s_step * grad_mu_s
        mu_s = np.clip(mu_s, 0.0, None)
        mu_s = _enforce_mean(mu_s, global_mu_s_avg)

    return mu_a , mu_s , history
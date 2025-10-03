import os, inspect, textwrap
import pandas as pd
from scipy.optimize import nnls, curve_fit
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
SCALE_E2MU = 2.303*150/64500/1000
#%% 
def _load_optical_extinction_coeff():
    '''
    Tabular data from https://omlc.org/spectra/hemoglobin/summary.html
    '''
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "artifacts", "optical_extinction_coeff.csv")
    df = pd.read_csv(csv_path, skiprows=[1]) # skips the 2nd row (units of each column)
    df.columns = ["wavelength", "hbo2", "hbr"]
    return df

def estimate_so2_from_dot(lambda_list: list , mua_list: list , verbose = False):
    '''
    From the mua estimated from DOT and its list of wavelengths, estimate oxy-/deoxy- hemoglobin decomposition
    The estimated concentrations are in [mmol/L]
    '''
    lambdas = np.asarray(lambda_list, dtype=float).ravel()
    mu_a = np.asarray(mua_list, dtype=float).ravel()
    df_ext = _load_optical_extinction_coeff()
    lambda_table    = df_ext["wavelength"].values
    eps_oxy_table   = df_ext['hbo2'].values
    eps_deoxy_table = df_ext['hbr'].values
    eps_oxy   = np.interp(lambdas, lambda_table, eps_oxy_table)
    eps_deoxy = np.interp(lambdas, lambda_table, eps_deoxy_table)
    E = np.vstack([eps_oxy , eps_deoxy]).T # (N_lambda, 2)
    x , residual_norm = nnls(E*SCALE_E2MU, mu_a)
    c_oxy , c_deoxy = float(x[0]) , float(x[1]) # unit: mmol/L
    residual_norm = float(residual_norm)

    thb = (c_oxy + c_deoxy)
    so2 = (c_oxy / thb) if thb != 0 else np.nan
    if verbose:
        print("Estimated hemoglobin concentrations from DOT")
        print(f"[HbO2] = {c_oxy:.6g} mmol/L")
        print(f"[Hb]   = {c_deoxy:.6g} mmol/L")
        print(f"sO2    = {so2*100:.2f}%")
    return {"c_oxy": c_oxy,
            "c_deoxy": c_deoxy,
            "THb": thb,
            "sO2": so2}

def query_bkg_mua_for_pa(lambda_list: list , c_oxy: float , c_deoxy: float):
    '''
    From [HbO2] and [HbR] estimated from DOT, find background mua at the PAT wavelengths
    '''
    lambdas = np.asarray(lambda_list, dtype=float).ravel()
    df_ext = _load_optical_extinction_coeff()
    lambda_table    = df_ext["wavelength"].values
    eps_oxy_table   = df_ext['hbo2'].values
    eps_deoxy_table = df_ext['hbr'].values
    eps_oxy   = np.interp(lambdas, lambda_table, eps_oxy_table)
    eps_deoxy = np.interp(lambdas, lambda_table, eps_deoxy_table)
    mu_a_pred = (eps_oxy * c_oxy + eps_deoxy * c_deoxy)*SCALE_E2MU
    return {"PAT wavelengths": lambdas,
            "Background mua": mu_a_pred}

def fit_bkg_mus_for_pa(lamda_list_dot: list, mus_list_dot: list, lambda_list_pa: list):
    '''
    Estimate background mus at the PAT wavelengths
    Citation: Jacques, S. L. (2013). Optical properties of biological tissues: a review. Physics in Medicine & Biology, 58(11), R37.
    '''
    lambdas_dot = np.asarray(lamda_list_dot, dtype=float).ravel()
    mus_dot     = np.asarray(mus_list_dot, dtype=float).ravel()
    lambdas_pa  = np.asarray(lambda_list_pa, dtype=float).ravel()
    # Scattering coefficient model definition
    def mus_model(lam, a, b):
        return a * (lam / 500.0) ** (-b)
    p0 = [mus_dot[0], 1.0] # Initial guess for (a, b)
    popt, _ = curve_fit(mus_model, lambdas_dot, mus_dot, p0=p0, maxfev=10000)
    a, b = popt
    mus_pred = mus_model(lambdas_pa, a, b) # Predict mus at new wavelengths
    return {"a": a,
            "b": b,
            "Background mus": mus_pred}
#%%
def compute_phi_homogeneous(mu_a: float,
                            mu_sp: float,
                            Lx_total: float = 5.0, Ly_total: float = 2.0, Lz_total: float = 4.0, 
                            dx: float = 0.1, dy: float = 0.1, dz: float = 0.1,
                            fiber_offsets = [(+0.5, +0.9), (+0.5, -0.9), (-0.5, +0.9), (-0.5, -0.9)], sigma_src: float = 0.2,
                            h_robin: float = 1.0,
                            verbose: bool = False,
                            ):
    nx , ny, nz = int(round(Lx_total / dx)) , int(round(Ly_total / dy)) , int(round(Lz_total / dz))
    N = nx * ny * nz
    x = np.linspace(-Lx_total/2, Lx_total/2, nx)
    y = np.linspace(-Ly_total/2, Ly_total/2, ny)
    z = np.linspace(0.0, Lz_total, nz)
    D = 1.0 / (3.0 * (mu_a + mu_sp))
    inv_dx2 , inv_dy2 , inv_dz2 = 1.0 / (dx * dx) , 1.0 / (dy * dy) , 1.0 / (dz * dz)

    X2d, Y2d = np.meshgrid(x, y, indexing='xy')
    S = np.zeros((ny, nx, nz), dtype=float)
    for xf, yf in fiber_offsets:
        r2 = (X2d - xf) ** 2 + (Y2d - yf) ** 2
        gauss = np.exp(-r2 / (2.0 * sigma_src ** 2))
        norm = np.sum(gauss) * dx * dy      # important: *dx*dy
        if norm <= 0:
            continue
        surface_density = gauss / norm     # integrates to ~1 over area (1/cm^2)
        S[:, :, 0] += surface_density / dz   # convert W/cm^2 -> W/cm^3

    def idx(ix, iy, iz): return ix + nx*(iy + ny*iz)

    data , rows , cols = [] , [] , []
    print("Building sparse matrix ...")
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                row = idx(ix, iy, iz)
                center = mu_a
                # x+
                if ix + 1 < nx:
                    rows.append(row); cols.append(idx(ix+1, iy, iz)); data.append(-D*inv_dx2)
                    center += D*inv_dx2
                else:
                    center += D*inv_dx2*(1.0 + h_robin * dx / D)
                # x-
                if ix - 1 >= 0:
                    rows.append(row); cols.append(idx(ix-1, iy, iz)); data.append(-D*inv_dx2)
                    center += D*inv_dx2
                else:
                    center += D*inv_dx2*(1.0 + h_robin * dx / D)
                # y+
                if iy + 1 < ny:
                    rows.append(row); cols.append(idx(ix, iy+1, iz)); data.append(-D*inv_dy2)
                    center += D*inv_dy2
                else:
                    center += D*inv_dy2*(1.0 + h_robin * dy / D)
                # y-
                if iy - 1 >= 0:
                    rows.append(row); cols.append(idx(ix, iy-1, iz)); data.append(-D*inv_dy2)
                    center += D*inv_dy2
                else:
                    center += D*inv_dy2*(1.0 + h_robin * dy / D)
                # z+
                if iz + 1 < nz:
                    rows.append(row); cols.append(idx(ix, iy, iz+1)); data.append(-D*inv_dz2)
                    center += D*inv_dz2
                else:
                    center += D*inv_dz2*(1.0 + h_robin * dz / D)
                # z-
                if iz - 1 >= 0:
                    rows.append(row); cols.append(idx(ix, iy, iz-1)); data.append(-D*inv_dz2)
                    center += D*inv_dz2
                else:
                    center += D*inv_dz2*(1.0 + h_robin * dz / D)

                rows.append(row); cols.append(row); data.append(center)
    
    A = spsp.csr_matrix((data, (rows, cols)), shape=(N, N))
    b = np.transpose(S, (1, 0, 2)).ravel(order='F')
    #b = S.ravel(order='F')

    if verbose:
        print(f"Solving diffusion equation on grid {nx}x{ny}x{nz} (N={N})")
        print(f"dx,dy,dz = {dx:.4g}, {dy:.4g}, {dz:.4g} cm; D={D:.4e} cm; h_robin={h_robin:.4g}")

    phi_flat = spla.spsolve(A, b)
    Phi = phi_flat.reshape((nx, ny, nz), order='F')
    Phi = np.transpose(Phi, (1, 0, 2))  # (ny, nx, nz)
    if verbose:
        print(f"Done. max(Phi) = {Phi.max():.4e} 1/cm^3, mean(Phi) = {Phi.mean():.4e} 1/cm^3")
        _plot_fluence_panels(Phi, x, y, z)

    return {"x grid": x,
            "y grid": y,
            "z grid": z,
            "fluence": Phi}

def _plot_fluence_panels(phi, X, Y, Z, src_positions=[(+0.5, +0.9), (+0.5, -0.9), (-0.5, +0.9), (-0.5, -0.9)], cmap = 'inferno'):
    ny, nx, nz = phi.shape
    yc, xc = ny // 2, nx // 2

    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    axs = axs.flatten()

    # Surface fluence (z=0)
    fig_tmp = phi[:, :, 1]
    im0 = axs[0].imshow(fig_tmp, extent=[X.min(), X.max(), Y.min(), Y.max()], vmax = np.percentile(fig_tmp, 99.5),
                        origin='lower', cmap=cmap)
    axs[0].set_title("Surface fluence (z = 0.075 cm)")
    axs[0].set_xlabel("x (cm)")
    axs[0].set_ylabel("y (cm)")
    for xf, yf in src_positions:
        axs[0].plot(xf, yf, marker='o', markersize=6, markeredgecolor='k', markerfacecolor='w', linewidth=1)
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label('W·cm\u207B\u00B3', fontsize=11)

    # Center x-z slice (at central y)
    fig_tmp = phi[yc, :, :].T
    im1 = axs[2].imshow(fig_tmp, extent=[X.min(), X.max(), Z.min(), Z.max()], vmax = np.percentile(fig_tmp, 99.5),
                        origin='lower', aspect='auto', cmap=cmap)
    axs[2].set_title("Center x–z slice (central y)")
    axs[2].set_xlabel("x (cm)")
    axs[2].set_ylabel("z (cm)")
    axs[2].invert_yaxis() 
    cbar1 = fig.colorbar(im1, ax=axs[2])
    cbar1.set_label('W·cm\u207B\u00B3', fontsize=11)

    # Average x-z over y
    fig_tmp = phi.mean(axis=0).T
    im2 = axs[3].imshow(fig_tmp, extent=[X.min(), X.max(), Z.min(), Z.max()],vmax = np.percentile(fig_tmp, 99.5),
                        origin='lower', aspect='auto', cmap=cmap)
    axs[3].set_title("x–z averaged over y")
    axs[3].set_xlabel("x (cm)")
    axs[3].set_ylabel("z (cm)")
    axs[3].invert_yaxis() 
    cbar2 = fig.colorbar(im2, ax=axs[3])
    cbar2.set_label('W·cm\u207B\u00B3', fontsize=11)

    # Center y-z slice (at central x)
    fig_tmp = phi[:, xc, :].T
    im3 = axs[4].imshow(fig_tmp, extent=[Y.min(), Y.max(), Z.min(), Z.max()],vmax = np.percentile(fig_tmp, 99.5),
                        origin='lower', aspect='auto', cmap=cmap)
    axs[4].set_title("Center y–z slice (central x)")
    axs[4].set_xlabel("y (cm)")
    axs[4].set_ylabel("z (cm)")
    axs[4].invert_yaxis() 
    cbar3 = fig.colorbar(im3, ax=axs[4])
    cbar3.set_label('W·cm\u207B\u00B3', fontsize=11)

    # Average y-z over x
    fig_tmp = phi.mean(axis=1).T
    im4 = axs[5].imshow(fig_tmp, extent=[Y.min(), Y.max(), Z.min(), Z.max()],vmax = np.percentile(fig_tmp, 99.5),
                        origin='lower', aspect='auto', cmap=cmap)
    axs[5].set_title("y–z averaged over x")
    axs[5].set_xlabel("y (cm)")
    axs[5].set_ylabel("z (cm)")
    axs[5].invert_yaxis() 
    cbar4 = fig.colorbar(im4, ax=axs[5])
    cbar4.set_label('W·cm\u207B\u00B3', fontsize=11)

    # Hide empty 6th subplot
    axs[1].axis("off")

    fig.suptitle("Fluence in homogeneous medium (Diffusion equation with Robin BC)", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


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

def estimate_mu_a_from_pa_with_ywindow(
    pa_zx: np.ndarray,
    fluence3d: np.ndarray,
    mu_a_global: float,
    y_coords: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    smoothing_sigma: float = 0.25,
    clip_min: float = 0.0,
) -> Dict[str, np.ndarray]:
    '''
    pa_zx: (Nz, Nx)
    fluence3d: (Ny, Nx, Nz)
    '''
    ny, nx, nz = fluence3d.shape
    assert(pa_zx.shape == (nz,nx))
    assert(y_coords.size == ny)
    w = _tukey_weights(y_coords, alpha=0.2)
    w /= (w.sum() + eps)

    flu_2d = np.tensordot(w, fluence3d, axes=([0], [0]))   # shape (Nx, Nz)
    flu_2d = flu_2d.T   # shape (Nz, Nx)

    mask = (flu_2d > eps)
    safe_flu = flu_2d.copy()
    safe_flu[~mask] = np.nan
    mu_raw = pa_zx / (safe_flu + eps)
    mu_raw_filled = np.nan_to_num(mu_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # compute scaling to enforce mu_a_global (from DOT)
    mu_raw_mean = mu_raw_filled[mask].mean()
    scale = float(mu_a_global) / float(mu_raw_mean)
    mu_a_map = mu_raw_filled * scale
    if smoothing_sigma and smoothing_sigma > 0:
        mu_a_map = gaussian_filter(mu_a_map, sigma=smoothing_sigma)

    mu_mean_after = mu_a_map[mask].mean()
    renorm = float(mu_a_global) / float(mu_mean_after)
    mu_a_map *= renorm
    scale *= renorm
    mu_a_map = np.maximum(mu_a_map, clip_min)

    return {
        "Scaled PAT mu_a": mu_a_map,
        "Raw PAT mu_a": mu_raw_filled,
        "Elevational window": w,
        "Background DOT mu_a": mu_a_global,
        "Scan plane fluence": flu_2d,
    }
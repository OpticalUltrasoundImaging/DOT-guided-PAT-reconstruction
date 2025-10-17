import os, inspect, textwrap
import pandas as pd
from scipy.optimize import nnls, curve_fit
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
import pyamg
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
                            Lx_total: float = 5.0, Ly_total: float = 1.0, Lz_total: float = 4.0, 
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

def _map_pos_to_phys_and_voxel_center_cm(pos, shape, voxel_size_cm):
    Nz, Ny, Nx = shape
    dx, dy, dz = voxel_size_cm

    width_x = Nx * dx   # cm
    width_y = Ny * dy
    depth_z = Nz * dz

    x_phys, y_phys = pos
    z_phys = - depth_z / 2.0
    
    x_v = (x_phys + width_x/2.0)/dx - 0.5
    y_v = (y_phys + width_y/2.0)/dy - 0.5
    z_v = (z_phys + depth_z/2.0)/dz - 0.5
    return (x_phys, y_phys, z_phys), (z_v, y_v, x_v)

def compute_phi_heterogeneous(mu_a_cm, mu_sp_cm,
                              voxel_size_cm=(0.1, 0.1, 0.1),
                              fiber_positions=[(+0.5, +0.9), (+0.5, -0.9), (-0.5, +0.9), (-0.5, -0.9)],
                              fiber_sigma_cm = 0.3,  # in cm
                              tol=1e-8, maxiter=2000,
                              use_pyamg=True, verbose=False,
                              surface_only=True):

    if mu_a_cm.shape != mu_sp_cm.shape:
        raise ValueError("mu_a_cm and mu_sp_cm must have same shape (Nz,Ny,Nx).")
    Nz, Ny, Nx = mu_a_cm.shape
    dx, dy, dz = voxel_size_cm
    inv_dx2 , inv_dy2 , inv_dz2 = 1.0 / (dx * dx) , 1.0 / (dy * dy) , 1.0 / (dz * dz)

    # diffusion coefficient D as 3D array (cm)
    D = 1.0 / (3.0 * np.maximum(mu_sp_cm + mu_a_cm, 1e-12))   # shape (Nz,Ny,Nx), units cm
    # build physical coords (cm) of voxel centers on top slice
    width_x = Nx * dx
    width_y = Ny * dy
    depth_z = Nz * dz
    x_coords_phys = (np.arange(Nx) + 0.5) * dx - width_x/2.0  # shape (Nx,)
    y_coords_phys = (np.arange(Ny) + 0.5) * dy - width_y/2.0  # shape (Ny,)
    # create 2D mesh (Ny, Nx) matching array indexing [y,x]
    X_phys = np.repeat(x_coords_phys[None, :], Ny, axis=0)    # (Ny, Nx)
    Y_phys = np.repeat(y_coords_phys[:, None], Nx, axis=1)    # (Ny, Nx)

    # prepare source S (Nz, Ny, Nx)
    S = np.zeros_like(mu_a_cm, dtype=float)
    for p in fiber_positions:
        (x_phys, y_phys, _), (z_v, _, _) = _map_pos_to_phys_and_voxel_center_cm(p, (Nz,Ny,Nx), voxel_size_cm)

        if surface_only:
            z_slice = 0
        else:
            z_slice = int(round(z_v))
            z_slice = np.clip(z_slice, 0, Nz-1)

        # compute 2D Gaussian in physical cm (no integer rounding of center)
        r2 = (X_phys - x_phys)**2 + (Y_phys - y_phys)**2   # cm^2
        kernel2d = np.exp(-0.5 * r2 / (fiber_sigma_cm**2))
        norm = np.sum(kernel2d) * dx * dy
        if norm <= 0:
            continue
        surface_density = kernel2d / norm     # integrates to ~1 over area (1/cm^2)
        S[z_slice, :, :] += surface_density / dz

    # Assemble sparse matrix A (variable-coefficient 7-point stencil)
    def idx(z,y,x): return (z * Ny + y) * Nx + x
    N = Nz * Ny * Nx
    eps = 1e-18
    # face-centered (harmonic mean) D on faces
    D_x_face = np.zeros((Nz, Ny, max(0, Nx-1)))
    D_y_face = np.zeros((Nz, max(0, Ny-1), Nx))
    D_z_face = np.zeros((max(0, Nz-1), Ny, Nx))
    if Nx > 1:
        D_x_face[:] = 2.0 * (D[:,:,:-1] * D[:,:,1:]) / (D[:,:,:-1] + D[:,:,1:] + eps)
    if Ny > 1:
        D_y_face[:] = 2.0 * (D[:,:-1,:] * D[:,1:,:]) / (D[:,:-1,:] + D[:,1:,:] + eps)
    if Nz > 1:
        D_z_face[:] = 2.0 * (D[:-1,:,:] * D[1:,:,:]) / (D[:-1,:,:] + D[1:,:,:] + eps)

    rows = []; cols = []; vals = []
    h_robin = 1.0
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                center = idx(z,y,x)
                diag = mu_a_cm[z,y,x]  # cm^-1

                # ------ X-direction ------
                # face between (x) and (x+1) is indexed at x in D_x_face (valid for x=0..Nx-2)
                if x+1 < Nx and Nx > 1:
                    Dfp = D_x_face[z,y,x]
                    coeff = Dfp / (dx*dx)
                    rows.append(center); cols.append(idx(z,y,x+1)); vals.append(-coeff)
                    diag += coeff
                else:
                    # use interior-adjacent face for right boundary (x == Nx-1)
                    if Nx > 1:
                        face_idx = min(x-1, Nx-2) if x-1 >= 0 else 0
                        Dfp = D_x_face[z,y,face_idx]
                    else:
                        # degenerate single-column: fallback to cell D
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dx*dx) * (1.0 + h_robin * dz / Dfp)

                # x-1 neighbor
                if x-1 >= 0 and Nx > 1:
                    Dfm = D_x_face[z,y,x-1]
                    coeff = Dfm / (dx*dx)
                    rows.append(center); cols.append(idx(z,y,x-1)); vals.append(-coeff)
                    diag += coeff
                else:
                    # left boundary: use adjacent face (face between 0 and 1 is D_x_face[...,0]) or fallback
                    if Nx > 1:
                        face_idx = 0 if x == 0 else max(0, x-1)
                        Dfm = D_x_face[z,y,face_idx]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dx*dx) * (1.0 + h_robin * dz / Dfm)

                # ------ Y-direction ------
                if y+1 < Ny and Ny > 1:
                    Dfp = D_y_face[z,y,x]
                    coeff = Dfp / (dy*dy)
                    rows.append(center); cols.append(idx(z,y+1,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Ny > 1:
                        face_idx = min(y-1, Ny-2) if y-1 >= 0 else 0
                        Dfp = D_y_face[z,face_idx,x]
                    else:
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dy*dy) * (1.0 + h_robin * dz / Dfp)

                if y-1 >= 0 and Ny > 1:
                    Dfm = D_y_face[z,y-1,x]
                    coeff = Dfm / (dy*dy)
                    rows.append(center); cols.append(idx(z,y-1,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Ny > 1:
                        face_idx = 0 if y == 0 else max(0, y-1)
                        Dfm = D_y_face[z,face_idx,x]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dy*dy) * (1.0 + h_robin * dz / Dfm)

                # ------ Z-direction ------
                if z+1 < Nz and Nz > 1:
                    Dfp = D_z_face[z,y,x]
                    coeff = Dfp / (dz*dz)
                    rows.append(center); cols.append(idx(z+1,y,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Nz > 1:
                        face_idx = min(z-1, Nz-2) if z-1 >= 0 else 0
                        Dfp = D_z_face[face_idx,y,x]
                    else:
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dz*dz) * (1.0 + h_robin * dz / Dfp)

                if z-1 >= 0 and Nz > 1:
                    Dfm = D_z_face[z-1,y,x]
                    coeff = Dfm / (dz*dz)
                    rows.append(center); cols.append(idx(z-1,y,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Nz > 1:
                        face_idx = 0 if z == 0 else max(0, z-1)
                        Dfm = D_z_face[face_idx,y,x]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dz*dz) * (1.0 + h_robin * dz / Dfm)

                rows.append(center); cols.append(center); vals.append(diag)

    # final matrix
    A = spsp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    b = S.ravel()

    use_solver = 'cg_jacobi'
    cg_info = None
    phi_flat = None

    if use_pyamg:
        try:
            import pyamg
            if verbose: print("Building pyamg multigrid hierarchy...")
            ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric', max_coarse=500)
            try:
                if verbose: print("Solving with pyamg.ml.solve(...)")
                phi_flat = ml.solve(b, tol=tol, maxiter=maxiter, accel=None)
                use_solver = 'pyamg_full_solve'
            except Exception as e_solve:
                if verbose: print("pyamg.ml.solve failed; falling back to CG with pyamg preconditioner:", e_solve)
                P = ml.aspreconditioner(cycle='V')
                phi_flat, cg_info = spla.cg(A, b, tol=tol, maxiter=maxiter, M=P)
                use_solver = 'cg_with_pyamg_prec'
        except Exception as e:
            if verbose: print("pyamg unavailable or failed to build ml:", e)
            use_solver = 'cg_jacobi'

    if phi_flat is None:
        if verbose: print("Using Jacobi-preconditioned CG (fallback).")
        M_diag = A.diagonal()
        M_inv = 1.0 / (M_diag + 1e-18)
        M = spla.LinearOperator((N,N), lambda x: M_inv * x)
        phi_flat, cg_info = spla.cg(A, b, tol=tol, maxiter=maxiter, M=M)
        use_solver = 'cg_jacobi'

    if cg_info is not None and cg_info != 0 and verbose:
        print("CG finished with info =", cg_info, "(0 means converged)")

    phi = phi_flat.reshape((Nz, Ny, Nx))
    phi = np.transpose(phi, (1, 2, 0))  # (ny, nx, nz)
    if verbose:
        print(f"Done. max(Phi) = {phi.max():.4e} 1/cm^3, mean(Phi) = {phi.mean():.4e} 1/cm^3")
        _plot_fluence_panels(phi, x_coords_phys, y_coords_phys, np.linspace(0, depth_z, Nz))

    info = {'solver': use_solver, 'cg_info': int(cg_info) if cg_info is not None else None}

    return phi, info

def compute_phi_and_grad(mu_a_cm, 
                         mu_sp_cm,
                         voxel_size_cm=(0.1, 0.1, 0.1),
                         fiber_positions=[(+0.5, +0.9), (+0.5, -0.9), (-0.5, +0.9), (-0.5, -0.9)],
                         fiber_sigma_cm=0.03,
                         tol=1e-8, maxiter=2000,
                         use_pyamg=True, verbose=False,
                         compute_grad=False,
                         adjoint_rhs=None,
                         dtype=np.float64):

    if mu_a_cm.shape != mu_sp_cm.shape:
        raise ValueError("mu_a_cm and mu_sp_cm must have same shape (Nz,Ny,Nx).")
    Nz, Ny, Nx = mu_a_cm.shape
    dx, dy, dz = voxel_size_cm
    inv_dx2, inv_dy2, inv_dz2 = 1.0/(dx*dx), 1.0/(dy*dy), 1.0/(dz*dz)
    eps = 1e-18
    h_robin = 1.0

    mu_a = np.asarray(mu_a_cm, dtype=dtype)
    mu_s = np.asarray(mu_sp_cm, dtype=dtype)

    # Diffusion coefficient D
    D = 1.0 / (3.0 * np.maximum(mu_s + mu_a, 1e-12))    # (Nz,Ny,Nx)
    dD = - (D * D) / 3.0 # derivative of D wrt mu_s

    # construct source S (Nz,Ny,Nx)
    width_x , width_y = Nx * dx , Ny * dy
    x_coords_phys = (np.arange(Nx) + 0.5) * dx - width_x/2.0
    y_coords_phys = (np.arange(Ny) + 0.5) * dy - width_y/2.0
    X_phys = np.repeat(x_coords_phys[None, :], Ny, axis=0)    # (Ny, Nx)
    Y_phys = np.repeat(y_coords_phys[:, None], Nx, axis=1)    # (Ny, Nx)

    S = np.zeros((Nz, Ny, Nx), dtype=dtype)
    for p in fiber_positions:
        (x_phys, y_phys, _), (_, _, _) = _map_pos_to_phys_and_voxel_center_cm(p, (Nz,Ny,Nx), voxel_size_cm)
        r2 = (X_phys - x_phys)**2 + (Y_phys - y_phys)**2
        kernel2d = np.exp(-0.5 * r2 / (fiber_sigma_cm**2))
        norm = np.sum(kernel2d) * dx * dy
        if norm <= 0:
            continue
        surface_density = kernel2d / norm
        S[0, :, :] += surface_density / dz

    # Build face-centered D (harmonic means)
    if Nx > 1:
        D_x_face = 2.0 * (D[:, :, :-1] * D[:, :, 1:]) / (D[:, :, :-1] + D[:, :, 1:] + eps)
    else:
        D_x_face = np.zeros((Nz, Ny, 0), dtype=dtype)
    if Ny > 1:
        D_y_face = 2.0 * (D[:, :-1, :] * D[:, 1:, :]) / (D[:, :-1, :] + D[:, 1:, :] + eps)
    else:
        D_y_face = np.zeros((Nz, 0, Nx), dtype=dtype)
    if Nz > 1:
        D_z_face = 2.0 * (D[:-1, :, :] * D[1:, :, :]) / (D[:-1, :, :] + D[1:, :, :] + eps)
    else:
        D_z_face = np.zeros((0, Ny, Nx), dtype=dtype)

    def idx(z,y,x): return (z * Ny + y) * Nx + x
    N = Nz * Ny * Nx
    rows = []; cols = []; vals = []
    h_robin = 1.0
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                center = idx(z,y,x)
                diag = mu_a_cm[z,y,x]  # cm^-1

                # ------ X-direction ------
                # face between (x) and (x+1) is indexed at x in D_x_face (valid for x=0..Nx-2)
                if x+1 < Nx and Nx > 1:
                    Dfp = D_x_face[z,y,x]
                    coeff = Dfp / (dx*dx)
                    rows.append(center); cols.append(idx(z,y,x+1)); vals.append(-coeff)
                    diag += coeff
                else:
                    # use interior-adjacent face for right boundary (x == Nx-1)
                    if Nx > 1:
                        face_idx = min(x-1, Nx-2) if x-1 >= 0 else 0
                        Dfp = D_x_face[z,y,face_idx]
                    else:
                        # degenerate single-column: fallback to cell D
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dx*dx) * (1.0 + h_robin * dz / Dfp)

                # x-1 neighbor
                if x-1 >= 0 and Nx > 1:
                    Dfm = D_x_face[z,y,x-1]
                    coeff = Dfm / (dx*dx)
                    rows.append(center); cols.append(idx(z,y,x-1)); vals.append(-coeff)
                    diag += coeff
                else:
                    # left boundary: use adjacent face (face between 0 and 1 is D_x_face[...,0]) or fallback
                    if Nx > 1:
                        face_idx = 0 if x == 0 else max(0, x-1)
                        Dfm = D_x_face[z,y,face_idx]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dx*dx) * (1.0 + h_robin * dz / Dfm)

                # ------ Y-direction ------
                if y+1 < Ny and Ny > 1:
                    Dfp = D_y_face[z,y,x]
                    coeff = Dfp / (dy*dy)
                    rows.append(center); cols.append(idx(z,y+1,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Ny > 1:
                        face_idx = min(y-1, Ny-2) if y-1 >= 0 else 0
                        Dfp = D_y_face[z,face_idx,x]
                    else:
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dy*dy) * (1.0 + h_robin * dz / Dfp)

                if y-1 >= 0 and Ny > 1:
                    Dfm = D_y_face[z,y-1,x]
                    coeff = Dfm / (dy*dy)
                    rows.append(center); cols.append(idx(z,y-1,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Ny > 1:
                        face_idx = 0 if y == 0 else max(0, y-1)
                        Dfm = D_y_face[z,face_idx,x]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dy*dy) * (1.0 + h_robin * dz / Dfm)

                # ------ Z-direction ------
                if z+1 < Nz and Nz > 1:
                    Dfp = D_z_face[z,y,x]
                    coeff = Dfp / (dz*dz)
                    rows.append(center); cols.append(idx(z+1,y,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Nz > 1:
                        face_idx = min(z-1, Nz-2) if z-1 >= 0 else 0
                        Dfp = D_z_face[face_idx,y,x]
                    else:
                        Dfp = D[z,y,x]
                    Dfp = max(Dfp, eps)
                    diag += Dfp / (dz*dz) * (1.0 + h_robin * dz / Dfp)

                if z-1 >= 0 and Nz > 1:
                    Dfm = D_z_face[z-1,y,x]
                    coeff = Dfm / (dz*dz)
                    rows.append(center); cols.append(idx(z-1,y,x)); vals.append(-coeff)
                    diag += coeff
                else:
                    if Nz > 1:
                        face_idx = 0 if z == 0 else max(0, z-1)
                        Dfm = D_z_face[face_idx,y,x]
                    else:
                        Dfm = D[z,y,x]
                    Dfm = max(Dfm, eps)
                    diag += Dfm / (dz*dz) * (1.0 + h_robin * dz / Dfm)

                rows.append(center); cols.append(center); vals.append(diag)

    # final matrix
    A = spsp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    b = S.ravel(order='C')

    # --- Solve forward: try pyamg else CG w/ Jacobi precond ---
    use_solver = 'cg_jacobi'
    cg_info = None
    phi_flat = None
    ml = None

    if use_pyamg:
        try:
            import pyamg
            if verbose: print("Building pyamg multigrid hierarchy...")
            ml = pyamg.smoothed_aggregation_solver(A, symmetry='symmetric', max_coarse=500)
            try:
                if verbose: print("Solving forward with pyamg.ml.solve(...)")
                phi_flat = ml.solve(b, tol=tol, maxiter=maxiter, accel=None)
                use_solver = 'pyamg_full_solve'
            except Exception as e_solve:
                if verbose: print("pyamg.ml.solve failed; falling back to CG with pyamg preconditioner:", e_solve)
                P = ml.aspreconditioner(cycle='V')
                phi_flat, cg_info = spla.cg(A, b, tol=tol, maxiter=maxiter, M=P)
                use_solver = 'cg_with_pyamg_prec'
        except Exception as e:
            if verbose: print("pyamg unavailable or failed to build ml:", e)
            use_solver = 'cg_jacobi'

    if phi_flat is None:
        if verbose: print("Using Jacobi-preconditioned CG (fallback for forward).")
        M_diag = A.diagonal()
        M_inv = 1.0 / (M_diag + 1e-18)
        M = spla.LinearOperator((N,N), lambda x: M_inv * x)
        phi_flat, cg_info = spla.cg(A, b, tol=tol, maxiter=maxiter, M=M)
        use_solver = 'cg_jacobi'

    if cg_info is not None and cg_info != 0 and verbose:
        print("Forward CG finished with info =", cg_info, "(0 means converged)")

    phi = phi_flat.reshape((Nz, Ny, Nx), order='C')
    phi_out = np.transpose(phi, (1, 2, 0))

    info = {'solver': use_solver, 'cg_info': int(cg_info) if cg_info is not None else None}

    if not compute_grad:
        if verbose:
            print(f"Done (no grad). max(Phi) = {phi_out.max():.4e} 1/cm^3, mean(Phi) = {phi_out.mean():.4e} 1/cm^3")
            _plot_fluence_panels(phi_out, x_coords_phys, y_coords_phys, np.linspace(0, Nz*dz, Nz))
        return phi_out, info, None

    # compute adjoint w: require adjoint_rhs provided
    if adjoint_rhs is None:
        raise ValueError("compute_grad=True requires adjoint_rhs (lambda = dJ/dphi).")

    lam = np.asarray(adjoint_rhs)
    if lam.ndim == 1:
        lam_flat = lam
    else:
        # assume shape (Nz,Ny,Nx) or (Ny,Nx,Nz) — prefer (Nz,Ny,Nx)
        if lam.shape == (Nz, Ny, Nx):
            lam_flat = lam.ravel(order='C')
        elif lam.shape == (Ny, Nx, Nz):
            lam_flat = np.transpose(lam, (2,0,1)).ravel(order='C')
        else:
            raise ValueError("adjoint_rhs shape not recognized; expected flattened or (Nz,Ny,Nx) or (Ny,Nx,Nz).")

    # Solve A w = lam_flat (symmetric A)
    w_flat = None
    w_info = None
    if use_solver == 'pyamg_full_solve' and ml is not None:
        try:
            if verbose: print("Solving adjoint with pyamg.ml.solve(...)")
            w_flat = ml.solve(lam_flat, tol=tol, maxiter=maxiter, accel=None)
        except Exception as e:
            if verbose: print("pyamg.ml.solve failed on adjoint; falling back to CG:", e)
            P = ml.aspreconditioner(cycle='V') if ml is not None else None
            if P is not None:
                w_flat, w_info = spla.cg(A, lam_flat, tol=tol, maxiter=maxiter, M=P)
            else:
                M_diag = A.diagonal()
                M_inv = 1.0/(M_diag + 1e-18)
                M = spla.LinearOperator((N,N), lambda x: M_inv * x)
                w_flat, w_info = spla.cg(A, lam_flat, tol=tol, maxiter=maxiter, M=M)
    else:
        # use available preconditioner if any
        if ml is not None:
            try:
                P = ml.aspreconditioner(cycle='V')
                w_flat, w_info = spla.cg(A, lam_flat, tol=tol, maxiter=maxiter, M=P)
            except Exception:
                M_diag = A.diagonal()
                M_inv = 1.0/(M_diag + 1e-18)
                M = spla.LinearOperator((N,N), lambda x: M_inv * x)
                w_flat, w_info = spla.cg(A, lam_flat, tol=tol, maxiter=maxiter, M=M)
        else:
            M_diag = A.diagonal()
            M_inv = 1.0/(M_diag + 1e-18)
            M = spla.LinearOperator((N,N), lambda x: M_inv * x)
            w_flat, w_info = spla.cg(A, lam_flat, tol=tol, maxiter=maxiter, M=M)

    if w_info is not None and w_info != 0 and verbose:
        print("Adjoint CG finished with info =", w_info, "(0 means converged)")

    w = w_flat.reshape((Nz, Ny, Nx), order='C')

    # vectorized gradient accumulation over faces
    grad = np.zeros_like(mu_s, dtype=np.float32)

    # X-faces
    if Nx > 1:
        u , v = D[:, :, :-1] , D[:, :, 1:]    # D at left and right cell
        du,dv = dD[:, :, :-1], dD[:, :, 1:]  # derivative at left and right
        denom = (u + v) + eps
        denom2 = denom * denom
        # ∂G/∂u = 2 v^2 /(u+v)^2; ∂G/∂v = 2 u^2 /(u+v)^2
        C_left  = 2.0 * du * (v * v) / denom2 * inv_dx2
        C_right = 2.0 * dv * (u * u) / denom2 * inv_dx2
        phi_L , phi_R = phi[:, :, :-1] , phi[:, :, 1:]
        w_L , w_R = w[:, :, :-1] , w[:, :, 1:]
        T = (w_L - w_R) * (phi_R - phi_L)
        grad[:, :, :-1] += - C_left  * T
        grad[:, :, 1: ] += - C_right * T

    # Y-faces
    if Ny > 1:
        u , v = D[:, :-1, :] , D[:, 1:, :]
        du,dv = dD[:, :-1, :], dD[:, 1:, :]
        denom = (u + v) + eps
        denom2 = denom * denom

        C_front = 2.0 * du * (v * v) / denom2 * inv_dy2
        C_back  = 2.0 * dv * (u * u) / denom2 * inv_dy2
        phi_F , phi_B = phi[:, :-1, :] , phi[:, 1:, :]
        w_F , w_B = w[:, :-1, :] , w[:, 1:, :]
        T = (w_F - w_B) * (phi_B - phi_F)
        grad[:, :-1, :] += - C_front * T
        grad[:, 1:,  :] += - C_back  * T

    # Z-faces
    if Nz > 1:
        u , v = D[:-1, :, :] , D[1:, :, :]
        du,dv = dD[:-1, :, :], dD[1:, :, :]
        denom = (u + v) + eps
        denom2 = denom * denom

        C_top = 2.0 * du * (v * v) / denom2 * inv_dz2
        C_bot = 2.0 * dv * (u * u) / denom2 * inv_dz2

        phi_T , phi_B = phi[:-1, :, :] , phi[1:, :, :]
        w_T , w_B = w[:-1, :, :] , w[1:, :, :]
        T = (w_T - w_B) * (phi_B - phi_T)
        grad[:-1, :, :] += - C_top * T
        grad[1:,  :, :] += - C_bot * T

    grad = np.transpose(grad, (1,2,0))
    if verbose:
        print(f"Done. max(Phi) = {phi_out.max():.4e} 1/cm^3, mean(Phi) = {phi_out.mean():.4e} 1/cm^3")
        _plot_fluence_panels(phi_out, x_coords_phys, y_coords_phys, np.linspace(0, Nz*dz, Nz))

    return phi_out, info, grad

import os, inspect, textwrap
import pandas as pd
from scipy.optimize import nnls, curve_fit
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

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

def _plot_fluence_panels(Phi, x, y, z, fiber_offsets=[(+0.5, +0.9), (+0.5, -0.9), (-0.5, +0.9), (-0.5, -0.9)], figsize=(10,12), cmap='turbo'):
    ny, nx, nz = Phi.shape

    # Prepare derived images
    surface = Phi[:, :, 0]                       # ny x nx (y rows, x cols)
    iy_center = ny // 2
    ix_center = nx // 2
    xz_center = Phi[iy_center, :, :].T           # (nz, nx)
    xz_avg = Phi.mean(axis=0).T                  # (nz, nx)
    yz_center = Phi[:, ix_center, :].T           # (nz, ny)
    yz_avg = Phi.mean(axis=1).T                  # (nz, ny)

    # Shared color scale: use robust percentile to avoid outliers dominating
    vmin = np.percentile(Phi, 1.0)
    vmax = np.percentile(Phi, 99.9)
    if vmin == vmax:
        vmin = Phi.min()
        vmax = Phi.max()

    # Create figure + grid
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1.0, 0.9, 0.9], hspace=0.25, wspace=0.18)

    # Row 1: Surface (span both columns)
    ax_surf = fig.add_subplot(gs[0, :])
    im0 = ax_surf.imshow(surface, origin='lower',
                         extent=[x[0], x[-1], y[0], y[-1]],
                         aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear')
    ax_surf.set_title('Surface fluence (z = 0 cm)', fontsize=14, fontweight='semibold')
    ax_surf.set_xlabel('x (cm)', fontsize=11)
    ax_surf.set_ylabel('y (cm)', fontsize=11)
    ax_surf.tick_params(labelsize=9)
    if fiber_offsets:
        fx = [pt[0] for pt in fiber_offsets]
        fy = [pt[1] for pt in fiber_offsets]
        ax_surf.scatter(fx, fy, s=64, facecolors='white', edgecolors='black', lw=0.8, zorder=4)

    # colorbar for the top (shared visually) — place to the right of row 1 spanning rows
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.36])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.set_label('W·cm\u207B\u00B3', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Row 2: center x-z (left) and x-z averaged over y (right)
    ax_xz_center = fig.add_subplot(gs[1, 0])
    im1 = ax_xz_center.imshow(xz_center, origin='lower',
                              extent=[x[0], x[-1], z[0], z[-1]],
                              aspect='auto', vmin=vmin, vmax=vmax/7, cmap=cmap, interpolation='bilinear')
    ax_xz_center.set_title('Center x–z slice', fontsize=12, fontweight='semibold')
    ax_xz_center.set_xlabel('x (cm)'); ax_xz_center.set_ylabel('z (cm)')
    ax_xz_center.tick_params(labelsize=9)

    ax_xz_avg = fig.add_subplot(gs[1, 1])
    im2 = ax_xz_avg.imshow(xz_avg, origin='lower',
                           extent=[x[0], x[-1], z[0], z[-1]],
                           aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear')
    ax_xz_avg.set_title('x–z averaged over y', fontsize=12, fontweight='semibold')
    ax_xz_avg.set_xlabel('x (cm)'); ax_xz_avg.set_ylabel('z (cm)')
    ax_xz_avg.tick_params(labelsize=9)

    # Row 3: center y-z (left) and y-z averaged over x (right)
    ax_yz_center = fig.add_subplot(gs[2, 0])
    im3 = ax_yz_center.imshow(yz_center, origin='lower',
                             extent=[y[0], y[-1], z[0], z[-1]],
                             aspect='auto', vmin=vmin, vmax=vmax/7, cmap=cmap, interpolation='bilinear')
    ax_yz_center.set_title('Center y–z slice', fontsize=12, fontweight='semibold')
    ax_yz_center.set_xlabel('y (cm)'); ax_yz_center.set_ylabel('z (cm)')
    ax_yz_center.tick_params(labelsize=9)

    ax_yz_avg = fig.add_subplot(gs[2, 1])
    im4 = ax_yz_avg.imshow(yz_avg, origin='lower',
                          extent=[y[0], y[-1], z[0], z[-1]],
                          aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, interpolation='bilinear')
    ax_yz_avg.set_title('y–z averaged over x', fontsize=12, fontweight='semibold')
    ax_yz_avg.set_xlabel('y (cm)'); ax_yz_avg.set_ylabel('z (cm)')
    ax_yz_avg.tick_params(labelsize=9)

    # Subtle styling
    for ax in [ax_xz_center, ax_xz_avg, ax_yz_center, ax_yz_avg, ax_surf]:
        # thin border, light grid lines optional
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.grid(False)

    fig.suptitle('Fluence in homogeneous medium (Diffusion equation with Robin BC)', fontsize=16, fontweight='bold', y=0.98)

    # Tidy layout & show
    plt.subplots_adjust(left=0.06, right=0.9, top=0.94, bottom=0.06, hspace=0.28, wspace=0.22)
    plt.show()


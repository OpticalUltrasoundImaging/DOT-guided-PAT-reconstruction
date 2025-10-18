import numpy as np
from numba import njit, prange
from typing import Sequence, Tuple

TWO_PI = 2.0 * np.pi
EPS = 1e-12

@njit(parallel=True)
def compute_h_linear_array(t, zs, xs, xe_left, xe_right, c_val):
    """
    Computation of fractional coverage h(t, element, voxel)
    t: (Nt,) time points
    zs, xs: (Nvox,) source coordinates
    xe_left, xe_right: (Ne,) element edges
    c_val: sound speed
    Returns: h (Nt, Ne, Nvox)
    """
    Nt = t.size
    Ne = xe_left.size
    Nvox = zs.size
    h = np.zeros((Nt, Ne, Nvox), dtype=np.float32)
    
    for ivox in prange(Nvox):
        z = zs[ivox]
        x = xs[ivox]
        for ie in range(Ne):
            xl = xe_left[ie]
            xr = xe_right[ie]
            for it in range(Nt):
                r = c_val * t[it]
                dz = z
                if r < dz:
                    h[it, ie, ivox] = 0.0
                    continue
                rho = np.sqrt(r**2 - dz**2)
                overlap = min(x + rho, xr) - max(x - rho, xl)
                if overlap > 0:
                    h[it, ie, ivox] = overlap / (xr - xl)
                else:
                    h[it, ie, ivox] = 0.0
    return h

@njit(parallel=True)
def generate_G(t, z_grid, x_grid, element_centers, c, ele_width, ele_height, Bt, Bt_deriv):
    """
    System matrix generation for a linear array.
    Returns: G (Nt*Ne, Nz*Nx)
    """
    Nt = t.size
    Ne = element_centers.size
    Nz, Nx = z_grid.shape
    Nvox = Nz * Nx

    zs = z_grid.ravel().astype(np.float32)
    xs = x_grid.ravel().astype(np.float32)
    ys = np.full(Nvox, 0.0, dtype=np.float32)
    
    xe_left = (element_centers - ele_width/2).astype(np.float32)
    xe_right = (element_centers + ele_width/2).astype(np.float32)
    ye_top , ye_bottom = ele_height / 2,  -ele_height / 2
    G2D = np.zeros((Nt*Ne, Nvox), dtype=np.float32)

    Bt = Bt.astype(np.float32)
    Bt_deriv = Bt_deriv.astype(np.float32)

    Bt_fft = np.fft.fft(Bt)
    Bt_deriv_fft = np.fft.fft(Bt_deriv)

    for ivox in prange(Nvox):
        z = zs[ivox]
        x = xs[ivox]
        y = ys[ivox]

        rho_arr = np.sqrt(np.maximum((c * t)**2 - z**2, 0.0))

        for ie in range(Ne):
            x_elem = element_centers[ie]
            xl = xe_left[ie]
            xr = xe_right[ie]
            # vectorized overlap computation
            overlap_x = np.clip(np.minimum(xr, x + rho_arr) - np.maximum(xl, x - rho_arr), 0.0, None)
            overlap_y = np.clip(np.minimum(ye_top, y + rho_arr) - np.maximum(ye_bottom, y - rho_arr), 0.0, None)
            h_vec = (overlap_x / ele_width) * (overlap_y / ele_height)
            r = np.sqrt((z) ** 2 + (x - x_elem) ** 2)
            r[r < 1e-4] = 1e-4
            h_vec /= r
            cos_theta = np.clip(z / r, 0.0, 1.0)
            h_vec *= cos_theta
            # FFT-based convolution
            H_fft = np.fft.fft(h_vec)
            SIR = np.fft.ifft(Bt_fft * H_fft).real.astype(np.float32)
            SIR_deriv = np.fft.ifft(Bt_deriv_fft * H_fft).real.astype(np.float32)
            SIR_deriv *= t.astype(np.float32)

            SIR_sum = SIR - SIR_deriv
            maxabs = np.max(np.abs(SIR_sum))
            if maxabs > 0:
                for it in range(Nt):
                    G2D[it + ie*Nt, ivox] = SIR_sum[it] / maxabs
    return G2D

@njit(parallel=True)
def compute_h_numba(t, z_grid, x_grid, element_centers,
                    ele_width, ele_height, c_val, y_plane=0.0) -> np.ndarray:
    """
    Numba-jitted computation of h(t, element, voxel) including:
      - element width and height fractional overlap
      - 1/r distance decay (r = distance from voxel to element center)
      - cosine angular weighting (z / r clipped to [0,1])
    Inputs:
      t: (Nt,) float32 (seconds)
      z_grid, x_grid: (Nz, Nx) float32 grids (same shape)
      element_centers: (Ne,) float32 (x positions, same units as x_grid)
      ele_width, ele_height: floats (same units as grid)
      c_val: float (sound speed, same length units / second)
      y_plane: float (y position of element center plane)
    Returns:
      h: (Nt, Ne, Nvox) float32
    """
    Nt = t.size
    Nz, Nx = z_grid.shape
    Nvox = Nz * Nx
    Ne = element_centers.size

    # ravel grids
    zs = z_grid.ravel()
    xs = x_grid.ravel()
    ys = np.full(Nvox, y_plane, dtype=np.float32)

    # precompute element edges
    xe_left = (element_centers - ele_width / 2.0).astype(np.float32)
    xe_right = (element_centers + ele_width / 2.0).astype(np.float32)
    ye_top = ele_height / 2.0
    ye_bottom = -ele_height / 2.0

    h = np.zeros((Nt, Ne, Nvox), np.float32)

    # Loop in parallel over voxels
    for ivox in prange(Nvox):
        z = zs[ivox]
        x = xs[ivox]
        y = ys[ivox]

        # Precompute r_voxel_elem and cos_theta per element (independent of time)
        # r_elem = sqrt(z^2 + (x - x_elem)^2 + (y - y_plane)^2)
        r_elem = np.empty(Ne, dtype=np.float32)
        cos_theta_elem = np.empty(Ne, dtype=np.float32)
        for ie in range(Ne):
            dx_e = x - element_centers[ie]
            re = np.sqrt(z * z + dx_e * dx_e)
            if re < EPS:
                re = EPS
            r_elem[ie] = re
            ct = z / re
            if ct < 0.0:
                ct = 0.0
            cos_theta_elem[ie] = ct

        # For each element compute h over time
        for ie in range(Ne):
            xl = xe_left[ie]
            xr = xe_right[ie]
            re = r_elem[ie]
            ct = cos_theta_elem[ie]
            inv_r = 1.0 / re

            # Loop over time samples (compiled; this is fast)
            for it in range(Nt):
                # radius in lateral plane at this time: rho = sqrt((c*t)^2 - z^2)
                ctt = c_val * t[it]
                val = ctt * ctt - z * z
                if val <= 0.0:
                    # wavefront hasn't reached plane at this time
                    h[it, ie, ivox] = 0.0
                    continue
                rho = np.sqrt(val)

                # overlap in x
                left , right = x - rho , x + rho
                overlap_x = (xr if xr < right else right) - (xl if xl > left else left)
                if overlap_x <= 0.0:
                    h[it, ie, ivox] = 0.0
                    continue

                # overlap in y (assuming element extends along y symmetrically)
                top = y + rho
                bottom = y - rho
                overlap_y = (ye_top if ye_top < top else top) - (ye_bottom if ye_bottom > bottom else bottom)
                if overlap_y <= 0.0:
                    h[it, ie, ivox] = 0.0
                    continue

                frac = (overlap_x / ele_width) * (overlap_y / ele_height)
                h[it, ie, ivox] = frac * inv_r * ct
    return h.astype(np.float32)

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def generate_G_from_h_fft(h: np.ndarray,
                          Bt: np.ndarray,
                          Bt_deriv: np.ndarray,
                          t: np.ndarray,
                          normalize: bool = False,
                          subsampling: bool = True,
                          ) -> np.ndarray:
    Nt, Ne, Nvox = h.shape
    h_mat = h.reshape(Nt, Ne * Nvox, order='F')  # columns = ie*Nvox + ivox

    # choose FFT length for linear conv: Lfull = Nt + Nt - 1
    Lfull = Nt + Nt - 1
    nfft = _next_pow2(Lfull)

    H_fft = np.fft.rfft(h_mat, n=nfft, axis=0)         # (k, ncols)
    Bt_fft = np.fft.rfft(Bt, n=nfft)             # (k,)
    Bt_deriv_fft = np.fft.rfft(Bt_deriv, n=nfft) # (k,)

    S_full = np.fft.irfft(Bt_fft[:, None] * H_fft, n=nfft, axis=0)[:Lfull, :]
    Sderiv_full = np.fft.irfft(Bt_deriv_fft[:, None] * H_fft, n=nfft, axis=0)[:Lfull, :]
    start = (Lfull - Nt) // 2
    end = start + Nt
    S = S_full[start:end, :]           # (Nt, Ne*Nvox)
    Sderiv = Sderiv_full[start:end, :] # (Nt, Ne*Nvox)

    # multiply derivative term by t (time axis), broadcasting along columns
    Sderiv = Sderiv * t[:, None]
    Ssum = S - Sderiv  # (Nt, Ne*Nvox)
    Ssum = Ssum.astype(np.float32)
    if normalize:
        maxabs = np.max(np.abs(Ssum), axis=0)
        maxabs[maxabs == 0] = 1.0
        Ssum = Ssum / maxabs[None, :]

    # reshape back to (Nt, Ne, Nvox)
    S3 = Ssum.reshape(Nt, Ne, Nvox, order='F')
    L_end_artifact = 32
    S3[-L_end_artifact:, :, :] = 0.0  # zero out end artifacts due to FFT circular conv
    if subsampling:
        S3 = 0.5*S3[::2,:,:] + 0.5*S3[1::2,:,:]
        Nt = Nt//2
    # permute to (Ne, Nt, Nvox) then flatten to (Ne*Nt, Nvox) with row = ie*Nt + it
    S_perm = np.transpose(S3, (1, 0, 2))
    G2D = S_perm.reshape(Ne * Nt, Nvox, order='C')
    return G2D

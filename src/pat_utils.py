#%% IMPORT LIBRARIES
from __future__ import annotations

from typing import Tuple, Union, Optional, Any, Sequence, Mapping
import os
import warnings
import numpy as np
from scipy import io as spio
from scipy.signal import firwin2, convolve, hilbert
from scipy.ndimage import zoom, gaussian_filter1d
from .load_data_utils import LinearSystemParam
from .recon_iq_utils import Apodization, Coherence, PATInverseSolver
from .math_utils import compute_h_numba, generate_G_from_h_fft
from tqdm import tqdm

F_LO_CUTOFF = 0.04
F_HI_CUTOFF = 0.75
N32SEG = 6
N_ELEMENTS = 128
#%% PA BEAMFORMING
def _ch2sc_pa_batch(mat: dict, info: LinearSystemParam, verbose=False) -> np.ndarray:
    '''
    Ouputs Reconstruction matrix: (AlignedSampleNum, N_ele, N_sc)
    '''
    data_raw = mat["AdcData_frame000"]  # (N_ch per acquisition, samples, frames)
    if verbose:
        print(f"Raw data shape: {data_raw.shape}, dtype={data_raw.dtype}")
    #N_ele = int(info.N_ele)
    AlignedSampleNum = int(getattr(mat.get('AlignedSampleNum', None), 'item', mat.get('AlignedSampleNum', None)))
    frames = data_raw.shape[2]
    if frames < N32SEG:
        raise ValueError(f"Not enough frames ({frames}) for N32seg={N32SEG}")
    
    n_groups = frames // N32SEG
    if n_groups < 1:
        raise ValueError("Not enough full groups of frames to average (need at least one group)")
    
    frames_use = n_groups * N32SEG
    if frames_use != frames:
        data_raw = data_raw[:, :, :frames_use]

    data_rg = data_raw.reshape(32, data_raw.shape[1], n_groups, N32SEG)
    tem = data_rg.sum(axis=2)
    # divide by number of groups to get per-cycle-position averages
    data_avg = tem / float(n_groups)
    p = data_avg  # (32, samples, 6)
    A = 0.5 * (p[:, :, 1] + p[:, :, 5])   # shape (N_ch, samples)
    B = p[:, :, 2]
    C = p[:, :, 3]
    D = 0.5 * (p[:, :, 0] + p[:, :, 4])
    data_combined = np.concatenate([A, B, C, D], axis=0)  # (32*4, samples)
    # truncate or pad to AlignedSampleNum
    if data_combined.shape[1] >= AlignedSampleNum:
        data_final = data_combined[:, :AlignedSampleNum]
    else:
        pad_width = AlignedSampleNum - data_combined.shape[1]
        data_final = np.pad(data_combined, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    if verbose:
        print(f"Data after frame averaging and combining: {data_final.shape}, dtype={data_final.dtype}")
    
    # Build reorder index for ALL scanlines
    N_sc = int(info.N_sc)
    N_ch = int(info.N_ch)
    
    # MuxRatio and sc center index mapping (0-based)
    MuxRatio = float(N_ELEMENTS) / float(N_sc)
    sc_idx_arr = np.arange(N_sc, dtype=float)
    sc_idx1 = np.floor(sc_idx_arr * MuxRatio).astype(int)   # shape (Nsc,)
    start_idx = sc_idx1 - (N_ch / 2.0)                      # float array (Nsc,)
    k0 = np.floor(((N_ELEMENTS * 3) + start_idx) % N_ELEMENTS).astype(int)  # (Nsc,)
    reorder_idx = (k0[None, :] + np.arange(N_ch)[:, None]) % N_ELEMENTS  # (N_ele, Nsc)
    gathered = np.take(data_final, reorder_idx, axis=0)  # (N_ele, Nsc, AlignedSampleNum)
    Reconstruction = np.moveaxis(gathered, -1, 0)  # (AlignedSampleNum, N_ele, Nsc)

    trim_matrix = start_idx[None, :] + np.arange(N_ch)[:, None]  # (N_ch, Nsc)
    rows_to_mask_count = min(N_ch, Reconstruction.shape[1])  # apply mask to top rows_to_mask_count rows
    mask_neg = trim_matrix < 0.0
    mask_pos = trim_matrix > (N_ELEMENTS - 1)
    for sc in range(N_sc):
        # Determine which rows (within 0..rows_to_mask_count-1) to zero
        neg_rows = np.nonzero(mask_neg[:rows_to_mask_count, sc])[0]
        pos_rows = np.nonzero(mask_pos[:rows_to_mask_count, sc])[0]
        if neg_rows.size:
            Reconstruction[:, neg_rows, sc] = 0.0
        if pos_rows.size:
            Reconstruction[:, pos_rows, sc] = 0.0
    if verbose:
        print(f"Full Reconstruction Matrix shape: {Reconstruction.shape}, dtype={Reconstruction.dtype}")
    return Reconstruction

def _process_scanline(Reconstruction: np.ndarray,
                      RxMux: np.ndarray,
                      sc: int,
                      info: LinearSystemParam,
                      d_sample: np.ndarray,
                      data_total1: int) -> tuple[np.ndarray, list[int]]:
    """
    Process a single scanline from the Reconstruction matrix with TOF adjustment and RX apodization.

    Returns:
        RF_scanline_tofadjusted: np.ndarray of shape (data_total1, N_ch)
        active_ch: list of active channel indices
    """
    _, N_ch, n_scan_rec = Reconstruction.shape
    if sc < 0 or sc >= n_scan_rec:
        raise IndexError(f"scanline index sc={sc} out of range (0..{n_scan_rec-1})")

    # Extract channel data for this scanline
    ChData = Reconstruction[:, :, sc].astype(np.float64, copy=False)  # (aligned_samples, N_ch)
    
    # Get the RX Mux table for this scanline
    MuxTbl = RxMux[sc, :].astype(int) if RxMux.ndim == 2 else RxMux[sc].astype(int)  # (N_ch,)

    # Active channels: mapping to valid element indices
    N_ele = int(getattr(info, "N_ele", np.max(MuxTbl)))
    active_mask = (MuxTbl >= 1) & (MuxTbl <= N_ele)
    active_ch = np.nonzero(active_mask)[0]  # 0-based indices
    if active_ch.size == 0:
        return np.zeros((data_total1, N_ch), dtype=np.float64), []

    # Select only active channels
    ChData_active = ChData[:, active_ch]  # (aligned_samples, N_active)
    Mux_active = MuxTbl[active_ch]        # 1-based element numbers

    # ---------------------
    # FIR filters for DC and high-frequency noise cancellation
    f_lo_nodes = [0.0, F_LO_CUTOFF, F_LO_CUTOFF, 1.0]
    m_lo_nodes = [0.0, 0.0, 1.0, 1.0]
    DC_cancel = firwin2(65, f_lo_nodes, m_lo_nodes)

    f_hi_nodes = [0.0, F_HI_CUTOFF, F_HI_CUTOFF, 1.0]
    m_hi_nodes = [1.0, 1.0, 0.0, 0.0]
    HFN_cancel = firwin2(33, f_hi_nodes, m_hi_nodes)

    # Apply filters sequentially
    fil_tmp = convolve(ChData_active, DC_cancel[:, None], mode='same', method='auto')
    fil_tmp = convolve(fil_tmp, HFN_cancel[:, None], mode='same', method='auto')
    # TOF mapping
    sc_pos = float(info.ScanPosition[sc]) if np.ndim(info.ScanPosition) else float(info.ScanPosition)
    ElePos = np.array([info.ElePosition[i - 1] for i in Mux_active], dtype=float)  # (N_active,)
    x = np.abs(sc_pos - ElePos)  # lateral distances (N_active,)
    # Compute TOF and sample indices
    TOF = np.sqrt(d_sample[:, None]**2 + x[None, :]**2) / info.c  # (data_total1, N_active)
    raw_idx = np.round(TOF * info.fs).astype(np.int64) - 1  # MATLAB 1-based to 0-based
    # Clip to valid range
    data_total_local = fil_tmp.shape[0]
    idx_clipped = np.clip(raw_idx, 0, data_total_local - 1).astype(np.intp)
    below_mask = raw_idx < 0
    above_mask = raw_idx >= data_total_local

    # Gather TOF-adjusted samples
    rf_tmp = np.take_along_axis(fil_tmp, idx_clipped, axis=0)
    rf_tmp[below_mask] = 0.0
    rf_tmp[above_mask] = 0.0
    # RX apodization
    h_aper_size = (d_sample[:, None] / info.RxFnum) * 0.5
    half_rx_ch = float(info.half_rx_ch)
    x_b = x[None, :]
    h_safe = np.where(h_aper_size > 0.0, h_aper_size, np.inf)  # (data_total1, 1)
    mask_large = (h_aper_size >= half_rx_ch)
    Rx_apod_idx = np.where(mask_large, x_b / half_rx_ch, x_b / h_safe)
    Rx_apo_r = (Rx_apod_idx < 1.0).astype(np.float64)
    Rx_apo_r = np.nan_to_num(Rx_apo_r, nan=0.0, posinf=0.0, neginf=0.0)
    # Place results into full-width RF matrix
    RF_scanline_tofadjusted = np.zeros((data_total1, N_ch), dtype=np.float64)
    RF_scanline_tofadjusted[:, active_ch] = rf_tmp * Rx_apo_r

    return RF_scanline_tofadjusted, active_ch.tolist()

def pa_das_linear(input_dir: str, 
                  info: LinearSystemParam, 
                  apod_method: str, 
                  coherence_method: str, 
                  channel_file: str = '1_layer0_idx1_CUSTOMDATA_RF.mat', 
                  ) -> Tuple[np.ndarray, np.ndarray]:

    # Build path to channel file
    channel_path = os.path.join(input_dir, channel_file)
    if not os.path.isfile(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    mat = spio.loadmat(channel_path, squeeze_me=True, struct_as_record=False)
    data_total1 = int(info.Nfocus)

    N_ch = int(info.N_ch)
    N_sc = int(info.N_sc)

    # Precompute Rx mux lookup table
    rx_half_ch = N_ch * 0.5
    rx_ch_mtx = np.arange(-int(rx_half_ch), int(rx_half_ch))
    SCvsEle = info.dx / info.pitch

    # RxMux: shape (Nsc, N_ch)
    sc_indices = np.arange(N_sc)
    base_idx = np.floor(sc_indices * SCvsEle).astype(int)
    RxMux = (base_idx[:, None] + 1) + rx_ch_mtx[None, :]

    # Reconstruction shape (AlignedSampleNum, N_ch, Nsc)
    Reconstruction = _ch2sc_pa_batch(mat, info, verbose=False)
    # Prepare outputs
    RF_Sum = np.zeros((data_total1, N_sc), dtype=np.float64)
    d_sample = np.asarray(info.d_sample)
    for sc in tqdm(range(N_sc), desc='BEAMFORMING',leave=False):
        RF_scanline_tofadjusted , active_ch = _process_scanline(Reconstruction, RxMux, sc, info, d_sample, data_total1)
        if len(active_ch) == 0:
            continue

        # Apodization weighting across active channels
        active_ch = np.array(active_ch, dtype=int)
        apodizer = Apodization(RF_scanline_tofadjusted , active_ch, info)
        sum_tmp = apodizer.apply(apod_method)

        coherencer = Coherence(RF_scanline_tofadjusted , active_ch, info)
        cf = coherencer.apply(coherence_method)
        RF_Sum[:, sc] = sum_tmp * cf

    RF_Sum = RF_Sum[:data_total1, :]
    RF_env = np.abs(hilbert(RF_Sum, axis=0))
    return RF_Sum, RF_env

def apply_tgc(RF_env_raw: np.ndarray,
              info: LinearSystemParam,
              alpha: float = 0.8,
              gain_max: float = 1.0,
              use_dB: bool = False,
              smooth_sigma: float = 0.0) -> np.ndarray:
    N_sample = RF_env_raw.shape[0]
    t = np.arange(N_sample) / info.fs
    z = info.c * t / 2.0

    # Compute TGC curve
    if use_dB:
        tgc_curve = 10 ** (alpha * z / 20.0)
    else:
        tgc_curve = np.exp(alpha * z)

    # Normalize to gain_max
    tgc_curve /= np.max(tgc_curve)
    tgc_curve *= gain_max

    if smooth_sigma > 0:
        tgc_curve = gaussian_filter1d(tgc_curve, sigma=smooth_sigma)

    # Apply TGC along depth axis (broadcast over scanlines)
    RF_env_tgc = RF_env_raw * tgc_curve[:, None]
    return RF_env_tgc

def _calculate_time_axis(bbox_cm: Tuple[float, float, float, float],
                         element_centers: np.ndarray,
                         info: LinearSystemParam,
                         buffer_cm: float = 0.1) -> np.ndarray:
    """
    Calculate time axis t for PAT system matrix based on maximum distance.

    Parameters:
    - bbox_cm: (z_min, z_max, x_min, x_max) in cm
    - element_centers: (Ne,) array of detector x positions in cm
    - info: object with attributes c (speed of sound) and fs (sampling frequency)
    - buffer_cm: extra distance buffer in cm

    Returns:
    - t: (Nt,) time array in seconds
    """
    z_min_cm, z_max_cm, x_min_cm, x_max_cm = bbox_cm
    x_min, x_max = x_min_cm / 100.0, x_max_cm / 100.0
    z_min, z_max = z_min_cm / 100.0, z_max_cm / 100.0 # in meters

    corners = np.array([[z_min, x_min],
                        [z_min, x_max],
                        [z_max, x_min],
                        [z_max, x_max]], dtype=np.float32)

    max_dist = 0.0
    for corner in corners:
        z_vox, x_vox = corner
        for x_elem in element_centers:
            dist = np.sqrt((z_vox)**2 + (x_vox - x_elem)**2)
            if dist > max_dist:
                max_dist = dist
    max_dist += buffer_cm / 100.0
    t_max = max_dist / info.c  # in seconds
    Nt = int(np.ceil(t_max * info.fs)) + 1
    t = np.arange(Nt) / info.fs
    return t

def generate_imaging_matrix(bbox_cm: Tuple[float, float, float, float] , scaling_factor: Tuple[float, float], info: LinearSystemParam, 
                            verbose=False):
    x_min_cm , x_max_cm , z_min_cm , z_max_cm = bbox_cm
    scaling_factor_x , scaling_factor_z = scaling_factor
    x_min, x_max = x_min_cm / 100.0, x_max_cm / 100.0
    z_min, z_max = z_min_cm / 100.0, z_max_cm / 100.0 # in meters

    c = info.c
    fc = info.fc

    dx = c / fc / 2.0 * scaling_factor_x
    dz = c / fc / 2.0 * scaling_factor_z
    x_el = info.ElePosition # in meters

    x_grid = np.arange(x_min, x_max + dx, dx)
    z_grid = np.arange(z_min, z_max + dz, dz)
    if verbose:
        print(f"Imaging grid: x from {x_min*100:.1f} cm to {x_max*100:.1f} cm with dx={dx*100:.2f} cm ({x_grid.size} points)")
        print(f"              z from {z_min*100:.1f} cm to {z_max*100:.1f} cm with dz={dz*100:.2f} cm ({z_grid.size} points)")
    Z, X = np.meshgrid(z_grid, x_grid, indexing='ij')
    t_axis = _calculate_time_axis(bbox_cm, x_el, info, buffer_cm=0.1)
    nt = t_axis.size
    if verbose:
        print(f"Time axis: {nt} points, from {t_axis[0]*1e6:.2f} us to {t_axis[-1]*1e6:.2f} us")

    freq = np.linspace(-info.fs/2, info.fs/2, nt)
    freq_diff = 1j*freq
    freq_abs = np.abs(freq)
    Bf = 0.42 - 0.5*np.cos(np.pi*freq_abs/info.fc_signal) + 0.08*np.cos(2*np.pi*freq_abs/info.fc_signal)
    Bf /= np.sum(Bf)
    Bf_deriv = Bf * freq_diff
    Bt = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifftshift(Bf)))) / 1e3
    Bt_deriv = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifftshift(Bf_deriv)))) / 1e3
    win = np.hanning(len(Bt)*2)[:len(Bt)]
    Bt *= win
    Bt_deriv *= win
    h = compute_h_numba(t_axis.astype(np.float32),
                        Z.astype(np.float32), 
                        X.astype(np.float32),
                        x_el.astype(np.float32),
                        float(info.ele_width),
                        float(info.ele_height),
                        float(info.c), 
                        float(0.0),
                        )

    G = generate_G_from_h_fft(h, Bt, Bt_deriv, t_axis, normalize=False)
    meta = {
        't': t_axis,
        'Z': Z, 
        'X': X
    }

    if verbose:
        print(f"System matrix generated. G has shape {G.shape} (dtype={G.dtype})")
    return G, meta

def _load_pa_raw(input_dir: str, 
                channel_file: str = '1_layer0_idx1_CUSTOMDATA_RF.mat',
                verbose: bool = False,
                ) -> np.ndarray:
    channel_path = os.path.join(input_dir, channel_file)
    if not os.path.isfile(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    mat = spio.loadmat(channel_path, squeeze_me=True, struct_as_record=False)
    data_raw = mat["AdcData_frame000"]  # (N_ch per acquisition, samples, frames)
    if verbose:
        print(f"Raw data shape: {data_raw.shape}, dtype={data_raw.dtype}")
    AlignedSampleNum = int(getattr(mat.get('AlignedSampleNum', None), 'item', mat.get('AlignedSampleNum', None)))
    frames = data_raw.shape[2]
    if frames < N32SEG:
        raise ValueError(f"Not enough frames ({frames}) for N32seg={N32SEG}")
    
    n_groups = frames // N32SEG
    if n_groups < 1:
        raise ValueError("Not enough full groups of frames to average (need at least one group)")
    
    frames_use = n_groups * N32SEG
    if frames_use != frames:
        data_raw = data_raw[:, :, :frames_use]

    data_rg = data_raw.reshape(32, data_raw.shape[1], n_groups, N32SEG)
    tem = data_rg.sum(axis=2)
    # divide by number of groups to get per-cycle-position averages
    data_avg = tem / float(n_groups)
    p = data_avg  # (32, samples, 6)
    A = 0.5 * (p[:, :, 1] + p[:, :, 5])   # shape (N_ch, samples)
    B = p[:, :, 2]
    C = p[:, :, 3]
    D = 0.5 * (p[:, :, 0] + p[:, :, 4])
    data_combined = np.concatenate([A, B, C, D], axis=0)  # (32*4, samples)
    # truncate or pad to AlignedSampleNum
    if data_combined.shape[1] >= AlignedSampleNum:
        data_final = data_combined[:, :AlignedSampleNum]
    else:
        pad_width = AlignedSampleNum - data_combined.shape[1]
        data_final = np.pad(data_combined, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    if verbose:
        print(f"Data after frame averaging and combining: {data_final.shape}, dtype={data_final.dtype}")
    return data_final

def pa_inverse_recon(input_dir: str,
                     info: LinearSystemParam,
                     recon_bbox_cm: Tuple[float, float, float, float],
                     scaling_factor: Tuple[float, float],
                     solver_method: str = 'lsqr',
                     normalize_G: bool = True,
                     channel_file: str = '1_layer0_idx1_CUSTOMDATA_RF.mat',
                     verbose: bool = False) -> np.ndarray:
    G, G_meta = generate_imaging_matrix(recon_bbox_cm, scaling_factor, info, verbose=True)
    if verbose:
        print("System matrix generated.")
        print(f"Imaging grid shape: Z {G_meta['Z'].shape[0]}, X {G_meta['Z'].shape[1]}")
    if normalize_G:
        G_norms = np.linalg.norm(G, axis=0)
        G_norms[G_norms == 0] = 1.0
        G /= G_norms
        if verbose:
            print("System matrix normalized.")
    t_axis_recon = G_meta['t']
    t0 = int(t_axis_recon[0]*info.fs)
    Nt = t_axis_recon.size

    pa_raw = _load_pa_raw(input_dir, channel_file, verbose=False) # (N_ELEMENTS, AlignedSampleNum)
    f_lo_nodes = [0.0, F_LO_CUTOFF, F_LO_CUTOFF, 1.0]
    m_lo_nodes = [0.0, 0.0, 1.0, 1.0]
    DC_cancel = firwin2(65, f_lo_nodes, m_lo_nodes)
    f_hi_nodes = [0.0, F_HI_CUTOFF, F_HI_CUTOFF, 1.0]
    m_hi_nodes = [1.0, 1.0, 0.0, 0.0]
    HFN_cancel = firwin2(33, f_hi_nodes, m_hi_nodes)
    pa_raw = convolve(pa_raw, DC_cancel[:, None], mode='same', method='auto')
    pa_raw = convolve(pa_raw, HFN_cancel[:, None], mode='same', method='auto')
    pa_raw = pa_raw[:, t0:t0+Nt]
    pa_raw_flatten = pa_raw.flatten(order='C')  # (N_ELEMENTS*Nt,)
    if verbose:
        print("Raw PA data loaded and preprocessed.")

    solver = PATInverseSolver(G, pa_raw_flatten)
    if verbose:
        print("Starting reconstruction...")
    x_recon = solver.reconstruct(method=solver_method)
    if verbose:
        print("Reconstruction completed.")
    
    if normalize_G:
        x_recon *= G_norms
    x_recon = x_recon.reshape(G_meta['Z'].shape, order='C')
    x_recon = abs(hilbert(x_recon, axis=0))
    return x_recon, G_meta





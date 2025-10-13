#%% IMPORT LIBRARIES
from __future__ import annotations
from typing import Tuple, Union, Mapping
import os
import numpy as np
from scipy import io as spio
from scipy.signal import firwin2, convolve, hilbert
from scipy.ndimage import zoom, uniform_filter, gaussian_filter, median_filter
import cv2
from .load_data_utils import LinearSystemParam
from .recon_iq_utils import Apodization, Coherence
from tqdm import tqdm

#%% US BEAMFORMING
def _ch2sc_us_batch(mat: dict, info: LinearSystemParam, aligned_samples: int) -> np.ndarray:
    '''
    Ouputs Reconstruction matrix: (aligned_samples, N_ch, N_sc)
    '''
    N_sc = info.N_sc
    N_ch = info.N_ch
    N_ele = info.N_ele
    Reconstruction = np.zeros((aligned_samples, N_ch, N_sc), dtype=np.float64)
    data_all = np.stack([
        mat[f'AdcData_scanline{sc:03d}_thi0_sa0'] for sc in range(N_sc)
    ], axis=-1).astype(np.float64, copy=False)  # shape: (samples, channels, 2, scanlines)

    page1 = data_all[:, :, 0, :]  # (samples, channels, scanlines)
    page2 = data_all[:, :, 1, :]
    data64ch_all = np.concatenate((page1, page2), axis=0)
    total_rows = data64ch_all.shape[0]

    # Compute MuxRatio mapping for all scanlines
    MuxRatio = N_ele / N_sc
    sc_indices = np.floor(np.arange(N_sc) * MuxRatio).astype(int)  # array of sc_idx1
    start_idx = sc_indices - N_ch / 2.0

    k_float = np.mod(N_ch * 3 + start_idx, N_ch) + 1
    k = np.clip(np.floor(k_float).astype(int), 1, N_ch) - 1
    reordered_indices = [np.r_[k_i:total_rows, 0:k_i] for k_i in k]  # list of length N_sc, each (total_rows,)
    reordered_indices = np.stack(reordered_indices, axis=1)  # (total_rows, N_sc)

    scan_idx = np.arange(N_sc)[None, :]       # (1, N_sc)
    reordered = data64ch_all[reordered_indices, :, scan_idx]  # (total_rows, channels, N_sc)
    trim_idx = start_idx[None, :] + np.arange(N_ch)[:, None]  # (N_ch, N_sc)
    mask = (trim_idx < 0) | (trim_idx > (N_ele - 1))
    rows_to_mask = min(N_ch, reordered.shape[0])
    reordered[:rows_to_mask, :, :][mask] = 0.0

    Reconstruction = reordered.T  # (N_ch, total_rows, N_sc)
    Reconstruction = Reconstruction[:, :aligned_samples, :].transpose(0,2,1)  # (aligned_samples, N_ch, N_sc)
    
    return Reconstruction

def _process_scanline(Reconstruction: np.ndarray,
                      RxMux: np.ndarray,
                      sc: int,
                      info: LinearSystemParam,
                      DC_cancel: np.ndarray,
                      d_sample: np.ndarray,
                      data_total: int,
                      data_total1: int) -> Tuple[np.ndarray, list[int]]:
    
    _, N_ch, n_scan_rec = Reconstruction.shape
    if sc < 0 or sc >= n_scan_rec:
        raise IndexError(f"scanline index sc={sc} is out of range (0..{n_scan_rec-1}) for Reconstruction")

    ChData = Reconstruction[:, :, sc].astype(np.float64, copy=False)  # (aligned_samples, N_ch)
    MuxTbl = RxMux[sc, :].astype(int) if RxMux.ndim == 2 else RxMux[sc].astype(int)  # (N_ch,)

    # Active channels are those mapping to valid element indices (1-based in RxMux)
    active_mask = (MuxTbl >= 1) & (MuxTbl <= int(getattr(info, "N_ele", np.max(MuxTbl))))
    active_ch = np.nonzero(active_mask)[0]  # indices into channels (0-based)
    if active_ch.size == 0:
        return np.zeros((data_total1, N_ch), dtype=np.float64), []

    # Select only active channels
    ChData_active = ChData[:, active_ch]     # (aligned_samples, N_active)
    Mux_active = MuxTbl[active_ch]           # element numbers (1-based)
    fil_tmp = convolve(ChData_active, DC_cancel[:, None], mode='same', method='auto')

    # ---------------------
    # Compute per-channel geometry and TOF mapping
    sc_pos = float(info.ScanPosition[sc]) if np.ndim(info.ScanPosition) else float(info.ScanPosition)
    ElePos = np.array([info.ElePosition[i - 1] for i in Mux_active], dtype=float)  # (N_active,)
    x = np.abs(sc_pos - ElePos)  # (N_active,)

    # tx_t (data_total1,) and rx_t (data_total1, N_active)
    tx_t = d_sample / info.c  # (data_total1,)
    rx_t = np.sqrt(d_sample[:, None] ** 2 + x[None, :] ** 2) / info.c  # (data_total1, N_active)
    TOF = tx_t[:, None] + rx_t  # (data_total1, N_active)

    # raw indices using MATLAB rounding convention -> convert to 0-based
    raw_idx = np.round(TOF * info.fs).astype(np.int64) - 1  # may contain OOB values

    # Use actual fil_tmp depth for clipping, NOT the passed-in data_total
    data_total_local = fil_tmp.shape[0]
    below_mask = raw_idx < 0
    above_mask = raw_idx >= data_total_local

    # Clip indices to valid range for safe indexing
    idx_clipped = np.clip(raw_idx, 0, data_total_local - 1).astype(np.intp)  # shape (data_total1, N_active)

    # Gather samples per (depth, channel) using take_along_axis
    # fil_tmp: (data_total_local, N_active), idx_clipped: (data_total1, N_active)
    rf_tmp = np.take_along_axis(fil_tmp, idx_clipped, axis=0)  # (data_total1, N_active)

    # Zero out-of-range samples according to original raw_idx masks
    if np.any(below_mask):
        rf_tmp[below_mask] = 0.0
    if np.any(above_mask):
        rf_tmp[above_mask] = 0.0

    # h_aper_size = (d_sample[:, None] / info.RxFnum) * 0.5               # (data_total1, 1)
    # half_rx_ch = float(info.half_rx_ch)

    # mask_large = (h_aper_size >= half_rx_ch)                           # bool (data_total1, 1)
    # x_b = x[None, :]
    # h_safe = np.where(h_aper_size > 0.0, h_aper_size, np.inf)
    # Rx_apod_idx = np.empty((h_aper_size.shape[0], x_b.shape[1]), dtype=np.float64)
    # Rx_apod_idx[mask_large[:, 0], :] = x_b[mask_large[:, 0], :] / half_rx_ch
    # Rx_apod_idx[~mask_large[:, 0], :] = x_b[~mask_large[:, 0], :] / h_safe[~mask_large[:, 0], :]

    # Rx_apo_r = (Rx_apod_idx < 1.0).astype(np.float64)
    # Rx_apo_r = np.nan_to_num(Rx_apo_r, copy=False)

    h_aper_size = (d_sample[:, None] / info.RxFnum) * 0.5
    half_rx_ch = float(info.half_rx_ch)
    x_b = x[None, :]
    h_safe = np.where(h_aper_size > 0.0, h_aper_size, np.inf)  # (data_total1, 1)
    mask_large = (h_aper_size >= half_rx_ch)
    Rx_apod_idx = np.where(mask_large, x_b / half_rx_ch, x_b / h_safe)
    Rx_apo_r = (Rx_apod_idx < 1.0).astype(np.float64)
    Rx_apo_r = np.nan_to_num(Rx_apo_r, nan=0.0, posinf=0.0, neginf=0.0)

    # Place results into full-width RF matrix (data_total1, N_ch)
    RF_scanline_tofadjusted = np.zeros((data_total1, N_ch), dtype=np.float64)
    RF_scanline_tofadjusted[:, active_ch] = rf_tmp * Rx_apo_r

    return RF_scanline_tofadjusted, active_ch.tolist()

def pe_das_linear(input_dir: str, 
                  info: LinearSystemParam, 
                  dB_US: float, 
                  apod_method: str, 
                  coherence_method: str, 
                  channel_file: str = '1_layer0_idx1_BDATA_RF.mat', 
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Build path to channel file
    channel_path = os.path.join(input_dir, channel_file)
    if not os.path.isfile(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    mat = spio.loadmat(channel_path, squeeze_me=True, struct_as_record=False)
    aligned_samples = int(getattr(mat.get('AlignedSampleNum', None), 'item', mat.get('AlignedSampleNum', None)))
    data_total = int(np.floor(aligned_samples))
    data_total1 = int(info.Nfocus)

    N_ch = int(info.N_ch)
    N_sc = int(info.N_sc)

    # Precompute Rx mux lookup table
    rx_half_ch = N_ch * 0.5
    rx_ch_mtx = np.arange(-int(rx_half_ch), int(rx_half_ch))
    SCvsEle = info.dx / info.pitch

    # RxMux: shape (Nsc, N_ch)
    sc_indices = np.arange(N_sc)
    base_idx = np.floor(sc_indices * SCvsEle).astype(int)  # zero-based offset
    RxMux = (base_idx[:, None] + 1) + rx_ch_mtx[None, :]  # 1-based like MATLAB

    # Reconstruction shape (AlignedSampleNum, N_ch, Nsc)
    Reconstruction = _ch2sc_us_batch(mat, info, aligned_samples)

    # Precompute DC cancellation FIR filter (fir2 equivalent)
    # MATLAB: f = [0 0.1 0.1 1]; m = [0 0 1 1]; DC_cancel = fir2(64,f,m)
    f_nodes = np.array([0.0, 0.1, 0.1, 1.0])
    m_nodes = np.array([0.0, 0.0, 1.0, 1.0])
    DC_cancel = firwin2(65, f_nodes, m_nodes)

    # Prepare outputs
    RF_Sum = np.zeros((data_total1, N_sc), dtype=np.float64)
    d_sample = np.asarray(info.d_sample)
    for sc in tqdm(range(N_sc), desc='US BEAMFORMING',leave=False):
        RF_scanline_tofadjusted , active_ch = _process_scanline(Reconstruction, RxMux, sc, info, DC_cancel, d_sample, data_total, data_total1)
        if len(active_ch) == 0:
            continue

        # Apply time-gain compensation
        tgc_us = 1.0 + 5.0e-3 * (np.arange(1, RF_scanline_tofadjusted.shape[0] + 1))
        RF_scanline_tofadjusted = RF_scanline_tofadjusted * tgc_us[:, None]
        
        # Apodization weighting across active channels
        active_ch = np.array(active_ch, dtype=int)
        apodizer = Apodization(RF_scanline_tofadjusted , active_ch, info)
        sum_tmp = apodizer.apply(apod_method)

        coherencer = Coherence(RF_scanline_tofadjusted , active_ch, info)
        cf = coherencer.apply(coherence_method)
        RF_Sum[:, sc] = sum_tmp * cf

    # trim and dynamic range processing
    RF_Sum = RF_Sum[:data_total1, :]

    # Dynamic Ranging
    min_dB = 10 ** (-dB_US / 20.0)
    RF_env = np.abs(hilbert(RF_Sum, axis=0))
    RF_env_raw = RF_env.copy()
    RF_env_norm = RF_env / np.max(RF_env) if np.max(RF_env) != 0 else RF_env
    # RF_log = (20/dB_US)*log10(RF_env_norm)+1
    with np.errstate(divide='ignore', invalid='ignore'):
        RF_log = (20.0 / dB_US) * np.log10(np.maximum(RF_env_norm, 1e-20)) + 1.0
    RF_log[RF_env_norm < min_dB] = 0.0
    
    # Compute target image size
    xz_ratio = info.FOV / np.max(d_sample)
    Lx = 4 * N_sc
    Lz = int(round(Lx / xz_ratio)) if xz_ratio != 0 else RF_log.shape[0]
    # Use scipy.ndimage.zoom to resize to (Lz, Lx)
    zoom_z = Lz / RF_log.shape[0] if RF_log.shape[0] > 0 else 1.0
    zoom_x = Lx / RF_log.shape[1] if RF_log.shape[1] > 0 else 1.0
    RF_log_resized = zoom(RF_log, (zoom_z, zoom_x), order=1)  # bilinear interpolation

    return RF_Sum, RF_env_raw, RF_log_resized

def _smooth_nakagami(
    naka_img: np.ndarray,
    us_img: np.ndarray = None,
    guided_radius: int = 8,
    guided_eps: float = 0.01,
):
    # normalize inputs to float32 and scale to 0..1 to keep parameters stable
    def _norm01(img):
        img = img.astype(np.float32, copy=False)
        mn, mx = np.nanmin(img), np.nanmax(img)
        if mx <= mn:
            return np.zeros_like(img)
        return (img - mn) / (mx - mn)

    naka = _norm01(naka_img)
    guide = _norm01(us_img) if us_img is not None else naka
    guided = cv2.ximgproc.guidedFilter(guide.astype(np.float32), naka.astype(np.float32), radius=guided_radius, eps=float(guided_eps))
    guided = gaussian_filter(guided, sigma=2.0)
    return guided.astype(np.float32)

def _clip01(img):
    img = np.asarray(img, dtype=np.float32)
    img = img - np.nanmin(img)
    mx = np.nanmax(img)
    if mx > 0:
        img = img / mx
    else:
        img = np.zeros_like(img)
    return img

def _despeckle_lee(us_img, win_size=7, var_noise=None):
    I = _clip01(us_img).astype(np.float32)
    if win_size % 2 == 0:
        win_size += 1

    # local mean and variance via box filter
    kernel = (win_size, win_size)
    mean = cv2.boxFilter(I, ddepth=-1, ksize=kernel, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(I * I, ddepth=-1, ksize=kernel, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_local = mean_sq - mean * mean
    var_local = np.maximum(var_local, 0.0)

    if var_noise is None:
        # estimate noise variance as the median of local variances (robust)
        var_noise = np.median(var_local[var_local > 0]) if np.any(var_local > 0) else 1e-6

    # compute Lee weight
    W = var_local / (var_local + var_noise)
    out = mean + W * (I - mean)
    return _clip01(out).astype(np.float32)

def nakagami_linear(R_env_raw: np.ndarray,
                    info: LinearSystemParam,
                    N_repeat: int = 7)-> np.ndarray:
    '''
    Nakagami parameter estimation from RF envelope data.
    R_env_raw: (N_focus, N_sc) beamformed RF envelope data
    N_repeat: number of different window sizes to use for local statistics
    Returns: (N_focus, N_sc) Nakagami parameter map
    '''
    N_focus, N_sc = R_env_raw.shape
    pulse_length = info.c / info.fc
    pitch = info.pitch
    pixel_d = info.pixel_d
    naka_stack = np.full((N_focus, N_sc, N_repeat), np.nan, dtype=np.float32)
    for i_repeat in tqdm(range(N_repeat), desc = "Nakagami Estimation", leave=False):
        w = max(1 , ((i_repeat+1)//2)) * 0.5e-3
        h = (i_repeat + 2) * pulse_length
        
        w_roi = int(round(w / pitch))
        w_roi = max(1, w_roi)  # at least 1 sample width
        if (w_roi % 2) == 0: w_roi += 1

        h_roi = int(round(h / pixel_d))
        h_roi = max(1, h_roi)
        if (h_roi % 2) == 0: h_roi += 1
        size_tuple = (h_roi, w_roi)

        # compute local second and fourth raw moment (i.e. mean of r^2 and r^4)
        r2_local_mean = uniform_filter(R_env_raw ** 2, size=size_tuple, mode="reflect")
        r4_local_mean = uniform_filter(R_env_raw ** 4, size=size_tuple, mode="reflect")
        m2 = r2_local_mean
        m4 = r4_local_mean
        denom = (m4 - m2 * m2)

        invalid_mask = denom <= 0
        k_map = np.empty_like(denom, dtype=np.float32)
        k_map[invalid_mask] = np.nan
        safe_denom = denom.copy()
        safe_denom[invalid_mask] = np.nan  # will produce nan in division
        k_map[~invalid_mask] = (m2[~invalid_mask] * m2[~invalid_mask]) / safe_denom[~invalid_mask]
        naka_stack[:, :, i_repeat] = k_map
    
    naka_avg = np.nanmedian(naka_stack, axis=2)
    tgc_nakagami = 1.0 - 2.0e-4 * (np.arange(1, naka_avg.shape[0] + 1))
    naka_avg = naka_avg * tgc_nakagami[:, None]

    RF_env_norm = R_env_raw / np.max(R_env_raw) if np.max(R_env_raw) != 0 else R_env_raw
    dB_US = 75
    min_dB = 10 ** (-dB_US / 20.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        RF_log = (20.0 / dB_US) * np.log10(np.maximum(RF_env_norm, 1e-20)) + 1.0
    RF_log[RF_env_norm < min_dB] = 0.0
    RF_log = _despeckle_lee(RF_log, win_size=9)
    naka_avg = _smooth_nakagami(naka_avg, RF_log, guided_radius=8, guided_eps=0.01)

    xz_ratio = info.FOV / np.max(info.d_sample)
    Lx = 4 * N_sc
    Lz = int(round(Lx / xz_ratio)) if xz_ratio != 0 else naka_avg.shape[0]
    # Use scipy.ndimage.zoom to resize to (Lz, Lx)
    zoom_z = Lz / naka_avg.shape[0] if naka_avg.shape[0] > 0 else 1.0
    zoom_x = Lx / naka_avg.shape[1] if naka_avg.shape[1] > 0 else 1.0
    naka_resized = zoom(naka_avg, (zoom_z, zoom_x), order=1)
    return naka_resized, naka_resized

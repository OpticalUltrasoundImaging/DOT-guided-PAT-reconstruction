#%% IMPORT LIBRARIES
from __future__ import annotations
from typing import Tuple
import os
import numpy as np
from scipy import io as spio
from scipy.signal import firwin2, convolve, hilbert
from scipy.ndimage import zoom
from load_data_utils import LSystemParam
from recon_iq_utils import Apodization, Coherence
from tqdm import tqdm

#%% US BEAMFORMING
def _ch2sc_us_batch(mat: dict, info: LSystemParam, aligned_samples: int) -> np.ndarray:
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
    Reconstruction = Reconstruction[:, :aligned_samples, :].transpose(1, 0, 2)  # (aligned_samples, N_ch, N_sc)
    return Reconstruction

def _process_scanline(Reconstruction: np.ndarray, 
                      RxMux: np.ndarray, 
                      sc: int,
                      info: LSystemParam,
                      DC_cancel: np.ndarray,
                      d_sample: np.ndarray,
                      data_total: int,
                      data_total1: int):
    '''
    Vectorized processing of one scanline:
        - TOF adjustment
        - DC cancellation
        - Rx apodization
    Returns
    -------
    RF_scanline_tofadjusted : np.ndarray
        (data_total1, N_ch) RF data for scanline
    active_ch : list
        List of active channels
    '''
    _, N_ch = Reconstruction[:, :, sc].shape
    ChData = Reconstruction[:, :, sc].astype(np.float64, copy=False)
    MuxTbl = RxMux[sc, :].astype(int)

    # Identify active channels
    active_mask = (MuxTbl >= 1) & (MuxTbl <= info.N_ele)
    active_ch = np.nonzero(active_mask)[0]
    if active_ch.size == 0:
        return np.zeros((data_total1, N_ch), dtype=np.float64), []
    ChData_active = ChData[:, active_ch]
    Mux_active = MuxTbl[active_ch]

    # DC cancel filter (along each A-line)
    fil_tmp = np.apply_along_axis(lambda x: convolve(x, DC_cancel, mode='same'), 0, ChData_active)

    # Compute per-channel distances
    sc_pos = info.ScanPosition[sc] if np.ndim(info.ScanPosition) else float(info.ScanPosition)
    ElePos = np.array([info.ElePosition[i - 1] for i in Mux_active], dtype=float)
    x = np.abs(sc_pos - ElePos)  # shape (N_active,)

    # TOF calculation
    tx_t = d_sample / info.c  # (data_total1,)
    rx_t = np.sqrt(d_sample[:, None] ** 2 + x[None, :] ** 2) / info.c  # (data_total1, N_active)
    TOF = tx_t[:, None] + rx_t
    RxReadPointer = np.round(TOF * info.fs).astype(int) - 1  # 0-based
    RxReadPointer = np.clip(RxReadPointer, 0, data_total - 1)
    row_idx = RxReadPointer
    col_idx = np.arange(active_ch.size)[None, :]
    rf_tmp = fil_tmp[row_idx, col_idx]  # shape (data_total1, N_active)

    # Zero out-of-bounds samples
    below_mask = RxReadPointer < 0
    above_mask = RxReadPointer >= data_total
    rf_tmp[below_mask] = 0.0
    rf_tmp[above_mask] = 0.0

    # Rx apodization
    h_aper_size = d_sample[:, None] / info.RxFnum * 0.5
    half_rx_ch = info.half_rx_ch
    Rx_apod_idx = np.where(h_aper_size >= half_rx_ch, x / half_rx_ch, x / h_aper_size)
    Rx_apo_r = np.ones_like(Rx_apod_idx)
    Rx_apo_r[Rx_apod_idx >= 1.0] = 0.0
    RF_scanline_tofadjusted = np.zeros((data_total1, N_ch), dtype=np.float64)
    RF_scanline_tofadjusted[:, active_ch] = rf_tmp * Rx_apo_r

    return RF_scanline_tofadjusted, active_ch.tolist()

def pe_das_linear(input_dir: str, 
                  info: LSystemParam, 
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
    for sc in tqdm(range(N_sc), desc='BEAMFORMING',leave=False):
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
#%% IMPORT LIBRARIES
from __future__ import annotations

from typing import Tuple, Union, Optional, Any, Sequence, Mapping
import os
import warnings
import numpy as np
from scipy import io as spio
from scipy.signal import firwin2, convolve, hilbert
from scipy.ndimage import zoom
from load_data_utils import LinearSystemParam
from recon_iq_utils import Apodization, Coherence
from tqdm import tqdm

F_LO_CUTOFF = 0.04
F_HI_CUTOFF = 0.75
N32SEG = 6
#%% PA BEAMFORMING
def _ch2sc_pa_batch(mat: dict, info: LinearSystemParam) -> np.ndarray:
    '''
    Ouputs Reconstruction matrix: (AlignedSampleNum, N_ele, N_sc)
    '''
    data_raw = getattr(mat, "AdcData_frame000")
    N_ele = int(info.N_ele)
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

    data_rg = data_raw.reshape(N_ele, data_raw.shape[1], n_groups, N32SEG)
    tem = data_rg.sum(axis=2)
    # divide by number of groups to get per-cycle-position averages
    data_avg = tem / float(n_groups)
    p = data_avg  # (N_ele, samples, 6)
    page1 = p[:, :, 1]
    page6 = p[:, :, 5]
    page2 = p[:, :, 2]
    page3 = p[:, :, 3]
    page4 = p[:, :, 0]
    page5 = p[:, :, 4]
    A = 0.5 * (page1 + page6)   # shape (N_ele, samples)
    B = page2
    C = page3
    D = 0.5 * (page4 + page5)
    data_combined = np.concatenate([A, B, C, D], axis=1)  # (N_ele, samples*4)
    # truncate or pad to AlignedSampleNum
    if data_combined.shape[1] >= AlignedSampleNum:
        data_final = data_combined[:, :AlignedSampleNum]
    else:
        pad_width = AlignedSampleNum - data_combined.shape[1]
        data_final = np.pad(data_combined, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    # -Build reorder index for ALL scanlines
    N_sc = int(info.N_sc)
    N_ch = int(info.N_ch)

    # MuxRatio and sc center index mapping (0-based)
    MuxRatio = float(info.N_ele) / float(N_sc)
    sc_idx_arr = np.arange(N_sc, dtype=float)
    sc_idx1 = np.floor(sc_idx_arr * MuxRatio).astype(int)   # shape (Nsc,)
    start_idx = sc_idx1 - (N_ch / 2.0)                      # float array (Nsc,)
    k0 = np.floor(((info.N_ele * 3) + start_idx) % info.N_ele).astype(int)  # (Nsc,)
    reorder_idx = (k0[None, :] + np.arange(info.N_ele)[:, None]) % info.N_ele  # (N_ele, Nsc)
    gathered = np.take(data_final, reorder_idx, axis=0)  # (N_ele, Nsc, AlignedSampleNum)
    Reconstruction = np.moveaxis(gathered, -1, 0)  # (AlignedSampleNum, N_ele, Nsc)

    trim_matrix = start_idx[None, :] + np.arange(N_ch)[:, None]  # (N_ch, Nsc)
    rows_to_mask_count = min(N_ch, Reconstruction.shape[1])  # apply mask to top rows_to_mask_count rows
    mask_neg = trim_matrix < 0.0
    mask_pos = trim_matrix > (info.N_ele - 1)
    for sc in range(N_sc):
        # Determine which rows (within 0..rows_to_mask_count-1) to zero
        neg_rows = np.nonzero(mask_neg[:rows_to_mask_count, sc])[0]
        pos_rows = np.nonzero(mask_pos[:rows_to_mask_count, sc])[0]
        if neg_rows.size:
            Reconstruction[:, neg_rows, sc] = 0.0
        if pos_rows.size:
            Reconstruction[:, pos_rows, sc] = 0.0
    return Reconstruction

def _process_scanline(Reconstruction: np.ndarray,
                      RxMux: np.ndarray,
                      sc: int,
                      info: LinearSystemParam,
                      d_sample: np.ndarray,
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
    f_nodes = np.array([0.0, F_LO_CUTOFF, F_LO_CUTOFF, 1.0])
    m_nodes = np.array([0.0, 0.0, 1.0, 1.0])
    DC_cancel = firwin2(65, f_nodes, m_nodes)
    f_nodes = np.array([0.0, F_HI_CUTOFF, F_HI_CUTOFF, 1.0])
    m_nodes = np.array([1.0, 1.0, 0.0, 0.0])
    HFN_cancel = firwin2(33, f_nodes, m_nodes)
    fil_tmp = convolve(ChData_active, DC_cancel[:, None],  mode='same', method='auto')
    fil_tmp = convolve(ChData_active, HFN_cancel[:, None], mode='same', method='auto')

    # ---------------------
    # Compute per-channel geometry and TOF mapping
    sc_pos = float(info.ScanPosition[sc]) if np.ndim(info.ScanPosition) else float(info.ScanPosition)
    ElePos = np.array([info.ElePosition[i - 1] for i in Mux_active], dtype=float)  # (N_active,)
    x = np.abs(sc_pos - ElePos)  # (N_active,)

    # tx_t = 0 and rx_t (data_total1, N_active)
    TOF = np.sqrt(d_sample[:, None] ** 2 + x[None, :] ** 2) / info.c  # (data_total1, N_active)

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

    h_aper_size = (d_sample[:, None] / info.RxFnum) * 0.5               # (data_total1, 1)
    half_rx_ch = float(info.half_rx_ch)

    mask_large = (h_aper_size >= half_rx_ch)                           # bool (data_total1, 1)
    x_b = x[None, :]
    h_safe = np.where(h_aper_size > 0.0, h_aper_size, np.inf)
    Rx_apod_idx = np.empty((h_aper_size.shape[0], x_b.shape[1]), dtype=np.float64)
    Rx_apod_idx[mask_large[:, 0], :] = x_b[mask_large[:, 0], :] / half_rx_ch
    Rx_apod_idx[~mask_large[:, 0], :] = x_b[~mask_large[:, 0], :] / h_safe[~mask_large[:, 0], :]

    Rx_apo_r = (Rx_apod_idx < 1.0).astype(np.float64)
    Rx_apo_r = np.nan_to_num(Rx_apo_r, copy=False)

    # Place results into full-width RF matrix (data_total1, N_ch)
    RF_scanline_tofadjusted = np.zeros((data_total1, N_ch), dtype=np.float64)
    RF_scanline_tofadjusted[:, active_ch] = rf_tmp * Rx_apo_r

    return RF_scanline_tofadjusted, active_ch.tolist()

def pa_das_linear(input_dir: str, 
                  info: LinearSystemParam, 
                  apod_method: str, 
                  coherence_method: str, 
                  channel_file: str = '1_layer0_idx1_CUSTOMDATA_RF.mat', 
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Build path to channel file
    channel_path = os.path.join(input_dir, channel_file)
    if not os.path.isfile(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    mat = spio.loadmat(channel_path, squeeze_me=True, struct_as_record=False)
    aligned_samples = int(getattr(mat.get('AlignedSampleNum', None), 'item', mat.get('AlignedSampleNum', None)))
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
    Reconstruction = _ch2sc_pa_batch(mat, info, aligned_samples)
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

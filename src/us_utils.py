#%% IMPORT LIBRARIES
from typing import Tuple, Union, Optional, Any, Sequence, Mapping
import os
import warnings

import numpy as np
from scipy import io as spio
from scipy.signal import firwin2, convolve, hilbert
from scipy.ndimage import zoom
from types import SimpleNamespace

#%% US BEAMFORMING
def load_sequence_info():
    mat = spio.loadmat(r'artifact/LSequence.mat', squeeze_me=True, struct_as_record=False)
    Roi = mat.get('Roi')
    System = mat.get('System')
    return Roi, System


def _get_nested(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    '''
    '''
    cur = obj
    for p in path:
        if cur is None: return default
        # mapping/dict-like
        if isinstance(cur, Mapping):
            cur = cur.get(p, default)
        else:
            cur = getattr(cur, p, default)
    return cur


@dataclass
class LinearSystemParamUS:

def ch2sc_us(scanline_idx: int,
             elementNum: int,
             scNum: int,
             channelNum: int,
             data: Any) -> np.ndarray:

    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Expected `data` to be 3D like MATLAB's data(:,:,1/2). Got shape {arr.shape}")

    # Build data64ch exactly as MATLAB: vertical concatenation of the two pages
    # MATLAB: data64ch = [data(:,:,1) ; data(:,:,2)];
    page1 = arr[:, :, 0]
    page2 = arr[:, :, 1]
    data64ch = np.vstack((page1, page2))  # shape (total_rows, aligned_samples)
    total_rows, aligned_samples = data64ch.shape

    # Sanity: expected total_rows to equal elementNum
    if total_rows != elementNum:
        # warn but continue — MATLAB assumed elementNum matches this total
        # (we do not raise because some datasets may differ; user should verify)
        import warnings
        warnings.warn(f"elementNum ({elementNum}) != stacked rows ({total_rows}). Continuing with stacked rows={total_rows}.")

    # MuxRatio and index arithmetic (follow MATLAB behavior)
    # sc_idx1 = floor(sc_idx * MuxRatio)
    MuxRatio = float(elementNum) / float(scNum)
    sc_idx1 = int(np.floor(scanline_idx * MuxRatio))

    # start_idx = sc_idx1 - channelNum/2  (can be float if channelNum is odd)
    start_idx = sc_idx1 - (channelNum / 2.0)

    # k = rem(channelNum*3 + start_idx, channelNum) + 1
    # Use numpy.mod to emulate MATLAB rem for positive divisor
    k_float = np.mod(channelNum * 3 + start_idx, channelNum) + 1.0
    # Convert to integer (MATLAB uses 1-based integer index)
    # We use floor to be consistent with MATLAB integer indexing behavior
    k = int(np.floor(k_float))
    # Bound k within 1..channelNum
    if k < 1:
        k = 1
    elif k > channelNum:
        k = channelNum

    # Convert to 0-based start index for Python
    k0 = k - 1

    # Now reorder rows as MATLAB did: [k:elementNum, 1:k-1]
    # but use actual total_rows (stacked pages) instead of hard-coded 64
    # Build the index map:
    idx_part1 = np.arange(k0, total_rows, dtype=int)
    idx_part2 = np.arange(0, k0, dtype=int)
    reorder_rows = np.concatenate((idx_part1, idx_part2), axis=0)

    # Apply reordering to data64ch
    reordered = data64ch[reorder_rows, :]   # shape (total_rows, aligned_samples)

    # Now perform the trimming / zeroing:
    # trim_idx = start_idx : start_idx + channelNum - 1  (MATLAB colon)
    start_trim = start_idx
    # Build float array for comparisons, but also a corresponding integer row index map for row selection.
    trim_idx_floats = start_trim + np.arange(channelNum, dtype=float)

    # Determine which positions correspond to invalid element indices
    mask_neg = trim_idx_floats < 0.0
    mask_pos = trim_idx_floats > (elementNum - 1)

    # In MATLAB the boolean indexing is applied to the rows of reordered_data.
    # Usually channelNum equals number of rows (total_rows). If they differ, we map the masks
    # to the top-left portion (first `channelNum` rows) of `reordered`.
    rows_to_mask_count = min(channelNum, reordered.shape[0])

    if rows_to_mask_count != channelNum:
        # If shapes don't match exactly, we still apply masks to the first `rows_to_mask_count` rows.
        import warnings
        warnings.warn(
            f"channelNum ({channelNum}) != reordered rows ({reordered.shape[0]}). "
            f"Applying trimming mask to the first {rows_to_mask_count} rows."
        )

    # Which row indices (in reordered) correspond to logical mask positions?
    # We choose the first `rows_to_mask_count` rows to align with MATLAB's intent.
    base_row_indices = np.arange(rows_to_mask_count, dtype=int)

    # Apply negative-mask zeros
    neg_rows = base_row_indices[mask_neg[:rows_to_mask_count]]
    if neg_rows.size > 0:
        reordered[neg_rows, :] = 0.0

    # Apply positive-mask zeros
    pos_rows = base_row_indices[mask_pos[:rows_to_mask_count]]
    if pos_rows.size > 0:
        reordered[pos_rows, :] = 0.0

    # MATLAB returns reordered_data with shape (channelNum, AlignedSampleNum)
    # Previous Python caller expects (AlignedSampleNum, channelNum), so we transpose:
    reordered_T = reordered.T  # shape (aligned_samples, total_rows)

    # If user expects exactly channelNum columns, ensure final width is channelNum
    if reordered_T.shape[1] != channelNum:
        # If total_rows != channelNum, trim or pad columns to channelNum to keep compatibility.
        if reordered_T.shape[1] > channelNum:
            reordered_T = reordered_T[:, :channelNum]
        else:
            # pad with zeros
            pad_cols = channelNum - reordered_T.shape[1]
            reordered_T = np.pad(reordered_T, ((0, 0), (0, pad_cols)), mode='constant', constant_values=0.0)

    return reordered_T

def pe_das_linear(input_dir: str, 
                  infoUS: Union[SimpleNamespace, dict, object], 
                  dB_US: float, 
                  apod_method: str, 
                  coherence_method: str, 
                  channel_file: str = '1_layer0_idx1_BDATA_RF.mat', 
                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert infoUS to attribute access if dict
    if isinstance(infoUS, dict):
        info = SimpleNamespace(**infoUS)
    else:
        info = infoUS

    # Build path to channel file
    channel_path = os.path.join(input_dir, channel_file)

    if not os.path.isfile(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    mat = spio.loadmat(channel_path, squeeze_me=True, struct_as_record=False)

    # Expect AlignedSampleNum to be in mat file
    # We also expect variables named like 'AdcData_scanline%03d_thi0_sa0'
    aligned_samples = int(getattr(mat.get('AlignedSampleNum', None), 'item', mat.get('AlignedSampleNum', None)) if 'AlignedSampleNum' in mat else mat.get('AlignedSampleNum', None) or getattr(info, 'AlignedSampleNum', None))
    if aligned_samples is None:
        aligned_samples = int(info.AlignedSampleNum)
    data_total = int(np.floor(aligned_samples))
    data_total1 = int(getattr(info, 'Nfocus', info.Nfocus))

    N_ch = int(info.N_ch)
    Nsc = int(info.Nsc)
    N_ele = int(info.N_ele)

    # Precompute Rx mux lookup table
    rx_half_ch = N_ch * 0.5
    rx_ch_mtx = np.arange(-int(rx_half_ch), int(rx_half_ch))
    SCvsEle = info.dx / info.pitch

    # RxMux: shape (Nsc, N_ch)
    sc_indices = np.arange(Nsc)
    base_idx = np.floor(sc_indices * SCvsEle).astype(int)  # zero-based offset
    RxMux = (base_idx[:, None] + 1) + rx_ch_mtx[None, :]  # 1-based like MATLAB

    # Load and reconstruct per-scanline data into Reconstruction array
    # Reconstruction shape (AlignedSampleNum, N_ch, Nsc)
    Reconstruction = np.zeros((aligned_samples, N_ch, Nsc), dtype=np.float64)

    for sc in range(Nsc):
        key = f'AdcData_scanline{sc:03d}_thi0_sa0'  # sc is zero-based like MATLAB used sc-1
        if key not in mat:
            # try alternative keys or raise
            # allow both sc and sc-1 naming differences:
            key_alt = f'AdcData_scanline{sc-1:03d}_thi0_sa0' if sc > 0 else None
            if key_alt and key_alt in mat:
                key = key_alt
            else:
                # raise a helpful error
                raise KeyError(f"Could not find expected variable {key} in {channel_path}. Keys: {list(mat.keys())[:20]}")
        imgtmp = mat[key].astype(np.float64, copy=False)
        # produce (AlignedSampleNum, N_ch) block
        img_block = ch2sc_us(sc, N_ele, Nsc, N_ch, imgtmp)
        # Transpose in MATLAB had Reconstruction(:,:,sc) = img';
        # Our img_block is (AlignedSampleNum, N_ch) -> OK
        Reconstruction[:, :, sc] = img_block

    # Precompute DC cancellation FIR filter (fir2 equivalent)
    # MATLAB: f = [0 0.1 0.1 1]; m = [0 0 1 1]; DC_cancel = fir2(64,f,m)
    # Use firwin2: provide frequency response at nodes (0..1)
    f_nodes = np.array([0.0, 0.1, 0.1, 1.0])
    m_nodes = np.array([0.0, 0.0, 1.0, 1.0])
    # firwin2 requires frequency nodes be in increasing order and len(f_nodes)==len(m_nodes)
    DC_cancel = firwin2(65, f_nodes, m_nodes)  # length = 64 in MATLAB -> order 64 means 65 taps

    # Prepare outputs
    RF_Sum = np.zeros((data_total1, Nsc), dtype=np.float64)

    # Precompute depth/time arrays
    # info.d_sample is expected to be array of sample depths (meters) with length >= data_total1
    d_sample = np.asarray(info.d_sample)
    if d_sample.ndim == 0:
        # if scalar was given, create a vector of sample depths
        d_sample = np.arange(data_total1) * float(info.d_sample)
    # make sure d_sample has size >= data_total1
    if d_sample.size < data_total1:
        raise ValueError("infoUS.d_sample length is smaller than Nfocus/data_total1; please supply depth vector.")

    # For each scanline
    for sc in range(Nsc):
        if verbose and (sc % max(1, Nsc // 10) == 0):
            print(f"Beamforming scanline {sc+1}/{Nsc} ...")

        ChData = Reconstruction[:, :, sc]  # shape (aligned_samples, N_ch)
        MuxTbl = RxMux[sc, :].astype(int)  # 1-based positions

        # accumulate per-channel TOF-adjusted and apodized signals into a (data_total1, N_ch) matrix
        RF_scanline_tofadjusted = np.zeros((data_total1, N_ch), dtype=np.float64)
        active_ch = []

        sc_pos = float(info.ScanPosition[sc]) if np.ndim(info.ScanPosition) else float(info.ScanPosition)
        # tx_t constant across depths: t_tx = d_sample / c (vector)
        tx_t = d_sample / info.c  # vector length data_total1

        for ch in range(N_ch):
            ElePos = int(MuxTbl[ch])
            if 1 <= ElePos <= N_ele:
                active_ch.append(ch)
                tmp = ChData[:, ch]  # length aligned_samples
                # DC cancel filter (same as conv with 'same')
                fil_tmp = convolve(tmp, DC_cancel, mode='same')

                ele_pos = float(info.ElePosition[ElePos - 1]) if hasattr(info.ElePosition, '__len__') else float(info.ElePosition)
                x = np.abs(sc_pos - ele_pos)  # scalar

                # rx_t is vector across depths
                rx_t = np.sqrt(x ** 2 + d_sample ** 2) / info.c
                TOF = tx_t + rx_t
                RxReadPointer = np.round(TOF * info.fs).astype(int)  # 1-based indices in MATLAB

                # clamp indices to valid range [1, data_total]
                # convert to 0-based indexing for Python
                idx0 = RxReadPointer - 1
                below = idx0 < 0
                above = idx0 >= data_total
                idx0_clipped = np.clip(idx0, 0, data_total - 1)

                rf_tmp = fil_tmp[idx0_clipped]  # vector length data_total1
                rf_tmp[below] = 0.0
                rf_tmp[above] = 0.0

                # Rx apodization per depth
                Rx_apod_idx = np.zeros_like(d_sample)
                # h_aper_size = d_sample / RxFnum * 0.5
                h_aper_size = d_sample / info.RxFnum * 0.5
                # be careful: some elements may be scalars or arrays
                idx_mask = h_aper_size >= info.half_rx_ch
                # For indices where h_aper_size >= half_rx_ch
                Rx_apod_idx[idx_mask] = x / info.half_rx_ch
                # For other indices
                Rx_apod_idx[~idx_mask] = x / h_aper_size[~idx_mask]
                Rx_apo_r = np.ones_like(Rx_apod_idx)
                Rx_apo_r[Rx_apod_idx >= 1.0] = 0.0

                RF_scanline_tofadjusted[:, ch] = (rf_tmp * Rx_apo_r)

        if len(active_ch) == 0:
            continue

        # Apply depth-dependent gain: filt = 1 + 5e-3 * (1 : rows)
        # MATLAB used 1:size(RF_scanline_tofadjusted,1)
        filt = 1.0 + 5.0e-3 * (np.arange(1, RF_scanline_tofadjusted.shape[0] + 1))
        RF_scanline_tofadjusted = RF_scanline_tofadjusted * filt[:, None]

        # Apodization weighting across active channels
        active_ch = np.array(active_ch, dtype=int)
        L_window = active_ch.size

        if apod_method.lower() == 'boxcar':
            w_apod = np.ones(L_window, dtype=np.float64)
            sum_tmp = np.sum(RF_scanline_tofadjusted[:, active_ch] * w_apod[None, :], axis=1)

        elif apod_method.lower() == 'hann':
            # MATLAB hann across N_ch
            n = np.arange(1, N_ch + 1)
            hann_full = 0.5 * (1.0 - np.cos(2.0 * np.pi / (N_ch - 1) * n))
            w_apod = hann_full[active_ch - 1] if np.any(active_ch > 0) else hann_full[active_ch]
            sum_tmp = np.sum(RF_scanline_tofadjusted[:, active_ch] * w_apod[None, :], axis=1)

        elif apod_method.lower() == 'kaiser':
            # match MATLAB default kaiser(infoUS.N_ch,16)
            from scipy.signal import kaiser
            kaiser_full = kaiser(N_ch, beta=16.0)
            w_apod = kaiser_full[active_ch]
            sum_tmp = np.sum(RF_scanline_tofadjusted[:, active_ch] * w_apod[None, :], axis=1)

        elif apod_method.lower() == 'mv':
            # Minimum variance apodization (sampled / windowed). This is heavier but vectorized.
            L_subarray = 12
            L_pw = int(round(info.fs / info.fc))
            # Build corr_array: shape (N_subarrays, L_subarray, data_total1)
            n_sub = N_ch - L_subarray + 1
            if n_sub <= 0:
                raise ValueError("N_ch smaller than L_subarray for MV apodization.")
            # Use buffering with stride tricks or explicit loop to build corr_array
            # We'll vectorize: for each depth z, collect sliding windows across channels
            # corr_array_sub shape: (n_sub, L_subarray, data_total1)
            corr_array = np.zeros((n_sub, L_subarray, data_total1), dtype=np.float64)
            # RF_scanline_tofadjusted shape: (data_total1, N_ch)
            arr = RF_scanline_tofadjusted[:data_total1, :]  # ensure size
            # fill corr_array
            for idx_z in range(data_total1):
                row = arr[idx_z, :]  # shape (N_ch,)
                # sliding windows
                for s in range(n_sub):
                    corr_array[s, :, idx_z] = row[s:s + L_subarray]

            # compute correlation matrices
            corr_matrix = np.zeros((L_subarray, L_subarray, data_total1), dtype=np.float64)
            corr_matrix2 = np.zeros_like(corr_matrix)
            for s in range(n_sub):
                block = corr_array[s, :, :]  # shape (L_subarray, data_total1)
                # block @ block.T across windows:
                # for each depth z: corr += block[:,z] @ block[:,z].T
                corr_matrix += np.einsum('ik,jk->ijk', block, block)
                # flipped correlation (reverse order)
                corr_matrix2 += np.einsum('ik,jk->ijk', block[::-1, :], block[::-1, :])

            # moving mean along depth axis (3rd dim) with window L_pw (simple uniform filter)
            def movmean(a, window):
                if window <= 1:
                    return a
                kernel = np.ones(window) / window
                # apply along last axis
                from scipy.signal import convolve
                out = np.empty_like(a)
                # loop over i,j dims
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        out[i, j, :] = convolve(a[i, j, :], kernel, mode='same')
                return out

            corr_matrix = movmean(corr_matrix, L_pw)
            corr_matrix2 = movmean(corr_matrix2, L_pw)

            a_tmp = np.ones((L_subarray, 1), dtype=np.float64)
            corr_array_sum = np.mean(corr_array, axis=0)  # shape (L_subarray, data_total1)

            sum_tmp = np.zeros((data_total1,), dtype=np.float64)
            for idx_z in range(data_total1):
                C = 0.5 * corr_matrix[:, :, idx_z] + 0.5 * corr_matrix2[:, :, idx_z] + (0.2 / L_subarray) * np.trace(corr_matrix[:, :, idx_z]) * np.eye(L_subarray)
                # solve C * w = a and normalize
                try:
                    w_apod_mv = np.linalg.solve(C, a_tmp).ravel()
                    denom = float(a_tmp.T @ np.linalg.solve(C, a_tmp))
                    if denom != 0:
                        w_apod_mv = w_apod_mv / denom
                    else:
                        w_apod_mv = np.zeros_like(w_apod_mv)
                    # corr_array_sum[:, idx_z] is the averaged array element for that depth
                    sum_tmp[idx_z] = float(w_apod_mv @ corr_array_sum[:, idx_z])
                except np.linalg.LinAlgError:
                    sum_tmp[idx_z] = 0.0
            # replace NaNs
            sum_tmp = np.nan_to_num(sum_tmp, nan=0.0)

        else:
            raise ValueError(f"Apodization method '{apod_method}' not available.")

        # Coherence gating
        if coherence_method.lower() == 'none':
            CF = np.ones_like(sum_tmp)
        elif coherence_method.lower() == 'cf':
            # simple coherence factor (CF)
            r0 = 2
            start_left = int(round(N_ch / 2 - r0))
            start_right = int(round(N_ch / 2 + r0 + 1))
            # compute idx_left and idx_right arrays (MATLAB used a depth-dependent adjustment)
            # approximate MATLAB behavior:
            idx_left = np.maximum(active_ch[0], start_left - (np.arange(1, data_total1 + 1) * 64 // data_total1))
            idx_right = np.minimum(active_ch[-1], start_right + (np.arange(1, data_total1 + 1) * 64 // data_total1))
            row_sum_abs = np.zeros((data_total1,), dtype=np.float64)
            row_abs_sum = np.zeros_like(row_sum_abs)
            for t in range(data_total1):
                left = int(idx_left[t])
                right = int(idx_right[t])
                # ensure indices are valid and within active channels
                if right < left:
                    rf_temp = np.array([], dtype=np.float64)
                else:
                    # pick indices in that range from full channel indices
                    # map left/right (which are channel positions) to indices from active_ch
                    # For simplicity, pick the active channels within the channel range
                    mask = (active_ch >= left) & (active_ch <= right)
                    rf_temp = RF_scanline_tofadjusted[t, active_ch[mask]]
                row_sum_abs[t] = (np.sum(rf_temp) ** 2) if rf_temp.size > 0 else 0.0
                abs_rf = np.abs(rf_temp)
                row_abs_sum[t] = (np.sum(abs_rf ** 2) * max(1, rf_temp.size))
            CF = np.divide(row_sum_abs, row_abs_sum, out=np.ones_like(row_sum_abs), where=row_abs_sum != 0)
            CF = np.nan_to_num(CF, nan=1.0, posinf=1.0, neginf=1.0)

        elif coherence_method.lower() in ('gsf', 'gsf2'):
            kernel_size = int(round(info.fs / info.fc))
            Lmax_lag = 10
            arr = RF_scanline_tofadjusted[:, active_ch]  # shape (data_total1, L_active)
            # square_root4_terms = nthroot(kernel_size * movmean(arr^2, kernel_size, axis=1), 4)
            # Implement movmean along channels dimension with kernel kernel_size
            # Note: MATLAB used movmean along channels; we'll approximate with uniform filtering across channels
            # but vectorized per depth:
            sq = arr ** 2
            # moving mean across channels with window = kernel_size
            import scipy.ndimage as ndi
            mean_sq = ndi.uniform_filter1d(sq, size=kernel_size, axis=1, mode='nearest')
            square_root4_terms = (kernel_size * mean_sq) ** 0.25
            sc_terms = np.divide(arr, square_root4_terms, out=np.zeros_like(arr), where=square_root4_terms != 0)
            sc_terms = np.nan_to_num(sc_terms, nan=0.0)

            # build sc_terms_hat: for each lag, sum over next Lmax_lag channels
            L_active = sc_terms.shape[1]
            if L_active <= 1:
                CF = np.zeros((data_total1,))
            else:
                sc_terms_hat = np.zeros((data_total1, L_active - 1), dtype=np.float64)
                for idx_i in range(L_active - 1):
                    end = min(Lmax_lag + idx_i + 1, L_active)
                    sc_terms_hat[:, idx_i] = np.sum(sc_terms[:, (idx_i + 1):end], axis=1)
                if coherence_method.lower() == 'gsf':
                    CF = np.sum(sc_terms_hat * sc_terms[:, 0:(L_active - 1)], axis=1)
                else:  # gsf2 includes additional DC compensation
                    f_nodes2 = np.array([0.0, 0.03, 0.03, 1.0])
                    m_nodes2 = np.array([1.0, 1.0, 0.0, 0.0])
                    DC_comp_gsf = firwin2(65, f_nodes2, m_nodes2)
                    DC_comp_das = DC_comp_gsf.copy()  # same nodes in MATLAB; use same filter
                    sc_terms_hat_mul = sc_terms_hat * sc_terms[:, 0:(L_active - 1)]
                    # conv2 replacement via convolve along rows
                    dc_das = np.apply_along_axis(lambda row: convolve(row, DC_comp_das, mode='same'), 1, arr)
                    dc_gsf = np.apply_along_axis(lambda row: convolve(row, DC_comp_gsf, mode='same'), 1, sc_terms_hat_mul)
                    numerator = np.mean(dc_gsf - sc_terms_hat_mul, axis=1)
                    denom = np.sqrt(np.mean((dc_das - arr - np.mean(dc_das - arr, axis=1, keepdims=True)) ** 2, axis=1))
                    CF = numerator * denom

        else:
            raise ValueError(f"Coherence gating method '{coherence_method}' not available.")

        # finalize RF_Sum for this scanline
        if apod_method.lower() == 'mv':
            RF_Sum[:, sc] = sum_tmp * CF
        else:
            RF_Sum[:, sc] = sum_tmp * CF

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

    # Adjust final image depth & resize
    Nt = RF_log.shape[0]
    convert_factor = 1.0
    RF_log = RF_log[: int(round(convert_factor * Nt)), :]

    # Compute target image size
    xz_ratio = info.FOV / np.max(d_sample) / convert_factor
    Lx = 4 * Nsc
    Lz = int(round(Lx / xz_ratio)) if xz_ratio != 0 else RF_log.shape[0]
    # Use scipy.ndimage.zoom to resize to (Lz, Lx)
    zoom_z = Lz / RF_log.shape[0] if RF_log.shape[0] > 0 else 1.0
    zoom_x = Lx / RF_log.shape[1] if RF_log.shape[1] > 0 else 1.0
    RF_log_resized = zoom(RF_log, (zoom_z, zoom_x), order=1)  # bilinear interpolation

    return RF_Sum, RF_env_raw, RF_log_resized
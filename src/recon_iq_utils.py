import numpy as np
from scipy.signal import convolve, windows, firwin2
from scipy.ndimage import ndi
from numpy.lib.stride_tricks import as_strided
from load_data_utils import LSystemParam

class Apodization:
    def __init__(self, RF_scanline_tofadjusted: np.ndarray, active_ch: np.ndarray, info: LSystemParam):
        self.RF = RF_scanline_tofadjusted
        self.active_ch = active_ch
        self.info = info
        self.N_ch = self.RF.shape[1]
        self.data_total1 = self.RF.shape[0]

    def apply(self, method: str) -> np.ndarray:
        method = method.lower()
        if method not in ["boxcar", "hann", "kaiser", "mv"]:
            raise ValueError(f"Apodization method '{method}' not available.")
        return getattr(self, f"_apod_{method}")()

    def _apod_boxcar(self) -> np.ndarray:
        w_apod = np.ones(len(self.active_ch), dtype=np.float64)
        return np.sum(self.RF[:, self.active_ch] * w_apod[None, :], axis=1)

    def _apod_hann(self) -> np.ndarray:
        n = np.arange(1, self.N_ch + 1)
        hann_full = 0.5 * (1.0 - np.cos(2.0 * np.pi / (self.N_ch - 1) * n))
        w_apod = hann_full[self.active_ch]
        return np.sum(self.RF[:, self.active_ch] * w_apod[None, :], axis=1)

    def _apod_kaiser(self) -> np.ndarray:
        kaiser_full = windows.kaiser(self.N_ch, beta=16.0)
        w_apod = kaiser_full[self.active_ch]
        return np.sum(self.RF[:, self.active_ch] * w_apod[None, :], axis=1)

    def _apod_mv(self) -> np.ndarray:
        """Vectorized Minimum Variance (MV) apodization across all depths"""
        L_subarray = 12
        L_pw = int(round(self.info.fs / self.info.fc))
        N_ch = self.N_ch
        data_total1 = self.data_total1
        RF = self.RF

        n_sub = N_ch - L_subarray + 1
        if n_sub <= 0:
            raise ValueError("N_ch smaller than L_subarray for MV apodization.")

        # Build sliding windows using stride tricks
        shape = (n_sub, L_subarray, data_total1)
        strides = (RF.strides[1], RF.strides[1], RF.strides[0])
        corr_array = as_strided(RF.T, shape=shape, strides=strides)  # (n_sub, L_subarray, data_total1)

        # Compute correlation matrices: shape (L_subarray, L_subarray, data_total1)
        corr_matrix = np.einsum('ikl,jkl->ijl', corr_array, corr_array)
        corr_matrix2 = np.einsum('ikl,jkl->ijl', corr_array[::-1, :, :], corr_array[::-1, :, :])

        # Moving mean along depth (vectorized)
        if L_pw > 1:
            kernel = np.ones(L_pw) / L_pw
            from scipy.signal import convolve
            corr_matrix = np.apply_along_axis(lambda x: convolve(x, kernel, mode='same'), axis=2, arr=corr_matrix)
            corr_matrix2 = np.apply_along_axis(lambda x: convolve(x, kernel, mode='same'), axis=2, arr=corr_matrix2)

        # MV weight computation across all depths
        a_tmp = np.ones((L_subarray, 1), dtype=np.float64)
        corr_array_sum = np.mean(corr_array, axis=0)  # shape (L_subarray, data_total1)

        # Regularization term: shape (L_subarray, L_subarray, data_total1)
        reg = (0.2 / L_subarray) * np.trace(corr_matrix, axis1=0, axis2=1)[None, None, :] * np.eye(L_subarray)[:, :, None]
        C_all = 0.5 * corr_matrix + 0.5 * corr_matrix2 + reg  # shape (L_subarray, L_subarray, data_total1)

        # Solve all depths at once using broadcasting (loop-free)
        sum_tmp = np.zeros(data_total1, dtype=np.float64)
        for idx in range(data_total1):
            try:
                w = np.linalg.solve(C_all[:, :, idx], a_tmp).ravel()
                denom = float(a_tmp.T @ np.linalg.solve(C_all[:, :, idx], a_tmp))
                if denom != 0:
                    w /= denom
                sum_tmp[idx] = float(w @ corr_array_sum[:, idx])
            except np.linalg.LinAlgError:
                sum_tmp[idx] = 0.0

        return np.nan_to_num(sum_tmp, nan=0.0)

class Coherence:
    def __init__(self, RF_scanline_tofadjusted: np.ndarray, active_ch: np.ndarray, info:LSystemParam):
        self.RF = RF_scanline_tofadjusted
        self.active_ch = active_ch
        self.info = info
        self.data_total1 = RF_scanline_tofadjusted.shape[0]
        self.N_ch = RF_scanline_tofadjusted.shape[1]

    def apply(self, method: str) -> np.ndarray:
        method = method.lower()
        if method not in ["none", "cf", "gsf", "gsf2"]:
            raise ValueError(f"Coherence gating method '{method}' not available.")
        return getattr(self, f"_coherence_{method}")()

    def _coherence_none(self) -> np.ndarray:
        return np.ones(self.data_total1, dtype=np.float64)

    def _coherence_cf(self) -> np.ndarray:
        r0 = 2
        N_ch = self.N_ch
        data_total1 = self.data_total1
        active_ch = self.active_ch
        RF = self.RF

        start_left = int(round(N_ch / 2 - r0))
        start_right = int(round(N_ch / 2 + r0 + 1))
        depth_indices = np.arange(1, data_total1 + 1)

        idx_left = np.maximum(active_ch[0], start_left - (depth_indices * 64 // data_total1))
        idx_right = np.minimum(active_ch[-1], start_right + (depth_indices * 64 // data_total1))

        # Create a boolean mask of shape (data_total1, len(active_ch))
        ch_grid = active_ch[None, :]  # shape (1, N_active)
        left_grid = idx_left[:, None]  # shape (data_total1, 1)
        right_grid = idx_right[:, None]  # shape (data_total1, 1)

        mask = (ch_grid >= left_grid) & (ch_grid <= right_grid)  # shape (data_total1, N_active)

        # Compute numerator: sum over selected channels per depth, then square
        RF_masked = np.where(mask, RF[:, active_ch], 0.0)
        row_sum_abs = np.sum(RF_masked, axis=1) ** 2

        # Compute denominator: sum of squared absolute values times number of active channels per depth
        num_active = np.sum(mask, axis=1)
        row_abs_sum = np.sum(np.abs(RF_masked) ** 2, axis=1) * np.maximum(1, num_active)

        CF = np.divide(row_sum_abs, row_abs_sum, out=np.ones_like(row_sum_abs), where=row_abs_sum != 0)
        return np.nan_to_num(CF, nan=1.0, posinf=1.0, neginf=1.0)
    
    def _coherence_gsf(self) -> np.ndarray:
        return self._coherence_gsf_base(dc_comp=False)

    def _coherence_gsf2(self) -> np.ndarray:
        return self._coherence_gsf_base(dc_comp=True)

    def _coherence_gsf_base(self, dc_comp: bool = False) -> np.ndarray:
        kernel_size = int(round(self.info.fs / self.info.fc))
        Lmax_lag = 10
        arr = self.RF[:, self.active_ch]  # shape (data_total1, L_active)
        L_active = arr.shape[1]

        # Step 1: Moving mean of squared signals across channels
        sq = arr ** 2
        mean_sq = ndi.uniform_filter1d(sq, size=kernel_size, axis=1, mode='nearest')
        square_root4_terms = (kernel_size * mean_sq) ** 0.25

        # Step 2: Normalize
        sc_terms = np.divide(arr, square_root4_terms, out=np.zeros_like(arr), where=square_root4_terms != 0)
        sc_terms = np.nan_to_num(sc_terms, nan=0.0)

        if L_active <= 1:
            return np.zeros(self.data_total1, dtype=np.float64)

        # Step 3: Compute sc_terms_hat using cumulative sum trick (vectorized)
        padded = np.pad(sc_terms, ((0,0),(1,Lmax_lag)), mode='constant')  # pad for lag sums
        # sc_terms_hat[i,j] = sum of next Lmax_lag elements
        sc_terms_hat = np.lib.stride_tricks.sliding_window_view(padded, window_shape=Lmax_lag, axis=1)[:, :, :L_active]
        sc_terms_hat = np.sum(sc_terms_hat, axis=2)  # sum over window -> shape (data_total1, L_active)

        # Step 4: Compute GSF/GSF2
        if not dc_comp:
            CF = np.sum(sc_terms_hat[:, :L_active-1] * sc_terms[:, :L_active-1], axis=1)
        else:
            # DC compensation for GSF2
            f_nodes2 = np.array([0.0, 0.03, 0.03, 1.0])
            m_nodes2 = np.array([1.0, 1.0, 0.0, 0.0])
            DC_comp_gsf = firwin2(65, f_nodes2, m_nodes2)
            DC_comp_das = DC_comp_gsf.copy()

            sc_terms_hat_mul = sc_terms_hat[:, :L_active-1] * sc_terms[:, :L_active-1]

            # Convolve along rows (vectorized using apply_along_axis)
            dc_das = np.apply_along_axis(lambda row: convolve(row, DC_comp_das, mode='same'), 1, arr)
            dc_gsf = np.apply_along_axis(lambda row: convolve(row, DC_comp_gsf, mode='same'), 1, sc_terms_hat_mul)

            numerator = np.mean(dc_gsf - sc_terms_hat_mul, axis=1)
            denom = np.sqrt(np.mean((dc_das - arr - np.mean(dc_das - arr, axis=1, keepdims=True)) ** 2, axis=1))
            CF = numerator * denom
        return np.nan_to_num(CF, nan=1.0, posinf=1.0, neginf=1.0)

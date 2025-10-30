import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple
from scipy.signal import convolve, windows, firwin2
from scipy.ndimage import uniform_filter1d
from scipy.sparse.linalg import lsqr, LinearOperator, cg
from scipy.optimize import lsq_linear
from scipy.linalg import cho_factor, cho_solve
from numpy.lib.stride_tricks import as_strided
from .load_data_utils import LinearSystemParam
from tqdm import tqdm

class PATInverseSolver:
    def __init__(self, G: np.ndarray, RF_data: np.ndarray):
        self.G = G
        self.RF_data = RF_data
        self.mu = None
        self.M, self.N = G.shape
        self.stripe_width_vox = 10

    def _solve_lsqr(self, atol: float = 1e-6, btol: float = 1e-6, iter_lim: int = 1000, damp: float = 0.0) -> np.ndarray:
        res = lsqr(self.G, self.RF_data, atol=atol, btol=btol, iter_lim=iter_lim, damp=damp)
        x = res[0]
        self.mu = x
        return x
    
    def _solve_lsqr_tikhonov(G_sub: np.ndarray, b: np.ndarray, lambda_reg: float = 1e-3, iter_lim:int=100):
        damp = np.sqrt(lambda_reg)
        res = lsqr(G_sub.astype(np.float32), b.astype(np.float32), damp=damp, iter_lim=iter_lim)
        x = res[0].astype(np.float32)
        return x
    
    def _solve_tikhonov(self, lambda_reg: float = 5e-3, maxiter: int = 200, x0: Optional[np.ndarray] = None) -> np.ndarray:
        '''
        G = self.G
        b = self.RF_data
        N = self.N
        L_strip = self.stripe_width_vox
        mu = np.zeros(N, dtype=np.float32) if x0 is None else x0.astype(np.float32, copy=False)
        colnorms = np.linalg.norm(G, axis=0)
        colnorms[colnorms == 0] = 1.0

        starts = list(range(0,N,L_strip))
        for s in tqdm(starts , desc="Tikhonov stripes"):
            e = min(N, s + L_strip)
            G_sub = G[:, s:e].astype(np.float32)
            G_sub_norm = G_sub / colnorms[s:e][None, :]
            x_sub = _solve_lsqr_tikhonov(G_sub_norm, b, lambda_reg = lambda_reg, iter_lim = maxiter)
            # denormalize
            mu[s:e] = x_sub / colnorms[s:e]
        self.mu = mu
        '''
        damp = np.sqrt(lambda_reg)
        res = lsqr(self.G, self.RF_data, atol=1e-6, btol=1e-6, iter_lim=maxiter, damp=damp)
        x = res[0]
        self.mu = x
        return x
    
    def _solve_nnls(self, bounds: Tuple[float, float] = (0.0, np.inf), max_iter: int = 2000, tol: float = 1e-6) -> np.ndarray:
        res = lsq_linear(self.G, self.RF_data, bounds=bounds, method='trf', max_iter=max_iter, tol=tol, verbose=1)
        self.mu = res.x
        return res.x
    
    def _estimate_lipschitz(self, num_iter: int = 20) -> float:
        """
        Estimate Lipschitz constant L = ||G||_2^2 via power iteration on G^T G.
        """
        n = self.n
        x = np.random.randn(n).astype(np.float64)
        for _ in range(num_iter):
            if isinstance(self.G, LinearOperator):
                Gx = self.G.matvec(x)
                GTGx = self.G.rmatvec(Gx)
            else:
                Gx = self.G.dot(x)
                GTGx = self.G.T.dot(Gx)
            norm = np.linalg.norm(GTGx)
            if norm == 0:
                return 1.0
            x = GTGx / norm
        # Rayleigh quotient approx for largest eigenvalue of G^T G
        if isinstance(self.G, LinearOperator):
            Gx = self.G.matvec(x)
            GTGx = self.G.rmatvec(Gx)
        else:
            Gx = self.G.dot(x)
            GTGx = self.G.T.dot(Gx)
        eig_approx = np.dot(x, GTGx)
        return float(eig_approx) if eig_approx > 0 else 1.0
    
    def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
    
    def _solve_l1(self,
                  lambda_reg: float = 5e-3,
                  rho: float = 1.0,
                  max_admm_iters: int = 30,
                  lsqr_maxiter: int = 20,
                  abstol: float = 1e-4,
                  reltol: float = 1e-3,
                  verbose: bool = True,
                  x0: Optional[np.ndarray] = None) -> np.ndarray:
        G = self.G.astype(np.float32, copy=False)
        b = self.RF_data.astype(np.float32, copy=False)

        is_op = isinstance(G, LinearOperator)
        if not is_op:
            G = np.asarray(G, dtype=np.float32)
            m, n = G.shape
            def matvec_G(v): return G.dot(v)
            def rmatvec_G(w): return G.T.dot(w)
        else:
            m, n = G.shape
            def matvec_G(v): return G.matvec(v).astype(np.float32)
            if hasattr(G, 'rmatvec'):
                def rmatvec_G(w): return G.rmatvec(w).astype(np.float32)
            else:
                raise ValueError("LinearOperator G must provide rmatvec for ADMM LSQR solver.")

        # initial x, z, u (scaled dual variable u)
        if x0 is None:
            x = np.zeros(n, dtype=np.float32)
        else:
            x = np.asarray(x0, dtype=np.float32)
        z = x.copy()
        u = np.zeros_like(x, dtype=np.float32)  # scaled dual variable: u = y / rho

        sqrt_rho = np.float32(np.sqrt(rho))
        lam_over_rho = np.float32(lambda_reg / rho)

        # precompute operator for augmented system: A_aug matvec and rmatvec
        # A_aug: (m + n) x n  ; matvec(x) -> [Gx; sqrt(rho)*x]
        def A_aug_matvec(v: np.ndarray) -> np.ndarray:
            return np.concatenate([matvec_G(v).astype(np.float32),
                                (sqrt_rho * v).astype(np.float32)]).astype(np.float32)

        # rmatvec(y_aug) = G^T y0 + sqrt(rho) * y1, where y_aug = [y0; y1]
        def A_aug_rmatvec(y_aug: np.ndarray) -> np.ndarray:
            y0 = y_aug[:m].astype(np.float32)
            y1 = y_aug[m:].astype(np.float32)
            return (rmatvec_G(y0) + sqrt_rho * y1).astype(np.float32)

        A_aug = LinearOperator((m + n, n), matvec=A_aug_matvec, rmatvec=A_aug_rmatvec, dtype=np.float32)
        b_norm = norm(b)
        if b_norm == 0:
            b_norm = 1.0

        # ADMM iterations
        for k in range(max_admm_iters):
            # x-update: solve min_x 0.5||Gx - b||^2 + (rho/2)||x - z + u||^2 via LSQR on augmented system
            rhs_aug = np.concatenate([b, sqrt_rho * (z - u)]).astype(np.float32)

            # Use lsqr with the LinearOperator A_aug; limit iterations for speed
            # lsqr returns tuple; solution is first element
            lsqr_res = lsqr(A_aug, rhs_aug, atol=1e-6, btol=1e-6, iter_lim=lsqr_maxiter)
            x_new = np.asarray(lsqr_res[0], dtype=np.float32)

            # z-update: soft-thresholding on x_new + u
            x_plus_u = x_new + u
            z_new = np.sign(x_plus_u) * np.maximum(np.abs(x_plus_u) - lam_over_rho, 0.0)
            z_new = np.maximum(z_new, 0.0, dtype=np.float32)
            u_new = u + (x_new - z_new)

            # compute diagnostics
            prim_res = norm(x_new - z_new)                         # primal residual ||x - z||
            dual_res = rho * norm(z_new - z)                      # dual residual rho ||z - z_prev||
            # objective: 0.5||Gx - b||^2 + lambda ||z||
            r = matvec_G(x_new) - b
            obj = 0.5 * float(np.dot(r.astype(np.float32), r.astype(np.float32))) + lambda_reg * float(np.sum(np.abs(z_new)))
            # update variables
            x , z , u = x_new , z_new , u_new
            # stopping criterion
            eps_primal = np.sqrt(n) * abstol + reltol * max(norm(x), norm(z))
            eps_dual = np.sqrt(n) * abstol + reltol * norm(rho * u)
            if verbose:
                print(f"ADMM iter {k+1:3d}: obj={obj:.6e}, prim_res={prim_res:.3e}, dual_res={dual_res:.3e}, eps_primal={eps_primal:.3e}, eps_dual={eps_dual:.3e}")

            if prim_res <= eps_primal and dual_res <= eps_dual:
                if verbose:
                    print(f"Converged (ADMM iter {k+1}).")
                break
        self.mu = x
        return x

    def _solve_fista_l1(self, lam: float = 1e-3, maxiter: int = 500, L0: Optional[float] = None, nonneg: bool = False, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        L1-regularized least squares via FISTA:
            min_x 0.5||G x - b||_2^2 + lam * ||x||_1
        """
        G = self.G.astype(np.float32, copy=False)
        b = self.RF_data.astype(np.float32, copy=False)
        if x0 is None:
            try:
                # Solve (G G^T) y = b  =>  x0 = G^T y
                GGt = G @ G.T
                GGt += np.eye(GGt.shape[0], dtype=np.float32) * np.float32(1e-6)
                c, low = cho_factor(GGt, overwrite_a=True, check_finite=False)
                y = cho_solve((c, low), b, check_finite=False)
                x0 = G.T @ y
            except np.linalg.LinAlgError:
                x0 = G.T @ b
            x0 = x0.astype(np.float32, copy=False)
        yk = x0.copy()
        t_k = 1.0

        # compute Lipschitz constant estimate L = ||G||_2^2 (approx). We'll use power iteration if not provided.
        if L0 is None:
            L = self._estimate_lipschitz(num_iter=20)
        else:
            L = float(L0)
        invL = 1.0 / L

        for k in tqdm(range(maxiter), desc="FISTA iterations"):
            # gradient = G^T (G yk - b)
            Gy = (self.G.dot(yk) if not isinstance(self.G, LinearOperator) else self.G.matvec(yk))
            grad = (self.G.T.dot(Gy - self.b) if not isinstance(self.G, LinearOperator) else self.G.rmatvec(Gy - self.b))

            x_next = yk - invL * grad
            # soft-threshold
            thresh = lam * invL
            # soft-threshold operator
            x_next = np.sign(x_next) * np.maximum(np.abs(x_next) - thresh, 0.0)
            if nonneg:
                x_next = np.maximum(x_next, 0.0)

            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            yk = x_next + ((t_k - 1.0) / t_next) * (x_next - x)
            x = x_next
            t_k = t_next

        self.mu = x
        return x
    
    def reconstruct(self, method: str = 'pseudoinverse', **kwargs) -> np.ndarray:
        method = method.lower()
        if method == 'pseudoinverse' or method == 'lsqr':
            return self._solve_lsqr(**kwargs)
        elif method == 'tikhonov' or method == 'l2' or method == 'ridge':
            return self._solve_tikhonov(**kwargs)
        elif method == 'nnls' or method == 'nonnegative' or method == 'nonneg':
            return self._solve_nnls(**kwargs)
        elif method == 'fista' or method == 'l1' or method == 'lasso' or method == 'sparse':
            return self._solve_l1(**kwargs)
        else:
            raise ValueError("Unknown method: choose 'lsqr','l2','nonneg','l1'.")
    

class Apodization:
    def __init__(self, RF_scanline_tofadjusted: np.ndarray, active_ch: np.ndarray, info: LinearSystemParam):
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
    def __init__(self, RF_scanline_tofadjusted: np.ndarray, active_ch: np.ndarray, info:LinearSystemParam):
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
        mean_sq = uniform_filter1d(sq, size=kernel_size, axis=1, mode='nearest')
        mean_sq_clipped = np.maximum(mean_sq, 0.0)
        square_root4_terms = (kernel_size * mean_sq_clipped) ** 0.25
        safe_den = np.where(square_root4_terms > 0.0, square_root4_terms, np.inf)
        sc_terms = arr / safe_den
        sc_terms = np.nan_to_num(sc_terms, nan=0.0, posinf=0.0, neginf=0.0)

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

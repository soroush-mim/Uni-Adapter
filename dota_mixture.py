import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class DOTA_mix(nn.Module):
    """
    DOTA with Diagonal Gaussian Mixture Model per class (Streaming).
    Compatible with original DOTA pipeline (predict returns same scale as original).
    
    Key fixes vs previous version:
    1. sigma_init must match actual CLIP embedding variance (~0.002-0.005)
    2. delta_scale for mode offsets must be small (relative to embedding scale)
    3. predict() returns scores compatible with CLIP logits scale
    4. var is clamped after every update
    5. Diagonal covariance enforced - no full matrix needed
    """

    def __init__(self, cfg, input_shape, num_classes, clip_weights, num_modes=4,
                 streaming_update_Sigma=True):
        """
        Args:
            cfg: dict with keys:
                 'epsilon'   - regularization (default 0.001)
                 'sigma'     - initial variance (default 0.004 for CLIP embeddings!)
                 'alpha_max' - max blending weight for prior (default 0.5)
            input_shape:  D (e.g. 512 for CLIP ViT-B/32)
            num_classes:  K
            clip_weights: (D, K) - CLIP zero-shot weights, L2-normalized columns
            num_modes:    M - number of Gaussians per class
        """
        super(DOTA_mix, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.streaming_update_Sigma = streaming_update_Sigma

        self.epsilon = cfg.get('epsilon', 0.001)

        # CRITICAL: for L2-normalized CLIP embeddings, each dimension has
        # variance ≈ 1/D ≈ 0.002 for D=512. Use this as sigma_init!
        # If user passes sigma=1.0 (wrong), we correct it automatically.
        sigma_cfg = cfg.get('sigma', 1.0)
        if sigma_cfg >= 0.1:
            # User probably set sigma=1.0 thinking of full covariance scale.
            # For diagonal CLIP embeddings, correct to 1/D
            self.sigma_init = 1.0 / input_shape
            print(f"[DOTA-GMM] Warning: sigma={sigma_cfg} is too large for CLIP embeddings. "
                  f"Auto-corrected to 1/D = {self.sigma_init:.5f}")
        else:
            self.sigma_init = sigma_cfg

        self.alpha_max = cfg.get('alpha_max', 0.5)

        # --------------------------------------------------------
        # 1. Initialize Means
        # --------------------------------------------------------
        # clip_weights: (D, K) -> clip_mu: (K, D)
        clip_mu = clip_weights.T.to(self.device).float()

        # mu shape: (K, M, D)
        self.mu = torch.zeros(num_classes, num_modes, input_shape, device=self.device)

        # Modes are initialized around the CLIP center with small orthogonal offsets.
        # delta_scale must be small relative to embedding norm (~1.0 for L2-normed).
        # We use sigma_init * 0.1 so modes start very close to center.
        delta_scale = self.sigma_init * 0.1

        for k in range(num_classes):
            center = clip_mu[k]  # (D,)
            # Each mode gets a tiny offset along a distinct axis → breaks symmetry
            # without moving far from the CLIP center.
            offsets = torch.zeros(num_modes, input_shape, device=self.device)
            for m in range(num_modes):
                offsets[m, m % input_shape] = delta_scale * (m + 1)
            self.mu[k] = center.unsqueeze(0) + offsets  # (M, D)
            # use random noise to perturb the mu
        # --------------------------------------------------------
        # 2. Initialize Diagonal Variances
        # --------------------------------------------------------
        # var shape: (K, M, D) - diagonal of covariance matrix
        # Each mode starts with sigma_init but slightly different → breaks symmetry
        self.var = torch.ones(num_classes, num_modes, input_shape,
                              device=self.device) * self.sigma_init

        # Tiny symmetry-breaking noise per mode (NOT per element, to keep diagonal structure)
        # Each mode gets a slightly different scalar multiplier
        for m in range(num_modes):
            scale_m = 1.0 + 0.05 * m  # mode 0: 1.0x, mode 1: 1.05x, ...
            self.var[:, m, :] *= scale_m

        # Ensure positivity
        self.var = torch.clamp(self.var, min=1e-8)

        # --------------------------------------------------------
        # 3. Initialize Mixture Weights and Counts
        # --------------------------------------------------------
        # pi: (K, M) - uniform start
        self.pi = torch.ones(num_classes, num_modes, device=self.device) / num_modes

        # c: (K, M) - soft counts, initialized to small positive value
        # (avoids division by zero at first update)
        self.c = torch.full((num_classes, num_modes), 1.0 / (num_modes), #num_modes * a hyperparam
                            device=self.device, dtype=torch.float32)

        # Class-level counts for prior estimation
        self.class_counts = torch.zeros(num_classes, device=self.device)
        self.t = 0

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_var(self):
        """Regularized diagonal variance, always positive."""
        return torch.clamp(self.var + self.epsilon, min=1e-8)

    def _log_likelihood(self, x, mu, var):
        """
        Diagonal Gaussian log-likelihood.

        Args:
            x:   (B, D)
            mu:  (K, M, D)
            var: (K, M, D)  - diagonal variance

        Returns:
            log_lik: (B, K, M)

        Formula: -0.5 * [sum_d log(var_d) + sum_d (x_d - mu_d)^2/var_d]
        Note: we DROP the D*log(2π) constant because:
          - It is identical for all (k,m) → cancels in softmax/logsumexp over classes
          - Dropping it keeps scores in a reasonable numerical range
          - This matches the original DOTA predict() which also drops constants
        """
        B, D = x.shape
        K, M, _ = mu.shape

        # diff: (B, K, M, D)
        diff = x.unsqueeze(1).unsqueeze(2) - mu.unsqueeze(0)

        # var_b: (1, K, M, D)
        var_b = var.unsqueeze(0)

        # Mahalanobis term: (B, K, M)
        maha = torch.sum(diff ** 2 / var_b, dim=-1)

        # Log determinant term: (1, K, M) (sum of log of diagonal)
        log_det = torch.sum(torch.log(var_b), dim=-1)

        # Log likelihood (without constant)
        log_lik = -0.5 * (log_det + maha)  # (B, K, M)
        return log_lik

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x, gamma_class):
        """
        Streaming EM update (mini-batch).

        Args:
            x:           (B, D) - CLIP image embeddings (L2-normalized)
            gamma_class: (B, K) - zero-shot class probabilities from CLIP
        """
        x = x.to(self.device).float()
        gamma_class = gamma_class.to(self.device).float()
        B, D = x.shape

        with torch.no_grad():
            # ---- E-step ----
            current_var = self._get_var()
            log_lik = self._log_likelihood(x, self.mu, current_var)  # (B, K, M)

            log_pi = torch.log(self.pi + 1e-10).unsqueeze(0)  # (1, K, M)
            log_joint = log_pi + log_lik  # (B, K, M)

            # Within-class mode responsibilities: r_{b,k,m} = P(m | x_b, k)
            log_r = log_joint - torch.logsumexp(log_joint, dim=2, keepdim=True)
            r = torch.exp(log_r)  # (B, K, M)

            # Joint responsibility: gamma_{b,k,m} = gamma_{b,k} * r_{b,k,m}
            gamma = gamma_class.unsqueeze(2) * r  # (B, K, M)

            # ---- M-step ----
            sum_gamma = gamma.sum(dim=0)  # (K, M)

            c_old = self.c.clone()           # (K, M)
            mu_old = self.mu.clone()         # (K, M, D)
            var_old = self.var.clone()       # (K, M, D)

            c_new = c_old + sum_gamma        # (K, M)

            # gamma_perm: (K, M, B)
            gamma_perm = gamma.permute(1, 2, 0)

            # Weighted sum of x: (K, M, D)
            weighted_x = torch.matmul(gamma_perm, x)

            # Update means
            mu_new = (c_old.unsqueeze(-1) * mu_old + weighted_x) / \
                     (c_new.unsqueeze(-1) + 1e-10)

            # Update diagonal variances
            if self.streaming_update_Sigma:
                # Compute weighted sum of (x - mu_old)^2 efficiently:
                # sum_b gamma_{b,k,m} * (x_b - mu_old_{k,m})^2
                # = sum(gamma * x^2) - 2*mu_old*sum(gamma*x) + sum(gamma)*mu_old^2
                x_sq = x ** 2
                weighted_x_sq = torch.matmul(gamma_perm, x_sq)  # (K, M, D)
                term2 = -2.0 * mu_old * weighted_x    # (K, M, D)
                term3 = sum_gamma.unsqueeze(-1) * (mu_old ** 2)   # (K, M, D)
                weighted_sq_diff = weighted_x_sq + term2 + term3  # (K, M, D)

                var_new = (c_old.unsqueeze(-1) * var_old + weighted_sq_diff) / \
                          (c_new.unsqueeze(-1) + 1e-10)

                # CRITICAL: clamp variance - must stay positive!
                self.var = torch.clamp(var_new, min=1e-8)

            self.mu = mu_new
            self.c = c_new

            # Update mixture weights: pi_{k,m} = c_{k,m} / C_k
            C_k = self.c.sum(dim=1, keepdim=True)  # (K, 1)
            self.pi = self.c / (C_k + 1e-10)

            # Update class-level statistics
            self.class_counts += gamma_class.sum(dim=0)
            self.t += B

    def predict(self, x, source_priors=None):
        """
        Compute class scores compatible with CLIP logits scale.

        This follows the original DOTA predict() style:
        uses log P(x|y=k) and optionally adds log prior.

        Args:
            x:             (B, D)
            source_priors: (K,) optional

        Returns:
            scores: (B, K) - in the same scale as CLIP logits (can be added directly)
        """
        x = x.to(self.device).float()
        current_var = self._get_var()

        with torch.no_grad():
            log_lik = self._log_likelihood(x, self.mu, current_var)  # (B, K, M)
            log_pi = torch.log(self.pi + 1e-10).unsqueeze(0)         # (1, K, M)

            # Log P(x | y=k) = logsumexp_m [log pi_{k,m} + log lik_{k,m}]
            log_class_lik = torch.logsumexp(log_pi + log_lik, dim=2)  # (B, K)

            if source_priors is not None:
                p_est = self.class_counts / (self.class_counts.sum() + 1e-10)
                alpha_t = min(self.alpha_max, self.t / (self.t + 100.0))
                p_k = (1 - alpha_t) * source_priors.to(self.device) + alpha_t * p_est
                log_prior = torch.log(p_k + 1e-10)
                return log_class_lik + log_prior.unsqueeze(0)

            return log_class_lik

    def update(self):
        """
        No-op: diagonal covariance needs no explicit inversion.
        Kept for API compatibility with original DOTA.
        """
        pass


# -----------------------------------------------------------------------
# Usage in test loop (same as original DOTA):
#
# cfg = {
#     'epsilon': 0.001,
#     'sigma': 0.004,   # ← KEY: must be ~1/D for CLIP embeddings
#     'alpha_max': 0.5,
# }
# dota_model = DOTA(cfg, input_shape=512, num_classes=K,
#                   clip_weights=clip_weights, num_modes=4)
#
# # In the loop:
# dota_logits = dota_model.predict(image_features.mean(0).unsqueeze(0))
# dota_weights = torch.clamp(params['rho'] * dota_model.c.mean() /
#                             image_features.size(0), max=params['eta'])
# final_logits = clip_logits + dota_weights * dota_logits
# dota_model.fit(image_features, prob_map)
# dota_model.update()  # no-op, kept for compatibility
# -----------------------------------------------------------------------
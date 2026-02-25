import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class DOTA_mix_adaptive(nn.Module):
    """
    DOTA with Adaptive Gaussian Mixture Model per class (Streaming).
    
    Starts with 1 mode per class (recovering original DOTA behavior) and
    dynamically splits components when within-component variance exceeds
    a threshold. This avoids wasting capacity on modes that never receive
    enough data while allowing complex classes to grow more expressive.
    
    Split criterion: a component (k, m) is split when:
      1. Its max diagonal variance exceeds `split_threshold`
      2. Its effective count c_{k,m} exceeds `min_count_to_split`
      3. The class hasn't reached `max_modes` components yet
    
    Split mechanics:
      - The parent component is replaced by two children
      - Children share the parent's mean, offset along the highest-variance dimension
      - Each child gets half the parent's count and weight
      - Variance is reduced along the split dimension
    
    No confidence weighting — this is the baseline for comparison.
    """

    def __init__(self, cfg, input_shape, num_classes, clip_weights,
                 max_modes=8, split_threshold=None, min_count_to_split=5.0,
                 split_check_interval=50, streaming_update_Sigma=True):
        """
        Args:
            cfg: dict with keys:
                 'epsilon'   - regularization (default 0.001)
                 'sigma'     - initial variance
                 'alpha_max' - max blending weight for prior (default 0.5)
            input_shape:  D (e.g. 512)
            num_classes:  K
            clip_weights: (D, K) - CLIP zero-shot weights
            max_modes:    maximum number of modes per class
            split_threshold:      variance threshold to trigger split.
                                  If None, defaults to 4 * sigma_init
                                  (i.e. 4x the initial per-dimension variance)
            min_count_to_split:   minimum effective count before a component
                                  is eligible for splitting
            split_check_interval: check for splits every N fit() calls
                                  (avoids overhead on every single sample)
            streaming_update_Sigma: whether to update variance (should be True)
        """
        super(DOTA_mix_adaptive, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_modes = max_modes
        self.min_count_to_split = min_count_to_split
        self.split_check_interval = split_check_interval
        self.streaming_update_Sigma = streaming_update_Sigma

        self.epsilon = cfg.get('epsilon', 0.001)

        # Sigma init: 1/D for L2-normalized CLIP embeddings
        sigma_cfg = cfg.get('sigma', 1.0)
        if sigma_cfg >= 0.1:
            self.sigma_init = 1.0 / input_shape
            print(f"[DOTA-Adaptive] Warning: sigma={sigma_cfg} too large. "
                  f"Auto-corrected to 1/D = {self.sigma_init:.5f}")
        else:
            self.sigma_init = sigma_cfg

        # Split threshold: default to 4x initial variance
        if split_threshold is None:
            self.split_threshold = 10.0 * self.sigma_init
        else:
            self.split_threshold = split_threshold

        self.alpha_max = cfg.get('alpha_max', 0.5)

        # --------------------------------------------------------
        # Parse clip_weights to (K, D)
        # --------------------------------------------------------
        clip_mu = clip_weights.T.to(self.device).float()  # (K, D)

        # --------------------------------------------------------
        # Initialize with M=1 per class (lists of tensors per class)
        # Using lists because each class can have a different number
        # of modes after splitting.
        # --------------------------------------------------------
        # Per-class lists:
        #   mu_list[k]:  (M_k, D) tensor
        #   var_list[k]: (M_k, D) tensor
        #   pi_list[k]:  (M_k,) tensor
        #   c_list[k]:   (M_k,) tensor
        self.mu_list = []
        self.var_list = []
        self.pi_list = []
        self.c_list = []
        self.num_modes_per_class = []

        for k in range(num_classes):
            self.mu_list.append(clip_mu[k].unsqueeze(0).clone())          # (1, D)
            self.var_list.append(
                torch.full((1, input_shape), self.sigma_init, device=self.device)
            )                                                              # (1, D)
            self.pi_list.append(torch.ones(1, device=self.device))         # (1,)
            self.c_list.append(torch.full((1,), 1.0, device=self.device))  # (1,)
            self.num_modes_per_class.append(1)

        # Class prior tracking
        self.class_counts = torch.zeros(num_classes, device=self.device)
        self.t = 0
        self.fit_calls = 0

        # --------------------------------------------------------
        # Padded tensor cache (rebuilt when any class splits)
        # These are used for vectorized E-step and predict.
        # Shape: (K, M_max, D) etc., where M_max = current max modes
        # --------------------------------------------------------
        self._rebuild_padded_tensors()

    # ==================================================================
    # Padded tensor management
    # ==================================================================

    def _rebuild_padded_tensors(self):
        """
        Rebuild padded (K, M_max, ...) tensors from per-class lists.
        Called after initialization and after any split.
        
        Components beyond a class's actual mode count are masked with:
          - mu = 0, var = huge (so log-likelihood → -inf)
          - pi = 0 (so they contribute nothing in logsumexp)
          - c = 0
        """
        K = self.num_classes
        D = self.input_shape
        M_max = max(self.num_modes_per_class)
        self.M_max = M_max

        self.mu_pad = torch.zeros(K, M_max, D, device=self.device)
        self.var_pad = torch.full((K, M_max, D), 1e10, device=self.device)  # huge → -inf likelihood
        self.pi_pad = torch.zeros(K, M_max, device=self.device)
        self.c_pad = torch.zeros(K, M_max, device=self.device)
        # Mask: True for valid components
        self.mask = torch.zeros(K, M_max, dtype=torch.bool, device=self.device)

        for k in range(K):
            M_k = self.num_modes_per_class[k]
            self.mu_pad[k, :M_k, :] = self.mu_list[k]
            self.var_pad[k, :M_k, :] = self.var_list[k]
            self.pi_pad[k, :M_k] = self.pi_list[k]
            self.c_pad[k, :M_k] = self.c_list[k]
            self.mask[k, :M_k] = True

    def _write_back_from_padded(self):
        """
        Write padded tensors back to per-class lists.
        Called after fit() updates the padded tensors.
        """
        for k in range(self.num_classes):
            M_k = self.num_modes_per_class[k]
            self.mu_list[k] = self.mu_pad[k, :M_k, :].clone()
            self.var_list[k] = self.var_pad[k, :M_k, :].clone()
            self.pi_list[k] = self.pi_pad[k, :M_k].clone()
            self.c_list[k] = self.c_pad[k, :M_k].clone()

    # ==================================================================
    # Splitting
    # ==================================================================

    def _check_and_split(self):
        """
        Check all components across all classes for split eligibility.
        Returns True if any split occurred (triggers tensor rebuild).
        """
        any_split = False

        for k in range(self.num_classes):
            M_k = self.num_modes_per_class[k]
            if M_k >= self.max_modes:
                continue

            # Check each component
            splits_to_do = []
            for m in range(M_k):
                c_km = self.c_list[k][m].item()
                if c_km < self.min_count_to_split:
                    continue

                var_km = self.var_list[k][m]       # (D,)
                max_var = var_km.max().item()

                if max_var > self.split_threshold:
                    splits_to_do.append(m)

                    # Don't exceed max_modes
                    if M_k + len(splits_to_do) >= self.max_modes:
                        break

            # Execute splits (process in reverse order to keep indices valid)
            for m in reversed(splits_to_do):
                if self.num_modes_per_class[k] >= self.max_modes:
                    break
                self._split_component(k, m)
                any_split = True

        return any_split

    def _split_component(self, k, m):
        """
        Split component m of class k into two children.
        
        Strategy:
          - Find the dimension with highest variance
          - Offset children along that dimension by ±sqrt(var_d)
          - Halve the variance along the split dimension
          - Each child gets half the parent's count and weight
        """
        mu_km = self.mu_list[k][m].clone()         # (D,)
        var_km = self.var_list[k][m].clone()        # (D,)
        c_km = self.c_list[k][m].clone()            # scalar
        pi_km = self.pi_list[k][m].clone()          # scalar

        # Find highest-variance dimension
        split_dim = var_km.argmax().item()
        split_std = torch.sqrt(var_km[split_dim])

        # Child 1: offset in +direction
        mu_child1 = mu_km.clone()
        mu_child1[split_dim] += split_std * 0.5

        # Child 2: offset in -direction
        mu_child2 = mu_km.clone()
        mu_child2[split_dim] -= split_std * 0.5

        # Reduce variance along split dimension for both children
        var_child1 = var_km.clone()
        var_child1[split_dim] *= 0.5
        var_child1 = var_child1.clamp(min=1e-8)

        var_child2 = var_km.clone()
        var_child2[split_dim] *= 0.5
        var_child2 = var_child2.clamp(min=1e-8)

        # Half the counts and weights
        c_child = c_km * 0.5
        pi_child = pi_km * 0.5

        # Replace parent with child1, append child2
        self.mu_list[k][m] = mu_child1
        self.var_list[k][m] = var_child1
        self.c_list[k][m] = c_child
        self.pi_list[k][m] = pi_child

        self.mu_list[k] = torch.cat([self.mu_list[k], mu_child2.unsqueeze(0)], dim=0)
        self.var_list[k] = torch.cat([self.var_list[k], var_child2.unsqueeze(0)], dim=0)
        self.c_list[k] = torch.cat([self.c_list[k], c_child.unsqueeze(0)], dim=0)
        self.pi_list[k] = torch.cat([self.pi_list[k], pi_child.unsqueeze(0)], dim=0)

        self.num_modes_per_class[k] += 1

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _get_var(self):
        """Regularized diagonal variance from padded tensor."""
        return torch.clamp(self.var_pad + self.epsilon, min=1e-8)

    def _log_likelihood(self, x, mu, var):
        """
        Diagonal Gaussian log-likelihood (same as DOTA_mix).
        x: (B, D), mu: (K, M, D), var: (K, M, D)
        Returns: (B, K, M)
        """
        diff = x.unsqueeze(1).unsqueeze(2) - mu.unsqueeze(0)
        var_b = var.unsqueeze(0)
        maha = torch.sum(diff ** 2 / var_b, dim=-1)
        log_det = torch.sum(torch.log(var_b), dim=-1)
        log_lik = -0.5 * (log_det + maha)
        return log_lik

    # ==================================================================
    # Property: per-class counts for run_test_dota compatibility
    # ==================================================================
    @property
    def c(self):
        """Per-class effective sample counts, shape (K,)."""
        return self.c_pad.sum(dim=1)

    # ==================================================================
    # Public API
    # ==================================================================

    def fit(self, x, gamma_class):
        """
        Streaming EM update (mini-batch). No confidence weighting.
        Periodically checks for component splits.

        Args:
            x:           (B, D) - CLIP image embeddings
            gamma_class: (B, K) - zero-shot class probabilities
        """
        x = x.to(self.device).float()
        gamma_class = gamma_class.to(self.device).float()
        B, D = x.shape

        with torch.no_grad():
            # ---- E-step (on padded tensors) ----
            current_var = self._get_var()                          # (K, M_max, D)
            log_lik = self._log_likelihood(x, self.mu_pad, current_var)  # (B, K, M_max)

            # Mask invalid components: set their log_pi to -inf
            log_pi = torch.log(self.pi_pad + 1e-10).unsqueeze(0)  # (1, K, M_max)
            # Set invalid component log_pi to -inf so they get 0 responsibility
            invalid_mask = ~self.mask                              # (K, M_max)
            log_pi_masked = log_pi.clone()
            log_pi_masked[:, invalid_mask] = -float('inf')

            log_joint = log_pi_masked + log_lik                    # (B, K, M_max)

            # Within-class mode responsibilities
            log_r = log_joint - torch.logsumexp(log_joint, dim=2, keepdim=True)
            r = torch.exp(log_r)                                   # (B, K, M_max)
            # Zero out invalid components (safety)
            r[:, invalid_mask] = 0.0

            # Joint responsibility
            gamma = gamma_class.unsqueeze(2) * r                   # (B, K, M_max)

            # ---- M-step ----
            sum_gamma = gamma.sum(dim=0)                           # (K, M_max)

            c_old = self.c_pad.clone()
            mu_old = self.mu_pad.clone()
            var_old = self.var_pad.clone()

            c_new = c_old + sum_gamma

            gamma_perm = gamma.permute(1, 2, 0)                   # (K, M_max, B)
            weighted_x = torch.matmul(gamma_perm, x)              # (K, M_max, D)

            # Update means
            mu_new = (c_old.unsqueeze(-1) * mu_old + weighted_x) / \
                     (c_new.unsqueeze(-1) + 1e-10)

            # Update variances
            if self.streaming_update_Sigma:
                x_sq = x ** 2
                weighted_x_sq = torch.matmul(gamma_perm, x_sq)
                term2 = -2.0 * mu_old * weighted_x
                term3 = sum_gamma.unsqueeze(-1) * (mu_old ** 2)
                weighted_sq_diff = weighted_x_sq + term2 + term3

                var_new = (c_old.unsqueeze(-1) * var_old + weighted_sq_diff) / \
                          (c_new.unsqueeze(-1) + 1e-10)
                var_new = torch.clamp(var_new, min=1e-8)

                # Only update valid components
                self.var_pad[self.mask] = var_new[self.mask]

            self.mu_pad[self.mask] = mu_new[self.mask]
            self.c_pad = c_new
            # Zero invalid counts (shouldn't have any, but safety)
            self.c_pad[~self.mask] = 0.0

            # Update mixture weights
            C_k = self.c_pad.sum(dim=1, keepdim=True)
            self.pi_pad = self.c_pad / (C_k + 1e-10)

            # Update class-level statistics
            self.class_counts += gamma_class.sum(dim=0)
            self.t += B
            self.fit_calls += 1

            # Write back to lists
            self._write_back_from_padded()

            # ---- Periodic split check ----
            if self.fit_calls % self.split_check_interval == 0:
                if self._check_and_split():
                    self._rebuild_padded_tensors()

    def predict(self, x, source_priors=None):
        """
        Compute class scores compatible with CLIP logits scale.

        Args:
            x:             (B, D)
            source_priors: (K,) optional
        Returns:
            scores: (B, K)
        """
        x = x.to(self.device).float()
        current_var = self._get_var()

        with torch.no_grad():
            log_lik = self._log_likelihood(x, self.mu_pad, current_var)

            log_pi = torch.log(self.pi_pad + 1e-10).unsqueeze(0)
            # Mask invalid
            invalid_mask = ~self.mask
            log_pi_masked = log_pi.clone()
            log_pi_masked[:, invalid_mask] = -float('inf')

            log_class_lik = torch.logsumexp(log_pi_masked + log_lik, dim=2)

            if source_priors is not None:
                p_est = self.class_counts / (self.class_counts.sum() + 1e-10)
                alpha_t = min(self.alpha_max, self.t / (self.t + 100.0))
                p_k = (1 - alpha_t) * source_priors.to(self.device) + alpha_t * p_est
                log_prior = torch.log(p_k + 1e-10)
                return log_class_lik + log_prior.unsqueeze(0)

            return log_class_lik

    def update(self):
        """No-op for API compatibility."""
        pass

    def get_mode_stats(self):
        """
        Returns a summary of current mode counts per class.
        Useful for logging / debugging.
        
        Returns:
            dict with 'per_class' (list of ints), 'total', 'min', 'max', 'mean'
        """
        counts = self.num_modes_per_class
        return {
            'per_class': list(counts),
            'total': sum(counts),
            'min': min(counts),
            'max': max(counts),
            'mean': sum(counts) / len(counts),
        }
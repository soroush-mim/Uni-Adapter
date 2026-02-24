import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMDOTA(nn.Module):
    """
    Gaussian Mixture Model extension of DOTA for test-time adaptation.

    Each class k has M Gaussian components with diagonal covariance.
    Parameters are updated in a streaming EM fashion using zero-shot
    probabilities as soft class labels.

    Key design choices:
      1. Covariance update uses OLD mu (not the freshly updated one).
      2. Regularization applied only at prediction time (in update()).
      3. Predict returns discriminant scores compatible with CLIP logit fusion.
      4. Sigma initialized to 1/D to match L2-normalized CLIP embedding variance.
      5. Variance clamped after every fit() to prevent numerical issues.
      6. Class priors estimated online and blended with uniform prior.
      7. Fit is fully vectorized over the batch.
    """

    def __init__(self, cfg, input_shape, num_classes, clip_weights,
                 M=4, perturbation_scale=0.01):
        super(GMMDOTA, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape   # D
        self.num_classes = num_classes   # K
        self.M = M                       # mixture components per class
        self.epsilon = cfg['epsilon']    # regularization for covariance

        # ---- Sigma init: 1/D matches L2-normalized CLIP embeddings ----
        # For L2-normed vectors in R^D, per-dimension variance ~ 1/D.
        # Auto-correct if user passes a value that's clearly too large.
        sigma_cfg = cfg.get('sigma', 1.0)
        if sigma_cfg >= 0.1:
            self.sigma_init = 1.0 / input_shape
        else:
            self.sigma_init = sigma_cfg

        # Class prior blending: alpha ramps from 0 (uniform) to alpha_max
        # as total_samples grows, controlling how much we trust the
        # empirical class frequency estimate vs the uniform prior.
        self.alpha_max = cfg.get('alpha_max', 0.6)
        self.perturbation_scale = perturbation_scale

        # ---- Parse clip_weights to (K, D) ----
        clip_weights = clip_weights.to(self.device)
        if clip_weights.shape[0] == input_shape and clip_weights.shape[1] == num_classes:
            base_means = clip_weights.T.contiguous()          # (K, D)
        elif clip_weights.shape[0] == num_classes and clip_weights.shape[1] == input_shape:
            base_means = clip_weights.contiguous()            # (K, D)
        else:
            raise ValueError(
                f"clip_weights shape {clip_weights.shape} incompatible with "
                f"input_shape={input_shape}, num_classes={num_classes}"
            )

        # ---- Initialize mu (K, M, D): clip mean + orthonormal perturbations ----
        self.mu = torch.zeros(num_classes, M, input_shape, device=self.device)
        for k in range(num_classes):
            base = base_means[k]                               # (D,)
            if M > 1 and input_shape >= M:
                random_vecs = torch.randn(input_shape, M, device=self.device)
                Q, _ = torch.linalg.qr(random_vecs)           # Q: (D, M)
                ortho = Q.T                                     # (M, D)
            else:
                ortho = F.normalize(
                    torch.randn(M, input_shape, device=self.device), p=2, dim=-1
                )
            self.mu[k] = base.unsqueeze(0) + perturbation_scale * ortho

        # ---- Initialize Sigma (K, M, D): diagonal = 1/D ----
        self.Sigma = torch.full(
            (num_classes, M, input_shape), self.sigma_init, device=self.device
        )

        # ---- Regularized copy (used by predict; set by update()) ----
        self.Sigma_reg = self.Sigma.clone()

        # ---- Initialize pi (K, M) = 1/M ----
        self.pi = torch.full((num_classes, M), 1.0 / M, device=self.device)

        # ---- Initialize C (K, M) = 1/(K*M) ----
        self.C = torch.full((num_classes, M),
                            1.0 / (num_classes * M), device=self.device)

        # ---- Class prior tracking ----
        # class_counts: running sum of zero-shot probabilities per class
        # total_samples: number of samples seen so far
        self.class_counts = torch.zeros(num_classes, device=self.device)
        self.total_samples = 0

    # ------------------------------------------------------------------
    # Property: per-class counts for compatibility with run_test_dota
    # ------------------------------------------------------------------
    @property
    def c(self):
        """Per-class effective sample counts, shape (K,)."""
        return self.C.sum(dim=1)

    # ------------------------------------------------------------------
    # Internal: log-Gaussian with diagonal covariance
    # ------------------------------------------------------------------
    @staticmethod
    def _log_gauss_diag(x, mu, sigma_diag):
        """
        Log N(x | mu, diag(sigma_diag)), dropping the constant D*log(2pi)/2
        since it cancels across classes in softmax/logsumexp.

        Args
            x:          (..., D)
            mu:         (..., D)  broadcastable with x
            sigma_diag: (..., D)  broadcastable with x  (positive)
        Returns
            (...,)  log-density (up to additive constant)
        """
        sigma_safe = sigma_diag.clamp(min=1e-8)
        diff = x - mu
        mahal = (diff * diff / sigma_safe).sum(dim=-1)
        log_det = torch.log(sigma_safe).sum(dim=-1)
        return -0.5 * (mahal + log_det)

    # ------------------------------------------------------------------
    # Fit: vectorised batch EM update
    # ------------------------------------------------------------------
    def fit(self, x_batch, y_zs_prob_batch):
        """
        Streaming M-step update, vectorised over the batch.

        Args
            x_batch:          (B, D)   feature embeddings (L2-normalized)
            y_zs_prob_batch:  (B, K)   zero-shot class probabilities
        """
        with torch.no_grad():
            x_batch = x_batch.to(self.device).float()
            y_zs_prob_batch = y_zs_prob_batch.to(self.device).float()
            B = x_batch.shape[0]

            # ---------- E-step: within-class responsibilities ----------
            # log-likelihoods (B, K, M)
            log_l = self._log_gauss_diag(
                x_batch[:, None, None, :],          # (B, 1, 1, D)
                self.mu[None, :, :, :],              # (1, K, M, D)
                self.Sigma[None, :, :, :],           # (1, K, M, D)
            )

            # Mode responsibilities r (B, K, M) — softmax over M dim
            log_pi = torch.log(self.pi.clamp(min=1e-10))     # (K, M)
            r = torch.softmax(log_pi[None, :, :] + log_l, dim=2)

            # Joint responsibility gamma_{b,k,m} = P_zs(k|x_b) * r_{b,k,m}
            gamma = y_zs_prob_batch[:, :, None] * r           # (B, K, M)

            # ---------- M-step (batch, vectorized) ----------
            sum_gamma = gamma.sum(dim=0)                      # (K, M)
            old_C = self.C.clone()
            new_C = old_C + sum_gamma

            # Save old mu for covariance update (critical: use OLD mu)
            mu_old = self.mu.clone()

            # Weighted feature sum: sum_b gamma_{b,k,m} * x_b -> (K, M, D)
            weighted_x = torch.einsum('bkm,bd->kmd', gamma, x_batch)

            # Update mu
            self.mu = (old_C[:, :, None] * mu_old + weighted_x) \
                      / new_C[:, :, None].clamp(min=1e-10)

            # Update Sigma using OLD mu
            diff = x_batch[:, None, None, :] - mu_old[None, :, :, :]  # (B,K,M,D)
            weighted_diff_sq = torch.einsum('bkm,bkmd->kmd', gamma, diff * diff)
            self.Sigma = (old_C[:, :, None] * self.Sigma + weighted_diff_sq) \
                         / new_C[:, :, None].clamp(min=1e-10)

            # Clamp variance after every fit to prevent going negative/tiny
            self.Sigma = self.Sigma.clamp(min=1e-8)

            # Update counts
            self.C = new_C

            # Update mixture weights pi_{k,m} = C_{k,m} / C_k
            C_per_class = self.C.sum(dim=1, keepdim=True)    # (K, 1)
            self.pi = self.C / C_per_class.clamp(min=1e-10)

            # Update class prior tracking
            self.class_counts += y_zs_prob_batch.sum(dim=0)   # (K,)
            self.total_samples += B

    # ------------------------------------------------------------------
    # Update: apply regularization for the next predict call
    # ------------------------------------------------------------------
    def update(self):
        """
        Regularise covariance for prediction (shrinkage toward identity).
        Called after fit(), mirrors original DOTA's update() role.
        """
        self.Sigma_reg = (1 - self.epsilon) * self.Sigma \
                         + self.epsilon * torch.ones_like(self.Sigma)
        self.Sigma_reg = self.Sigma_reg.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Predict: GMM discriminant scores compatible with CLIP logit fusion
    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Compute per-class GMM discriminant scores with estimated class priors.

        score_k(x) = logsumexp_m[log pi_{k,m} + f_{k,m}(x)] + log p_k

        where f_{k,m} is the log-Gaussian density (up to constant) and p_k
        is the class prior blended between uniform and empirical estimate.

        Args
            X: (B, D)
        Returns
            (B, K) scores — higher means more likely class k
        """
        with torch.no_grad():
            X = X.to(self.device).float()

            # f_{k,m}(x) -> (B, K, M)
            f_km = self._log_gauss_diag(
                X[:, None, None, :],                # (B, 1, 1, D)
                self.mu[None, :, :, :],              # (1, K, M, D)
                self.Sigma_reg[None, :, :, :],       # (1, K, M, D)
            )

            log_pi = torch.log(self.pi.clamp(min=1e-10))     # (K, M)

            # logsumexp over M -> (B, K): log P(x | y=k)
            log_class_lik = torch.logsumexp(log_pi[None, :, :] + f_km, dim=-1)

            # ---- Estimated class prior blended with uniform ----
            uniform_prior = torch.full(
                (self.num_classes,), 1.0 / self.num_classes, device=self.device
            )

            if self.total_samples > 0:
                estimated_prior = self.class_counts \
                                  / self.class_counts.sum().clamp(min=1e-10)
                # alpha_t ramps from 0 -> alpha_max as samples grow.
                # At total_samples=100: alpha ~ alpha_max/2
                # At total_samples>>100: alpha -> alpha_max
                alpha_t = min(self.alpha_max,
                              self.total_samples / (self.total_samples + 100.0))
                p_k = (1 - alpha_t) * uniform_prior + alpha_t * estimated_prior
            else:
                p_k = uniform_prior

            log_prior = torch.log(p_k.clamp(min=1e-10))      # (K,)

            # Final score: log P(x|y=k) + log p(y=k)
            scores = log_class_lik + log_prior[None, :]       # (B, K)
            return scores
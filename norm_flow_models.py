import math

import torch
import torch.nn as nn

from config import SCENARIO_METHOD


class ConditionalAffineFlow(nn.Module):
    """
    Conditional single-step affine normalizing flow.

    Models  p(y | context) = N( mu(context),  L(context) @ L(context)^T )

    where L is a lower-triangular Cholesky factor whose entries are predicted
    by a context-encoder MLP.  This is the simplest possible normalizing flow:
    a single affine transform with a fully learned conditional covariance over
    the whole forecast horizon.

    Architecture is explicitly prepared for a future Bayesian (weight-sampling)
    variant.  Pass use_bayes=True once SCENARIO_METHOD='full_norm_flow_bayes'
    is implemented; it currently raises NotImplementedError to signal this
    clearly.

    Inputs
    ------
    context_dim  : number of context features fed to the encoder
    horizon_steps: length H of the joint prediction horizon
    hidden_dims  : hidden layer sizes for the encoder MLP
    use_bayes    : if True, raises NotImplementedError (future work)
    """

    def __init__(self, context_dim, horizon_steps, hidden_dims=None, use_bayes=False, l_diag_reg=0.05):
        super().__init__()

        # Regularization weight on log(diag(L))^2 — penalises diagonal scale
        # deviating from 1 in normalised space, preventing the model from
        # collapsing to mu≈0 with huge variance (variance-collapse / free-energy hack).
        self.l_diag_reg = l_diag_reg

        if use_bayes:
            raise NotImplementedError(
                "Bayesian ConditionalAffineFlow (SCENARIO_METHOD='full_norm_flow_bayes') "
                "is not yet implemented.  Set SCENARIO_METHOD='full_norm_flow_fixed'."
            )

        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.horizon_steps = horizon_steps
        self.n_tril = horizon_steps * (horizon_steps + 1) // 2

        # ── Context encoder MLP ───────────────────────────────────────────────
        enc_layers = []
        prev_dim = context_dim
        for h_dim in hidden_dims:
            enc_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        # ── Output heads ─────────────────────────────────────────────────────
        self.mu_head = nn.Linear(prev_dim, horizon_steps)
        # L_head outputs all lower-triangular entries; diagonal gets softplus later
        self.L_head = nn.Linear(prev_dim, self.n_tril)

        # ── Precompute lower-triangular index arrays (stored as buffers) ──────
        row_idx, col_idx = torch.tril_indices(horizon_steps, horizon_steps)
        self.register_buffer("tril_row", row_idx)           # (n_tril,)
        self.register_buffer("tril_col", col_idx)           # (n_tril,)
        self.register_buffer("diag_mask", row_idx == col_idx)   # (n_tril,) bool

    # ─────────────────────────────────────────────────────────────────────────
    def _get_mu_L(self, context):
        """
        Returns mu (batch, H) and lower-triangular L (batch, H, H).
        Diagonal of L is constrained positive via softplus + eps.
        """
        h = self.encoder(context)
        mu = self.mu_head(h)                        # (batch, H)
        L_entries = self.L_head(h)                  # (batch, n_tril)

        batch_size = context.shape[0]
        L = torch.zeros(
            batch_size, self.horizon_steps, self.horizon_steps,
            device=context.device, dtype=context.dtype,
        )

        # Diagonal: positive via softplus, clamped to (0, 5] to prevent runaway scale.
        # Off-diagonal: clamped to [-5, 5] for the same reason.
        L_entries_adj = L_entries.clone()
        L_entries_adj[:, self.diag_mask] = (
            torch.nn.functional.softplus(L_entries[:, self.diag_mask]).clamp(max=5.0) + 1e-4
        )
        L_entries_adj[:, ~self.diag_mask] = L_entries[:, ~self.diag_mask].clamp(-5.0, 5.0)
        L[:, self.tril_row, self.tril_col] = L_entries_adj
        return mu, L

    # ─────────────────────────────────────────────────────────────────────────
    def nll_loss(self, context, y):
        """
        Gaussian NLL: -log p(y | context) under N(mu, L L^T).

        Uses the Cholesky form for efficiency:
            z = L^{-1} (y - mu)     (solve lower-triangular system)
            -log p = 0.5*||z||^2 - sum(log diag(L)) + const
        """
        mu, L = self._get_mu_L(context)
        residual = (y - mu).unsqueeze(-1)           # (batch, H, 1)

        # Solve  L z = residual  (L lower-triangular)
        z = torch.linalg.solve_triangular(          # (batch, H, 1)
            L, residual, upper=False
        ).squeeze(-1)                               # (batch, H)

        log_diag = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))   # (batch, H)
        log_det = log_diag.sum(-1)                                    # (batch,)

        nll = (
            0.5 * (z ** 2).sum(-1)
            - log_det
            + 0.5 * self.horizon_steps * math.log(2 * math.pi)
        )

        # Regularisation: penalise log(L_ii) deviating from 0, i.e. diagonal
        # deviating from 1.  Equivalent to a N(0,1/sqrt(l_diag_reg)) prior on
        # log-scale.  This prevents the variance-collapse where the model drives
        # mu→0 and L_ii→∞ to minimise NLL without fitting the data.
        if self.l_diag_reg > 0:
            nll = nll + self.l_diag_reg * (log_diag ** 2).sum(-1)

        return nll.mean()

    # ─────────────────────────────────────────────────────────────────────────
    def sample(self, context, n_samples=1):
        """
        Draw n_samples jointly from  p(y | context).

        Parameters
        ----------
        context  : Tensor (context_dim,) or (batch, context_dim)
        n_samples: number of independent samples to draw

        Returns
        -------
        numpy array  (n_samples, H)         if input was 1-D or batch == 1
                     (n_samples, batch, H)  otherwise
        """
        squeeze_batch = False
        if context.dim() == 1:
            context = context.unsqueeze(0)
            squeeze_batch = True

        with torch.no_grad():
            mu, L = self._get_mu_L(context)         # (batch, H), (batch, H, H)
            batch_size = context.shape[0]

            z = torch.randn(
                n_samples, batch_size, self.horizon_steps,
                device=context.device,
            )                                       # (n_samples, batch, H)

            # y = mu + L @ z  (broadcast over samples)
            L_exp = L.unsqueeze(0).expand(n_samples, -1, -1, -1)   # (n_s, batch, H, H)
            samples = (
                mu.unsqueeze(0)
                + torch.matmul(L_exp, z.unsqueeze(-1)).squeeze(-1)  # (n_s, batch, H)
            )

            if squeeze_batch or batch_size == 1:
                samples = samples.squeeze(1)        # (n_samples, H)

        return samples.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
def make_flow(context_dim, horizon_steps, hidden_dims=None):
    """
    Instantiate a ConditionalAffineFlow according to SCENARIO_METHOD.

    'full_norm_flow_fixed'  → fixed-weight flow  (implemented)
    'full_norm_flow_bayes'  → Bayesian flow       (raises NotImplementedError)
    'cholesky_decomp_corr'  → should not be called; returns fixed flow as fallback
    """
    use_bayes = SCENARIO_METHOD == "full_norm_flow_bayes"
    return ConditionalAffineFlow(context_dim, horizon_steps, hidden_dims, use_bayes=use_bayes)

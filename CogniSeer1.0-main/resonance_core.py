import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonanceConfig:
    def __init__(
        self,
        grid_h=16,
        grid_w=16,
        dt=0.02,
        gamma_base=0.18,
        gamma_amp=0.08,
        alpha=0.10,
        beta=0.08,
        control_gain=0.35,
        syntropy_target=0.15,
        stability_threshold=0.55,
        global_feedback=0.01,
        chaos_reduction_enabled=True,
        sigma_nominal=10.0,
        r_nominal=28.0,
        beta_nominal=8.0 / 3.0,
        damping=5.0,
        modulation=0.08,
        tolerance=0.05,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dt = dt
        self.gamma_base = gamma_base
        self.gamma_amp = gamma_amp
        self.alpha = alpha
        self.beta = beta
        self.control_gain = control_gain
        self.syntropy_target = syntropy_target
        self.stability_threshold = stability_threshold
        self.global_feedback = global_feedback
        self.chaos_reduction_enabled = chaos_reduction_enabled
        self.sigma_nominal = sigma_nominal
        self.r_nominal = r_nominal
        self.beta_nominal = beta_nominal
        self.damping = damping
        self.modulation = modulation
        self.tolerance = tolerance


class ChaosReductionController(nn.Module):
    """
    Lorenz-style surrogate used to estimate local chaos and provide a
    damping gain for field stabilization.
    """

    def __init__(self, cfg: ResonanceConfig):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("xyz", torch.zeros(1, 3))
        self.register_buffer("t", torch.zeros(1))

        # Small deterministic parameter spread emulates tolerance band.
        self.register_buffer(
            "param_scale",
            torch.tensor([1.0 - cfg.tolerance, 1.0, 1.0 + cfg.tolerance]),
        )

    def reset(self, batch_size, device):
        self.xyz = torch.ones(batch_size, 3, device=device)
        self.t = torch.zeros(1, device=device)

    def _ensure_shape(self, batch_size, device):
        if self.xyz.shape[0] != batch_size or self.xyz.device != device:
            self.reset(batch_size=batch_size, device=device)

    def step(self, observed_xyz, dt):
        """
        observed_xyz: [B, 3], compact field summary used as noisy observation.
        Returns a dict with damping gain and a Lyapunov-like proxy.
        """
        batch_size = observed_xyz.shape[0]
        self._ensure_shape(batch_size=batch_size, device=observed_xyz.device)

        t_next = self.t + dt
        modulated_damping = self.cfg.damping * (
            1.0 + self.cfg.modulation * torch.sin(2.0 * torch.pi * t_next)
        )

        # Broadcast Lorenz parameters with fixed tolerance scaling.
        sigma = self.cfg.sigma_nominal * self.param_scale[0]
        r = self.cfg.r_nominal * self.param_scale[1]
        beta = self.cfg.beta_nominal * self.param_scale[2]

        x, y, z = self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]

        dx = sigma * (y - x) - modulated_damping * x
        dy = r * x - y - x * z - modulated_damping * y
        dz = x * y - beta * z - modulated_damping * z
        deriv = torch.stack([dx, dy, dz], dim=-1)

        # Pull state toward observed field summaries to mimic sensor injection.
        self.xyz = (self.xyz + dt * deriv + 0.1 * observed_xyz).detach()
        self.t = t_next.detach()

        # Baseline dynamics for reduction estimate (no damping, no modulation).
        bdx = sigma * (y - x)
        bdy = r * x - y - x * z
        bdz = x * y - beta * z
        baseline_norm = torch.sqrt(bdx * bdx + bdy * bdy + bdz * bdz + 1e-6)
        damped_norm = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)

        chaos_reduction = ((baseline_norm - damped_norm) / baseline_norm) * 100.0

        # Local Lyapunov proxy from finite-step Jacobian growth surrogate.
        lyapunov_proxy = torch.log1p(torch.norm(deriv, dim=-1))
        damping_gain = torch.clamp(1.0 + 0.01 * F.relu(chaos_reduction), 1.0, 2.0)

        return {
            "chaos_reduction": chaos_reduction,
            "lyapunov_proxy": lyapunov_proxy,
            "damping_gain": damping_gain,
        }


class ResonanceField(nn.Module):
    """
    Practical version of the attached 'resonance/emergence field':
    - 2D field state
    - amplitude-dependent diffusion
    - stability metric from variance
    - simple linear chaos-reduction feedback
    - returns compact modulation signals for CogniSeer
    """

    def __init__(self, cfg: ResonanceConfig):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("phi", torch.zeros(1, 1, cfg.grid_h, cfg.grid_w))
        self.register_buffer("stability", torch.ones(1))
        self.register_buffer("field_mean", torch.zeros(1))
        self.register_buffer("variance", torch.zeros(1))

        # learned/lightly tunable scalars
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha))
        self.beta = nn.Parameter(torch.tensor(cfg.beta))
        self.gamma_base = nn.Parameter(torch.tensor(cfg.gamma_base))
        self.gamma_amp = nn.Parameter(torch.tensor(cfg.gamma_amp))
        self.control_gain = nn.Parameter(torch.tensor(cfg.control_gain))
        self.chaos_controller = ChaosReductionController(cfg)

    def reset_state(self, batch_size=1, device="cpu"):
        self.phi = torch.zeros(batch_size, 1, self.cfg.grid_h, self.cfg.grid_w, device=device)
        self.stability = torch.ones(batch_size, device=device)
        self.field_mean = torch.zeros(batch_size, device=device)
        self.variance = torch.zeros(batch_size, device=device)
        self.chaos_controller.reset(batch_size=batch_size, device=device)

    def laplacian(self, x):
        return (
            torch.roll(x, 1, dims=-2)
            + torch.roll(x, -1, dims=-2)
            + torch.roll(x, 1, dims=-1)
            + torch.roll(x, -1, dims=-1)
            - 4.0 * x
        )

    def forward(self, sensory_map):
        """
        sensory_map: [B, 1, H, W]
        """
        phi = self.phi
        if phi.shape[0] != sensory_map.shape[0]:
            phi = phi.repeat(sensory_map.shape[0], 1, 1, 1)

        gamma_eff = self.gamma_base + self.gamma_amp * (phi ** 2)
        lap = self.laplacian(phi)

        spatial_mean = phi.mean(dim=(2, 3), keepdim=True)
        global_feedback = -self.cfg.global_feedback * spatial_mean

        dphi = (
            gamma_eff * lap
            + self.alpha * phi
            - self.beta * (phi ** 3)
            + global_feedback
            + 0.15 * sensory_map
        )

        phi_next = torch.clamp(phi + self.cfg.dt * dphi, -3.0, 3.0)

        variance = phi_next.var(dim=(1, 2, 3), unbiased=False)
        field_mean = phi_next.mean(dim=(1, 2, 3))
        stability = torch.clamp(1.0 / (1.0 + 10.0 * variance), 0.0, 1.0)

        chaos_reduction = torch.zeros_like(stability)
        lyapunov_proxy = torch.zeros_like(stability)
        damping_gain = torch.ones_like(stability)
        if self.cfg.chaos_reduction_enabled:
            observed_xyz = torch.stack(
                [
                    field_mean,
                    variance,
                    self.cfg.syntropy_target - field_mean,
                ],
                dim=-1,
            )
            chaos = self.chaos_controller.step(observed_xyz=observed_xyz, dt=self.cfg.dt)
            chaos_reduction = chaos["chaos_reduction"]
            lyapunov_proxy = chaos["lyapunov_proxy"]
            damping_gain = chaos["damping_gain"]

        # linear feedback control when unstable
        unstable = stability < self.cfg.stability_threshold
        if unstable.any():
            effective_gain = self.control_gain * damping_gain.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            control = -effective_gain[unstable] * (
                phi_next[unstable]
                - self.cfg.syntropy_target
            )
            phi_next[unstable] = torch.clamp(
                phi_next[unstable] + self.cfg.dt * control,
                -3.0,
                3.0,
            )
            variance = phi_next.var(dim=(1, 2, 3), unbiased=False)
            field_mean = phi_next.mean(dim=(1, 2, 3))
            stability = torch.clamp(1.0 / (1.0 + 10.0 * variance), 0.0, 1.0)

        self.phi = phi_next.detach()
        self.variance = variance.detach()
        self.field_mean = field_mean.detach()
        self.stability = stability.detach()

        # compact outputs for the main brain
        return {
            "field_state": self.phi,
            "field_mean": self.field_mean,
            "variance": self.variance,
            "stability": self.stability,
            "syntropy_error": (self.cfg.syntropy_target - self.field_mean),
            "chaos_reduction": chaos_reduction.detach(),
            "lyapunov_proxy": lyapunov_proxy.detach(),
            "damping_gain": damping_gain.detach(),
        }

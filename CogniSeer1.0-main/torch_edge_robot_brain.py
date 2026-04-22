import torch
import torch.nn as nn
import torch.nn.functional as F

from resonance_core import ResonanceConfig, ResonanceField


class TorchBrainConfig:
    def __init__(
        self,
        grid_h=16,
        grid_w=16,
        input_dim=32,
        hidden_dim=64,
        motor_dim=8,
        dt=0.02,
        use_field_enhanced_snn=False,
        chaos_reduction_enabled=True,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.motor_dim = motor_dim
        self.dt = dt
        self.use_field_enhanced_snn = use_field_enhanced_snn
        self.chaos_reduction_enabled = chaos_reduction_enabled


class TorchSplitIFTFieldCore(nn.Module):
    FAST_IDX = [0, 1, 2, 3, 4]
    SLOW_IDX = [5, 6, 7, 8, 9]

    def __init__(self, cfg: TorchBrainConfig):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("phi", torch.zeros(1, 10, cfg.grid_h, cfg.grid_w))
        self.gamma = nn.Parameter(torch.tensor([0.20, 0.15, 0.10, 0.18, 0.12, 0.13, 0.14, 0.16, 0.11, 0.10]))
        self.alpha = nn.Parameter(torch.tensor([0.30, 0.25, 0.22, 0.18, 0.20, 0.15, 0.19, 0.24, 0.17, 0.16]))

        self.fast_dt_scale = 1.0
        self.slow_dt_scale = 0.20
        self.coupling_eps_fast = 0.06
        self.coupling_eps_slow = 0.02
        self.fast_to_slow_gain = 0.04
        self.slow_to_fast_gain = 0.02

        self.mix = nn.Conv2d(10, 10, kernel_size=1, bias=False)

    def laplacian(self, x):
        return (
            torch.roll(x, 1, dims=-2)
            + torch.roll(x, -1, dims=-2)
            + torch.roll(x, 1, dims=-1)
            + torch.roll(x, -1, dims=-1)
            - 4.0 * x
        )

    def biharmonic(self, x):
        return self.laplacian(self.laplacian(x))

    def local_dynamics(self, phi):
        lap = self.laplacian(phi)
        bilap = self.biharmonic(phi)

        out = torch.zeros_like(phi)

        out[:, 0] = -self.gamma[0] * lap[:, 0] - self.alpha[0] * (phi[:, 0] ** 3) + 0.12 * bilap[:, 0]
        out[:, 1] = -self.gamma[1] * lap[:, 1] + 0.15 * torch.sin(phi[:, 1]) + 0.08 * (phi[:, 1] ** 3)
        out[:, 2] = -self.gamma[2] * lap[:, 2] + 0.20 * phi[:, 2] * (1.0 - phi[:, 2] ** 2)
        out[:, 3] = -self.gamma[3] * lap[:, 3] - 0.03 * (phi[:, 3] ** 5) + 0.08 * bilap[:, 3]
        out[:, 4] = -self.gamma[4] * lap[:, 4] + 0.12 * phi[:, 4] * torch.log(torch.abs(phi[:, 4]) + 1e-6) + 0.02 * bilap[:, 4]
        out[:, 5] = -self.gamma[5] * lap[:, 5] + 0.08 * (phi[:, 5] ** 3) - 0.02 * (phi[:, 5] ** 5)
        out[:, 6] = -self.gamma[6] * lap[:, 6] + 0.06 * (phi[:, 6] ** 4) - 0.04 * phi[:, 6]
        out[:, 7] = -self.gamma[7] * lap[:, 7] + 0.08 * torch.sinh(torch.clamp(phi[:, 7], -2.0, 2.0))
        out[:, 8] = -self.gamma[8] * lap[:, 8] + 0.05 * (phi[:, 8] ** 2) - 0.08 * phi[:, 8]
        out[:, 9] = -self.gamma[9] * lap[:, 9] + 0.02 * (phi[:, 9] ** 6) - 0.04 * (phi[:, 9] ** 3) + 0.01 * bilap[:, 9]

        return out

    def forward(self, sensory_map):
        phi = self.phi
        dphi = self.local_dynamics(phi)

        coupling = self.mix(phi)

        fast_mean = phi[:, self.FAST_IDX].mean(dim=1, keepdim=True)
        slow_mean = phi[:, self.SLOW_IDX].mean(dim=1, keepdim=True)

        dphi[:, self.FAST_IDX] += self.coupling_eps_fast * coupling[:, self.FAST_IDX]
        dphi[:, self.SLOW_IDX] += self.coupling_eps_slow * coupling[:, self.SLOW_IDX]

        dphi[:, self.FAST_IDX] += self.slow_to_fast_gain * slow_mean
        dphi[:, self.SLOW_IDX] += self.fast_to_slow_gain * fast_mean

        if sensory_map.dim() == 4:
            sensory_drive = sensory_map[:, 0]
        else:
            sensory_drive = sensory_map

        dphi[:, 0] += 0.20 * sensory_drive
        dphi[:, 4] += 0.08 * sensory_drive.abs()
        dphi[:, 9] += 0.05 * sensory_drive

        phi_next = phi.clone()
        phi_next[:, self.FAST_IDX] = phi[:, self.FAST_IDX] + self.cfg.dt * self.fast_dt_scale * dphi[:, self.FAST_IDX]
        phi_next[:, self.SLOW_IDX] = phi[:, self.SLOW_IDX] + self.cfg.dt * self.slow_dt_scale * dphi[:, self.SLOW_IDX]
        phi_next = torch.clamp(phi_next, -4.0, 4.0)

        self.phi = phi_next.detach()

        summaries = {
            "phi1_mean": phi_next[:, 0].mean(dim=(1, 2)),
            "phi5_mean": phi_next[:, 4].mean(dim=(1, 2)),
            "field_means": phi_next.mean(dim=(2, 3)),
        }
        return phi_next, summaries


class FieldEnhancedSNN(nn.Module):
    """
    Field-modulated SNN with PDE-updated excitability/plasticity fields.
    Implements phi1/phi5/Phi dynamics, then applies field-conditioned LIF.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

        # Field states
        self.register_buffer("phi1", torch.randn(size))
        self.register_buffer("phi5", torch.ones(size) * 0.1)
        self.register_buffer("Phi", torch.zeros(size))
        self.register_buffer("mem_potential", torch.zeros(size))

        # PDE coefficients
        self.alpha = nn.Parameter(torch.tensor(0.30))
        self.zeta = nn.Parameter(torch.tensor(0.08))
        self.beta = nn.Parameter(torch.tensor(0.12))
        self.gamma1 = nn.Parameter(torch.tensor(0.20))
        self.gamma5 = nn.Parameter(torch.tensor(0.12))
        self.omega = nn.Parameter(torch.tensor(0.12))
        self.theta = nn.Parameter(torch.tensor(0.05))

        # LIF-like constants
        self.base_threshold = 0.5
        self.base_leak = 0.90

    def laplacian(self, x):
        return (
            torch.roll(x, 1, dims=-1)
            + torch.roll(x, -1, dims=-1)
            - 2.0 * x
        )

    def pde_step(self, dt=0.01):
        # 1) Spatial derivatives
        lap_phi1 = self.laplacian(self.phi1)
        bi_lap_phi1 = self.laplacian(lap_phi1)
        lap_phi5 = self.laplacian(self.phi5)

        # 2) Pattern field (phi1)
        reaction = -self.alpha * (self.phi1 ** 3) + self.zeta * self.laplacian(self.Phi)
        diffusion = -self.gamma1 * lap_phi1 + self.beta * bi_lap_phi1
        d_phi1 = diffusion + reaction
        self.phi1 = (self.phi1 + d_phi1 * dt).detach()

        # 3) Memory-preserving field (phi5)
        log_term = self.omega * self.phi5 * torch.log(torch.abs(self.phi5) + 1e-6)
        advection = self.theta * (lap_phi5 * self.Phi + self.phi5 * self.laplacian(self.Phi))
        d_phi5 = -self.gamma5 * lap_phi5 + log_term + advection
        self.phi5 = (self.phi5 + d_phi5 * dt).detach()

    def integrate(self, drive, leak):
        if self.mem_potential.shape[0] != drive.shape[-1]:
            self.mem_potential = torch.zeros(drive.shape[-1], device=drive.device)

        mem = self.mem_potential.unsqueeze(0).expand_as(drive)
        next_mem = leak * mem + drive
        self.mem_potential = next_mem.mean(dim=0).detach()
        return next_mem

    def reset_state(self, device="cpu"):
        self.phi1 = torch.randn(self.size, device=device)
        self.phi5 = torch.ones(self.size, device=device) * 0.1
        self.Phi = torch.zeros(self.size, device=device)
        self.mem_potential = torch.zeros(self.size, device=device)

    def forward(self, input_spikes):
        # Step A: update PDE fields.
        self.pde_step()

        # Step B: modulate threshold/leak from field state.
        dynamic_threshold = self.base_threshold - (0.5 * self.phi1).unsqueeze(0)
        effective_leak = self.base_leak * (1.0 / (1.0 + self.phi5)).unsqueeze(0)

        # Step C: standard LIF integration driven by input_spikes.
        mem_potential = self.integrate(input_spikes, leak=effective_leak)
        synapse_fire = torch.sigmoid(mem_potential - dynamic_threshold)

        # Step D: spike feedback updates global potential.
        self.Phi = (0.9 * self.Phi + 0.1 * synapse_fire.mean(dim=0).float()).detach()

        return {
            "synapse_fire": synapse_fire,
            "phi1": self.phi1,
            "phi5": self.phi5,
            "Phi": self.Phi,
        }


class TorchEdgeRobotBrain(nn.Module):
    def __init__(self, cfg: TorchBrainConfig):
        super().__init__()
        self.cfg = cfg

        self.perception = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Tanh(),
        )

        self.memory_mix = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.reason1 = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        self.reason2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.action_head = nn.Linear(cfg.hidden_dim, cfg.motor_dim)
        self.sensor_proj = nn.Linear(cfg.input_dim, cfg.grid_h * cfg.grid_w)

        self.spike_ff = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fields = TorchSplitIFTFieldCore(cfg)

        # Optional drop-in replacement for spike integration.
        self.use_field_enhanced_snn = cfg.use_field_enhanced_snn
        self.field_enhanced_snn = FieldEnhancedSNN(cfg.hidden_dim)

        # Resonance sidecar for stability/syntropy modulation.
        self.resonance = ResonanceField(
            ResonanceConfig(
                grid_h=cfg.grid_h,
                grid_w=cfg.grid_w,
                dt=cfg.dt,
                chaos_reduction_enabled=cfg.chaos_reduction_enabled,
            )
        )

        self.register_buffer("memory_state", torch.zeros(1, cfg.hidden_dim))
        self.register_buffer("spike_mem", torch.zeros(1, cfg.hidden_dim))
        self.register_buffer("action_state", torch.zeros(1, cfg.motor_dim))

        self.base_threshold = 0.5
        self.base_leak = 0.90
        self.memory_decay = 0.98
        self.action_decay = 0.90

    def reset_state(self, batch_size=1, device="cpu"):
        self.memory_state = torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
        self.spike_mem = torch.zeros(batch_size, self.cfg.hidden_dim, device=device)
        self.action_state = torch.zeros(batch_size, self.cfg.motor_dim, device=device)
        self.fields.phi = torch.zeros(batch_size, 10, self.cfg.grid_h, self.cfg.grid_w, device=device)
        self.resonance.reset_state(batch_size=batch_size, device=device)
        self.field_enhanced_snn.reset_state(device=device)

    def detach_state(self):
        # Truncated BPTT: keep state values while dropping old graph history.
        self.memory_state = self.memory_state.detach()
        self.spike_mem = self.spike_mem.detach()
        self.action_state = self.action_state.detach()

    def set_chaos_control(self, enabled: bool):
        self.resonance.cfg.chaos_reduction_enabled = bool(enabled)

    def forward(self, obs):
        batch_size = obs.shape[0]

        sensory_map = torch.tanh(self.sensor_proj(obs)).view(
            batch_size, 1, self.cfg.grid_h, self.cfg.grid_w
        )

        # Main CogniSeer field and resonance/stability sidecar.
        _, fs = self.fields(sensory_map)
        rs = self.resonance(sensory_map)

        percept = self.perception(obs)

        stability_gain = rs["stability"].unsqueeze(-1)
        keep = torch.clamp(
            self.memory_decay
            + 0.04 * fs["phi5_mean"].unsqueeze(-1)
            + 0.03 * stability_gain,
            0.80,
            0.995,
        )
        candidate = torch.tanh(self.memory_mix(percept))
        self.memory_state = keep * self.memory_state + (1.0 - keep) * candidate

        reason_in = torch.cat([percept, self.memory_state], dim=-1)
        field_summary = torch.zeros(batch_size, self.cfg.hidden_dim, device=obs.device)
        field_summary[:, :10] = fs["field_means"]
        field_summary[:, 10] = rs["stability"]
        field_summary[:, 11] = rs["field_mean"]
        field_summary[:, 12] = rs["variance"]
        field_summary[:, 13] = rs["syntropy_error"]

        reasoning = torch.relu(self.reason1(reason_in))
        reasoning = reasoning + 0.1 * field_summary
        reasoning = torch.tanh(self.reason2(reasoning))

        phi1_gain = fs["phi1_mean"].unsqueeze(-1).expand(-1, self.cfg.hidden_dim)
        phi5_gain = fs["phi5_mean"].unsqueeze(-1).expand(-1, self.cfg.hidden_dim)

        dyn_threshold = (
            self.base_threshold
            - 0.20 * phi1_gain
            + 0.10 * rs["variance"].unsqueeze(-1)
            - 0.10 * rs["stability"].unsqueeze(-1)
        )

        leak = torch.clamp(
            self.base_leak
            + 0.05 * phi5_gain
            + 0.03 * rs["stability"].unsqueeze(-1),
            0.70,
            0.99,
        )

        relay_signal = None
        if self.use_field_enhanced_snn:
            snn_out = self.field_enhanced_snn(reasoning)
            spikes = snn_out["synapse_fire"]
            relay_signal = snn_out.get("relay_signal")
        else:
            current = self.spike_ff(reasoning)
            self.spike_mem = leak * self.spike_mem + current
            spikes = (self.spike_mem > dyn_threshold).float()
            self.spike_mem = torch.where(spikes > 0, self.spike_mem * 0.25, self.spike_mem)

        logits = self.action_head(reasoning) + 0.2 * spikes[:, :self.cfg.motor_dim]

        confidence = torch.clamp(
            0.35 + 0.65 * rs["stability"].unsqueeze(-1),
            0.20,
            1.00,
        )
        act = torch.tanh(logits) * confidence
        self.action_state = self.action_decay * self.action_state + (1.0 - self.action_decay) * act

        telemetry = {
            "motor_command": self.action_state,
            "phi1_mean": fs["phi1_mean"],
            "phi5_mean": fs["phi5_mean"],
            "spike_rate": spikes.mean(dim=-1),
            "res_stability": rs["stability"],
            "res_field_mean": rs["field_mean"],
            "res_variance": rs["variance"],
            "res_chaos_reduction": rs["chaos_reduction"],
            "res_lyapunov_proxy": rs["lyapunov_proxy"],
            "action_confidence": confidence.mean(dim=-1),
        }
        if relay_signal is not None:
            telemetry["relay_signal_norm"] = relay_signal.norm(dim=-1).mean(dim=-1)
        return telemetry

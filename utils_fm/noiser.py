from dataclasses import dataclass, field
import math
from typing import Protocol, List

import torch
import torch.nn as nn
import numpy as np
from utils.config import BaseParams, ConfigurableModule
from utils.misc import DTYPE_MAP

class NoiserProtocol(Protocol):
    """Protocol defining the interface that a noiser module should implement."""

    def alpha_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class FlowNoiserParams(BaseParams):
    """Parameters for FlowNoiser."""

    compute_dtype: str = "fp32"  # Internal computation dtype: "fp32", "fp16", "bf16"

def match_dims(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Match the dimensions of the tensor to the shape."""
    if tensor.ndim == len(shape):
        return tensor
    while tensor.ndim < len(shape):
        tensor = tensor.unsqueeze(-1)
    return tensor


class FlowNoiser(NoiserProtocol, nn.Module, ConfigurableModule[FlowNoiserParams]):
    def __init__(self, params: FlowNoiserParams) -> None:
        nn.Module.__init__(self)

        # Use the global DTYPE_MAP
        self.compute_dtype = DTYPE_MAP[params.compute_dtype]

    @classmethod
    def get_default_params(cls) -> FlowNoiserParams:
        """Return the default parameters for FlowNoiser."""
        return FlowNoiserParams()

    def alpha_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha and beta for the given timestep.
        t = 0 is clean data, t = 1 is pure noise.
        """
        alpha = 1 - t
        beta = t
        return alpha, beta

    def forward(
        self, x: torch.Tensor, x_datum_lens: torch.Tensor, t: torch.Tensor, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to the input tensor.
        """
        num_tokens = x.shape[0]
        if t.shape[0] != num_tokens:
            t = torch.repeat_interleave(t, x_datum_lens, output_size=num_tokens)
        alpha, beta = self.alpha_beta(t)
        # Reshape for proper broadcasting: (N,) -> (N, 1)
        # extend dims of alpha and beta to match the shape of x
        alpha = match_dims(alpha, x.shape)
        beta = match_dims(beta, x.shape)
        x_float = x.type(self.compute_dtype)
        gauss_noise = torch.randn(x_float.shape, device=x_float.device, dtype=x_float.dtype, generator=rng)
        # print(alpha, beta)
        x_noised = alpha * x_float + beta * gauss_noise
        v = gauss_noise - x_float
        return x_noised.type(x.dtype), v.type(x.dtype)



def linear_schedule(
    timesteps: int,
    beta_0: float = 0.0001,
    beta_1: float = 0.0200,
    zero_terminal: bool = False,
    legacy_mode: bool = False,
) -> dict[str, torch.Tensor]:
    assert 0.0 < beta_0 < beta_1 < 1.0, "beta_0 and beta_1 must be in (0, 1)"
    betas = np.linspace(beta_0, beta_1, timesteps, dtype=np.float64)
    betas = torch.from_numpy(betas)

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()

    """
    Based on:
    https://arxiv.org/abs/2305.08891
    """
    # store old value
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

    # shift and scale so first timestep is one and last timestep is zero
    alphas_cumprod_sqrt = (alphas_cumprod_sqrt - alphas_cumprod_sqrt_T) / (
        alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
    )

    # convert alpha_cumprod_sqrt back to betas
    alphas_cumprod = alphas_cumprod_sqrt.square()
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1.0 - alphas

    return alphas_cumprod  # this returns float64


@dataclass
class FilixNoiserParams(BaseParams):
    """Parameters for FilixNoiser."""

    compute_dtype: str = "fp32"  # Internal computation dtype: "fp32", "fp16", "bf16"
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    num_train_timesteps: int = 1000


class FilixNoiser(NoiserProtocol, nn.Module, ConfigurableModule[FilixNoiserParams]):
    def __init__(self, params: FilixNoiserParams) -> None:
        nn.Module.__init__(self)
        self.compute_dtype = DTYPE_MAP[params.compute_dtype]
        self.beta_start = params.beta_start
        self.beta_end = params.beta_end
        self.num_train_timesteps = params.num_train_timesteps
        alphas_cumprod = linear_schedule(self.num_train_timesteps, self.beta_start, self.beta_end)

        # Calculate st and nt terms
        alpha_sqrt = torch.sqrt(alphas_cumprod)
        alpha_m1_sqrt = torch.sqrt(1 - alphas_cumprod)
        denom = alpha_sqrt + alpha_m1_sqrt

        device = torch.cuda.current_device()
        st = (alpha_sqrt / denom).to(dtype=torch.float32, device=device)
        nt = (alpha_m1_sqrt / denom).to(dtype=torch.float32, device=device)
        denom = denom.to(dtype=torch.float32, device=device)  # used in DDIM SDE formulation
        self.register_buffer("st", st)
        self.register_buffer("nt", nt)
        self.register_buffer("denom", denom)

    @classmethod
    def get_default_params(cls) -> FilixNoiserParams:
        """Return the default parameters for FilixNoiser."""
        return FilixNoiserParams()

    def alpha_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha and beta for the given timestep.
        t = 0 is clean data, t = 1 is pure noise.
        """
        t = t.round().to(dtype=torch.int64, device=t.device).clamp(0, self.num_train_timesteps - 1)
        st = self.st[t.flatten()].view(t.shape)
        nt = self.nt[t.flatten()].view(t.shape)
        denom = self.denom[t.flatten()].view(t.shape)
        return st, nt, denom
    
    def forward(self, x: torch.Tensor, x_datum_lens: torch.Tensor, t: torch.Tensor, rng: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to the input tensor.
        """
        num_tokens = x.shape[0]
        if t.shape[0] != num_tokens:
            t = torch.repeat_interleave(t, x_datum_lens, output_size=num_tokens)
        st, nt, _ = self.alpha_beta(t)
        x_float = x.type(self.compute_dtype)
        gauss_noise = torch.randn(x_float.shape, device=x_float.device, dtype=x_float.dtype, generator=rng)
        x_noised = match_dims(st, x_float.shape) * x_float + match_dims(nt, x_float.shape) * gauss_noise
        v = x_float - gauss_noise
        return x_noised.type(x.dtype), v.type(x.dtype)


@dataclass
class TimeWarperParams(BaseParams):
    """Parameters for TimeWarper."""

    base_len: int = 256
    base_shift: float = 0.5
    max_len: int = 4096
    max_shift: float = 1.15
    shift: float = 3.0
    time_shift_type: str = "exponential"


class TimeWarper(nn.Module, ConfigurableModule[TimeWarperParams]):
    def __init__(self, params: TimeWarperParams) -> None:
        nn.Module.__init__(self)
        self.base_len = params.base_len
        self.base_shift = params.base_shift
        self.max_len = params.max_len
        self.max_shift = params.max_shift
        self.shift = params.shift
        self.time_shift_type = params.time_shift_type

        # Precompute linear function coefficients
        self.slope = (self.max_shift - self.base_shift) / (self.max_len - self.base_len)
        self.intercept = self.base_shift - self.slope * self.base_len

    @classmethod
    def get_default_params(cls) -> TimeWarperParams:
        """Return the default parameters for TimeWarper."""
        return TimeWarperParams()

    def time_shift(self, mu: torch.Tensor, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """Apply time shift transformation using exponential scaling."""
        exp_mu = torch.exp(mu)
        return exp_mu / (exp_mu + (1 / t - 1) ** sigma)

    def linear_time_shift(self, shift: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply time shift transformation using linear scaling."""
        return shift * t / (1 + (shift - 1) * t)
    
    def forward(self, t: torch.Tensor, target_len: int | torch.Tensor) -> torch.Tensor:
        """
        Apply time warping transformation based on target sequence length.

        Args:
            t (torch.Tensor): Time values tensor.
            target_len (int | torch.Tensor): Target sequence length. If int, applies the same length
                to all elements. If torch.Tensor, should have the same shape as t.

        Returns:
            torch.Tensor: Warped time values.
        """
        # Convert int to tensor with same shape as t
        if isinstance(target_len, int):
            target_len = torch.full_like(t, target_len, dtype=torch.int32)

        # Now target_len is always a tensor with the same shape as t
        mu = self.slope * target_len + self.intercept
        return self.time_shift(mu, 1.0, t) if self.time_shift_type == "exponential" else self.linear_time_shift(self.shift, t)



@dataclass
class FilixTimeWarperParams(BaseParams):
    """Parameters for FilixTimeWarper."""

    shift: float = 3.0

class FilixTimeWarper(nn.Module, ConfigurableModule[FilixTimeWarperParams]):
    def __init__(self, params: FilixTimeWarperParams) -> None:
        nn.Module.__init__(self)
        self.shift = params.shift

    @classmethod
    def get_default_params(cls) -> TimeWarperParams:
        """Return the default parameters for TimeWarper."""
        return TimeWarperParams()

    def linear_time_shift(self, shift: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply time shift transformation using linear scaling."""
        return shift * t / (1 + (shift - 1) * t)
    
    def solve_for_t_prime(self, t, st_schedule, num_train_timesteps):
        """
        Solve for t' given t and st_schedule, such that st_schedule[t'] = 1 - t / 1000
        is the linear flow matching schedule.
        """
        y = 1 - t / num_train_timesteps
        if torch.is_tensor(t):
            assert st_schedule.ndim == 1, f"st_schedule must be 1D tensor, got {st_schedule.shape}"
            assert t.ndim == 1, f"t must be 1D tensor, got {t.shape}"
            st_schedule = st_schedule.unsqueeze(0)  # (1, 1000)
            y = y.unsqueeze(1)  # (b, 1)
            diff = torch.abs(st_schedule - y)  # (b, 1000)
            index = torch.argmin(diff, dim=1).reshape_as(t)
        else:
            diff = torch.abs(st_schedule - y)
            index = torch.argmin(diff)
        return index

    def forward(self, t: torch.Tensor, target_len: int | torch.Tensor, num_train_timesteps: int, st_schedule: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping transformation based on target sequence length.

        Args:
            t (torch.Tensor): Time values tensor.
            target_len (int | torch.Tensor): Target sequence length. If int, applies the same length
                to all elements. If torch.Tensor, should have the same shape as t.

        Returns:
            torch.Tensor: Warped time values.
        """
        # Convert int to tensor with same shape as t
        t_norm = t/num_train_timesteps
        t_warped = self.linear_time_shift(self.shift, t_norm) * num_train_timesteps
        t_prime = self.solve_for_t_prime(t_warped, st_schedule, num_train_timesteps)
        return t_prime


def logit_normal_pdf(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    eps = 1e-6
    x = x.clamp(min=eps, max=1 - eps)

    logit_x = torch.log(x / (1 - x))
    log_pdf = -torch.log(x * (1 - x)) - math.log(sigma * math.sqrt(2 * math.pi)) - 0.5 * ((logit_x - mu) / sigma) ** 2
    pdf = torch.exp(log_pdf)
    return pdf


@dataclass
class TimeWeighterParams(BaseParams):
    """Parameters for TimeWeighter."""

    use_logit_normal: bool = True
    mu: float = 0.0
    sigma: float = 1.0
    num_train_timesteps: int | None = None


class TimeWeighter(nn.Module, ConfigurableModule[TimeWeighterParams]):
    def __init__(self, params: TimeWeighterParams) -> None:
        nn.Module.__init__(self)
        self.use_logit_normal = params.use_logit_normal
        self.mu = params.mu
        self.sigma = params.sigma
        self.num_train_timesteps = params.num_train_timesteps

    @classmethod
    def get_default_params(cls) -> TimeWeighterParams:
        """Return the default parameters for TimeWeighter."""
        return TimeWeighterParams()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute weights for time values.

        When use_logit_normal=True, computes weights using the logit normal probability density function
        with parameters mu and sigma. When use_logit_normal=False, returns uniform weights of 1.0.

        Args:
            t (torch.Tensor): Time values tensor of shape [batch_size,]. Values should be in range (0, 1).

        Returns:
            torch.Tensor: Weight values of shape [batch_size,]. When use_logit_normal=True, these are
            probability density values from the logit normal distribution. When use_logit_normal=False,
            these are uniform weights of 1.0.
        """
        if self.num_train_timesteps is not None:
            t = t/self.num_train_timesteps
        if self.use_logit_normal:
            return logit_normal_pdf(t, mu=self.mu, sigma=self.sigma)
        else:
            return torch.ones_like(t)


def logit_normal_sample(
    size: tuple[int, ...],
    mu: float = 0.0,
    sigma: float = 1.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    min_timestep: float = 0.0,
    max_timestep: float = 1.0,
) -> torch.Tensor:
    """
    Sample from logit normal distribution.

    Args:
        size (tuple[int, ...]): Size of the output tensor as a tuple of integers.
        mu (float): Mean parameter of the underlying normal distribution.
        sigma (float): Standard deviation parameter of the underlying normal distribution.
        device (torch.device, optional): Device to place the tensor on.
        generator (torch.Generator, optional): Random number generator.

    Returns:
        torch.Tensor: Samples from logit normal distribution in range (0, 1).
    """
    normal_samples = torch.randn(size, device=device, dtype=torch.float32, generator=generator)
    logit_normal_samples = torch.sigmoid(normal_samples * sigma + mu)

    return logit_normal_samples * (max_timestep - min_timestep) + min_timestep


@dataclass
class TimeSamplerParams(BaseParams):
    """Parameters for TimeSampler."""

    use_logit_normal: bool = True
    mu: float = 0.0
    sigma: float = 1.0
    dmd_time_steps: list[float] | None = None
    num_train_timesteps: int | None = None
    min_timestep: float = 0.0
    max_timestep: float = 1.0



class TimeSampler(nn.Module, ConfigurableModule[TimeSamplerParams]):
    def __init__(self, params: TimeSamplerParams) -> None:
        nn.Module.__init__(self)
        self.use_logit_normal = params.use_logit_normal
        self.mu = params.mu
        self.sigma = params.sigma
        self.num_train_timesteps = params.num_train_timesteps
        self.min_timestep = params.min_timestep
        self.max_timestep = params.max_timestep

    @classmethod
    def get_default_params(cls) -> TimeSamplerParams:
        """Return the default parameters for TimeSampler."""
        return TimeSamplerParams()

    def forward(
        self,
        size: tuple[int, ...],
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        min_timestep: float | None = None,
        max_timestep: float | None = None,
        multi_step: bool = False,
    ) -> torch.Tensor:
        """
        Sample time values from the specified distribution.

        When use_logit_normal=True, samples from the logit normal distribution with parameters
        mu and sigma. When use_logit_normal=False, samples uniformly from the range (0, 1).

        Args:
            size (tuple[int, ...]): Size of the output tensor as a tuple of integers.
            device (torch.device, optional): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator.

        Returns:
            torch.Tensor: Sampled time values of the specified size in range (0, 1).
        """
        min_timestep = min_timestep if min_timestep is not None else self.min_timestep
        max_timestep = max_timestep if max_timestep is not None else self.max_timestep
        if self.use_logit_normal:
            timestep = logit_normal_sample(size, mu=self.mu, sigma=self.sigma, device=device, generator=generator, min_timestep=min_timestep, max_timestep=max_timestep)            
            if self.num_train_timesteps is not None:
                timestep = (timestep * self.num_train_timesteps).clamp(0, self.num_train_timesteps - 1)
            return timestep
        else:
            timestep = (torch.rand(size, device=device, dtype=torch.float32, generator=generator)) * (max_timestep - min_timestep) + min_timestep
            #if timestep > 0.999 replace it with 1.0
            # timestep = torch.where(timestep > 0.99, torch.ones_like(timestep), timestep)
            if self.num_train_timesteps is not None:
                timestep = (timestep * self.num_train_timesteps).clamp(0, self.num_train_timesteps - 1)
            return timestep


class DMDTimeSampler(nn.Module, ConfigurableModule[TimeSamplerParams]):
    def __init__(self, params: TimeSamplerParams) -> None:
        nn.Module.__init__(self)
        self.dmd_time_steps = params.dmd_time_steps

    @classmethod
    def get_default_params(cls) -> TimeSamplerParams:
        """Return the default parameters for TimeSampler."""
        return TimeSamplerParams()

    def forward(
        self,
        size: tuple[int, ...],
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        multi_step: bool = False,
    ) -> torch.Tensor:
        """
        Sample time values from the specified distribution.

        When use_logit_normal=True, samples from the logit normal distribution with parameters
        mu and sigma. When use_logit_normal=False, samples uniformly from the range (0, 1).

        Args:
            size (tuple[int, ...]): Size of the output tensor as a tuple of integers.
            device (torch.device, optional): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator.

        Returns:
            torch.Tensor: Sampled time values of the specified size in range (0, 1).
        """
        if multi_step:
            indices = torch.randint(1, len(self.dmd_time_steps), (size[0],), device=device, generator=generator)
            t = torch.tensor(self.dmd_time_steps, dtype=torch.float32, device=device)[indices]
        else:
            t = torch.ones(size, dtype=torch.float32, device=device) * self.dmd_time_steps[0]

        return t


@dataclass
class DecoupledDMDTimeSamplerParams(BaseParams):
    """Parameters for DecoupledDMDTimeSampler (arXiv 2511.22677)."""

    # DM component time sampling
    dm_min_timestep: float = 0.0
    dm_max_timestep: float = 1.0
    dm_use_logit_normal: bool = False
    dm_mu: float = 0.0
    dm_sigma: float = 1.0
    dm_shift: float = 1.0  # Linear time shift for DM: shift*t / (1 + (shift-1)*t)

    # CA component time sampling
    ca_min_timestep: float = 0.5
    ca_max_timestep: float = 1.0
    ca_use_logit_normal: bool = False
    ca_mu: float = 0.0
    ca_sigma: float = 1.0
    ca_shift: float = 1.0  # Linear time shift for CA: shift*t / (1 + (shift-1)*t)


class DecoupledDMDTimeSampler(nn.Module, ConfigurableModule[DecoupledDMDTimeSamplerParams]):
    """Samples independent timesteps for DM and CA components of decoupled DMD."""

    def __init__(self, params: DecoupledDMDTimeSamplerParams) -> None:
        nn.Module.__init__(self)
        self.params = params

    @classmethod
    def get_default_params(cls) -> DecoupledDMDTimeSamplerParams:
        return DecoupledDMDTimeSamplerParams()

    @staticmethod
    def _linear_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
        """Apply linear time shift: shift*t / (1 + (shift-1)*t). Identity when shift=1."""
        return shift * t / (1.0 + (shift - 1.0) * t)

    def _sample(
        self,
        size: tuple[int, ...],
        device: torch.device | None,
        generator: torch.Generator | None,
        use_logit_normal: bool,
        mu: float,
        sigma: float,
        min_t: float,
        max_t: float,
        shift: float,
    ) -> torch.Tensor:
        if use_logit_normal:
            t = logit_normal_sample(
                size, mu=mu, sigma=sigma, device=device,
                generator=generator, min_timestep=min_t, max_timestep=max_t,
            )
        else:
            t = torch.rand(size, device=device, dtype=torch.float32, generator=generator) * (max_t - min_t) + min_t
        if shift != 1.0:
            t = self._linear_time_shift(shift, t)
        return t

    def sample_dm(
        self, size: tuple[int, ...], device: torch.device | None = None, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        p = self.params
        return self._sample(size, device, generator, p.dm_use_logit_normal, p.dm_mu, p.dm_sigma, p.dm_min_timestep, p.dm_max_timestep, p.dm_shift)

    def sample_ca(
        self, size: tuple[int, ...], device: torch.device | None = None, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        p = self.params
        return self._sample(size, device, generator, p.ca_use_logit_normal, p.ca_mu, p.ca_sigma, p.ca_min_timestep, p.ca_max_timestep, p.ca_shift)

    def forward(
        self, size: tuple[int, ...], device: torch.device | None = None, generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tau_DM, tau_CA)."""
        return self.sample_dm(size, device, generator), self.sample_ca(size, device, generator)

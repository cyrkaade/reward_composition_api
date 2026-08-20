import math

import torch as th
from torch import Tensor


class OutputBatchNorm(th.nn.Module):
    """Mini-batch normalization of a scalar model output: batch statistics
    during training (gradients flow through them), running statistics at eval
    time or when the batch is too small to estimate a std."""

    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", th.zeros(1))
        self.register_buffer("running_var", th.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, 1)
        if self.training and flat.shape[0] > 1:
            mean = flat.mean(0)
            var = flat.var(0, unbiased=False)
            with th.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var
        return (x - mean) / th.sqrt(var + self.eps)


class RewardModel(th.nn.Module):
    def __init__(self, input_size=10, hidden_sizes=(200,), learn_alpha=False, alpha_init=1.0, predict_partial=False, batchnorm_output=False, gate_partial=False, gate_init=0.5, tanh_output=False, tanh_scale=1.0):
        super().__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(th.nn.Linear(last_size, hidden_size))
            layers.append(th.nn.LeakyReLU())
            last_size = hidden_size
        self.trunk = th.nn.Sequential(*layers)
        self.head = th.nn.Linear(last_size, 1)
        self.partial_head = th.nn.Linear(last_size, 1) if predict_partial else None
        self.gate_head = th.nn.Linear(last_size, 1) if gate_partial else None
        if self.gate_head is not None:
            # Start the gate at a known constant g = gate_init everywhere: zero the
            # weights and put the value in the bias. gate_init=1 means "trust the
            # partial fully" (the naive baseline), so shrinking it takes evidence.
            clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
            bias = math.log(clamped / (1.0 - clamped))
            th.nn.init.zeros_(self.gate_head.weight)
            th.nn.init.constant_(self.gate_head.bias, bias)
        self.output_bn = OutputBatchNorm() if batchnorm_output else None
        # PEBBLE / B-Pref bound the per-state reward with tanh INSIDE the model, so
        # the bound shapes the Bradley-Terry loss landscape. Clipping in the reward
        # wrapper (model_reward_min/max) happens after training and does not.
        # tanh_scale widens the near-linear region without moving the [-1, 1]
        # bound: tanh(x / scale). scale=1.0 is tanh(x) exactly.
        self.tanh_output = bool(tanh_output)
        self.tanh_scale = float(tanh_scale)
        self.alpha = th.nn.Parameter(th.tensor(float(alpha_init))) if learn_alpha else None

    def forward(self, x):
        out = self.head(self.trunk(x))
        if self.tanh_output:
            out = th.tanh(out / self.tanh_scale)
        if self.output_bn is not None:
            out = self.output_bn(out)
        return out

    def predict_partial(self, x):
        return self.partial_head(self.trunk(x))

    def gate(self, x):
        """Per-state gate g(s,a) in [0,1]: how much to trust the partial reward."""
        return th.sigmoid(self.gate_head(self.trunk(x)))

    def dropout(self, prob):
        m = th.nn.Dropout(prob)
        for layer in self.modules():
            if isinstance(layer, th.nn.Linear):
                layer.weight = th.nn.Parameter(m.forward(layer.weight))
        return self


class PixelRewardModel(th.nn.Module):
    """Convolutional reward model for flattened stacked-pixel features.

    The public input remains ``(..., feature_dim)`` so all preference, delta,
    gate, and ensemble code is shared with :class:`RewardModel`.  Only the
    observation prefix is reshaped and encoded by a Nature-style CNN; the
    one-hot action and optional partial feature are concatenated afterwards.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        extra_feature_size: int,
        hidden_sizes=(200,),
        learn_alpha=False,
        alpha_init=1.0,
        predict_partial=False,
        batchnorm_output=False,
        gate_partial=False,
        gate_init=0.5,
        tanh_output=False,
        tanh_scale=1.0,
    ):
        super().__init__()
        self.observation_shape = tuple(int(value) for value in observation_shape)
        self.observation_size = math.prod(self.observation_shape)
        self.extra_feature_size = int(extra_feature_size)

        channels = self.observation_shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            cnn_size = int(self.cnn(th.zeros((1, *self.observation_shape))).shape[1])
        self.visual_projection = th.nn.Sequential(th.nn.Linear(cnn_size, 256), th.nn.ReLU())

        layers = []
        last_size = 256 + self.extra_feature_size
        for hidden_size in hidden_sizes:
            layers.append(th.nn.Linear(last_size, hidden_size))
            layers.append(th.nn.LeakyReLU())
            last_size = hidden_size
        self.trunk = th.nn.Sequential(*layers)
        self.head = th.nn.Linear(last_size, 1)
        self.partial_head = th.nn.Linear(last_size, 1) if predict_partial else None
        self.gate_head = th.nn.Linear(last_size, 1) if gate_partial else None
        if self.gate_head is not None:
            clamped = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
            bias = math.log(clamped / (1.0 - clamped))
            th.nn.init.zeros_(self.gate_head.weight)
            th.nn.init.constant_(self.gate_head.bias, bias)
        self.output_bn = OutputBatchNorm() if batchnorm_output else None
        self.tanh_output = bool(tanh_output)
        self.tanh_scale = float(tanh_scale)
        self.alpha = th.nn.Parameter(th.tensor(float(alpha_init))) if learn_alpha else None

    def _latent(self, x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        leading_shape = tuple(x.shape[:-1])
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[-1] != self.observation_size + self.extra_feature_size:
            raise ValueError(
                "pixel reward-model input has "
                f"{flat.shape[-1]} features; expected {self.observation_size + self.extra_feature_size}"
            )
        pixels = flat[:, : self.observation_size].reshape(-1, *self.observation_shape)
        extras = flat[:, self.observation_size :]
        visual = self.visual_projection(self.cnn(pixels))
        return self.trunk(th.cat((visual, extras), dim=1)), leading_shape

    @staticmethod
    def _restore(value: Tensor, leading_shape: tuple[int, ...]) -> Tensor:
        return value.reshape(*leading_shape, 1)

    def forward(self, x):
        latent, leading_shape = self._latent(x)
        out = self._restore(self.head(latent), leading_shape)
        if self.tanh_output:
            out = th.tanh(out / self.tanh_scale)
        if self.output_bn is not None:
            out = self.output_bn(out)
        return out

    def predict_partial(self, x):
        latent, leading_shape = self._latent(x)
        return self._restore(self.partial_head(latent), leading_shape)

    def gate(self, x):
        latent, leading_shape = self._latent(x)
        return th.sigmoid(self._restore(self.gate_head(latent), leading_shape))

    def dropout(self, prob):
        dropout = th.nn.Dropout(prob)
        for layer in self.modules():
            if isinstance(layer, (th.nn.Linear, th.nn.Conv2d)):
                layer.weight = th.nn.Parameter(dropout(layer.weight))
        return self


class DeltaLoss(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, input1: Tensor, input2: Tensor, input1_base: Tensor, input2_base: Tensor, target: Tensor, alpha: Tensor | float = 1.0
    ) -> Tensor:
        sum1_pred = input1 + alpha * input1_base
        sum2_pred = input2 + alpha * input2_base
        input_stack = th.stack((th.sum(sum1_pred, dim=[1, 2]), th.sum(sum2_pred, dim=[1, 2])))
        target_stack = th.stack((target, th.subtract(th.ones_like(target), target)))
        probs = th.log_softmax(input_stack, dim=0)
        loss = -th.sum(th.mul(probs, target_stack), dim=0)
        return loss


class PairwiseLoss(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        input_stack = th.stack((th.sum(input1, dim=[1, 2]), th.sum(input2, dim=[1, 2])))
        target_stack = th.stack((target, th.subtract(th.ones_like(target), target)))
        probs = th.log_softmax(input_stack, dim=0)
        loss = -th.sum(th.mul(probs, target_stack), dim=0)
        return loss


class RegularizationLoss(th.nn.Module):
    def __init__(self, regularization_type='L1', lambda_reg=0.01):
        super().__init__()
        if regularization_type not in ["L1", "L2"]:
            raise ValueError("regularization type not L1 or L2")
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg

    def forward(self, model: th.nn.Module) -> Tensor:
        # alpha is anchored by its own mse(alpha, alpha_init) term, not weight decay.
        # gate_head is excluded too: its output is already bounded to [0, 1], so L1
        # here does not regularize capacity, it just drags the gate toward g = 0.5
        # (and in naive mode it acts on a head that receives no gradient in phase 1,
        # destroying the gate_init before the gate is ever trained).
        parameters = [
            p
            for name, p in model.named_parameters()
            if name != "alpha" and not name.startswith("gate_head")
        ]
        if self.regularization_type == 'L1':
            l1_norm = sum(p.abs().sum() for p in parameters)
            return self.lambda_reg * l1_norm
        l2_norm = sum(p.pow(2).sum() for p in parameters)
        return self.lambda_reg * l2_norm


class OutputRegularizationLoss(th.nn.Module):
    def __init__(self, regularization_type='L1', lambda_reg=0.01):
        super().__init__()
        if regularization_type not in ["L1", "L2"]:
            raise ValueError("regularization type not L1 or L2")
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg

    def forward(self, input: Tensor) -> Tensor:
        if self.regularization_type == 'L1':
            return th.mean(th.abs(input), dim=[1, 2]) * self.lambda_reg
        return th.mean(th.pow(input, 2), dim=[1, 2]) * self.lambda_reg


def preference_prob(input, dim=0):
    probs = th.softmax(input, dim=dim)
    return probs

import torch as th
from torch import Tensor


class RewardModel(th.nn.Module):
    def __init__(self, input_size=10, hidden_sizes=(200,), learn_alpha=False, alpha_init=1.0, predict_partial=False):
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
        self.alpha = th.nn.Parameter(th.tensor(float(alpha_init))) if learn_alpha else None

    def forward(self, x):
        return self.head(self.trunk(x))

    def predict_partial(self, x):
        return self.partial_head(self.trunk(x))

    def dropout(self, prob):
        m = th.nn.Dropout(prob)
        for layer in self.modules():
            if isinstance(layer, th.nn.Linear):
                layer.weight = th.nn.Parameter(m.forward(layer.weight))
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
        # alpha is anchored by its own mse(alpha, alpha_init) term, not weight decay
        parameters = [p for name, p in model.named_parameters() if name != "alpha"]
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

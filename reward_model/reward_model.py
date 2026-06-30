import torch as th
from torch import Tensor

class RewardModel(th.nn.Module):
    def __init__(self, input_size=10, hidden_sizes=(200,)):
        super(RewardModel, self).__init__()
        hidden_sizes = tuple(hidden_sizes)
        if hidden_sizes == (200,):
            self.linear1 = th.nn.Linear(input_size, 200)
            self.act1 = th.nn.LeakyReLU()
            self.linear2 = th.nn.Linear(200, 1)
            self.net = None
            return

        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(th.nn.Linear(last_size, hidden_size))
            layers.append(th.nn.LeakyReLU())
            last_size = hidden_size
        layers.append(th.nn.Linear(last_size, 1))
        self.net = th.nn.Sequential(*layers)

    def forward(self, x):
        if self.net is not None:
            return self.net(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x
    
    def dropout(self, prob):
        m = th.nn.Dropout(prob)
        if self.net is not None:
            for layer in self.net:
                if isinstance(layer, th.nn.Linear):
                    layer.weight = th.nn.Parameter(m.forward(layer.weight))
            return self

        self.linear1.weight = th.nn.Parameter(m.forward(self.linear1.weight))
        self.linear2.weight = th.nn.Parameter(m.forward(self.linear2.weight))
        return self

class DeltaLoss(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor, input1_base: Tensor, input2_base: Tensor, target: Tensor) -> Tensor:
        sum1_pred = th.add(input1, input1_base)
        sum2_pred = th.add(input2, input2_base)
        input_stack = th.stack((th.sum(sum1_pred, dim=[1,2]), th.sum(sum2_pred, dim=[1,2])))
        target_stack = th.stack((target, th.subtract(th.ones_like(target), target)))
        probs = th.log_softmax(input_stack, dim=0)
        loss = -th.sum(th.mul(probs, target_stack), dim=0)
        return loss    

class PairwiseLoss(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        input_stack = th.stack((th.sum(input1, dim=[1,2]), th.sum(input2, dim=[1,2])))
        target_stack = th.stack((target, th.subtract(th.ones_like(target), target)))
        probs = th.log_softmax(input_stack, dim=0)
        loss = -th.sum(th.mul(probs, target_stack), dim=0)
        return loss
    
class KullbackLeiblerDivergenceLoss(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = th.kl_div(th.sigmoid(input), th.sigmoid(target), reduction=1)
        return loss
    
class RegularizationLoss():
    def __init__(self, regularization_type='L1', lambda_reg=0.01):
        super().__init__()
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        assert regularization_type in ['L1', 'L2'], "regularization type not L1 or L2"

    def forward(self, model: th.nn.Module) -> Tensor:
        if self.regularization_type == 'L1':
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            return self.lambda_reg * l1_norm
        elif self.regularization_type == 'L2':
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            return self.lambda_reg * l2_norm
        
class OutputRegularizationLoss():
    def __init__(self, regularization_type='L1', lambda_reg=0.01):
        super().__init__()
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        assert regularization_type in ['L1', 'L2'], "regularization type not L1 or L2"

    def forward(self, input: Tensor) -> Tensor:
        if self.regularization_type == 'L1':
            return th.mean(th.abs(input), dim=[1,2]) * self.lambda_reg 
        if self.regularization_type == 'L2':
            return th.mean(th.pow(input, 2), dim=[1,2]) * self.lambda_reg
        
def preference_prob(input, dim=0):
    probs = th.softmax(input, dim=dim)
    return probs

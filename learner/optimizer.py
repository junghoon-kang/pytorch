import numpy as np
import torch.optim as optim
from abc import abstractmethod, abstractproperty, ABCMeta


__all__ = [
    "SGD",
    "Adam",
]


class Base(metaclass=ABCMeta):
    """ Base class is initiated alongside the torch.optim.Optimizer class and
    helps the instance to get and update learning rate of the optimizer.
    """
    def get_lr(self):
        """ 
        Returns:
            lr (float): current learning rate of self (torch.optim.Optimzer)
        """
        for param_group in self.param_groups:
            lr = param_group["lr"]
            return lr

    def update_lr(self, lr):
        """
        Args:
            lr (float): new learning rate value for self (torch.optim.Optimizer)
        """
        for param_group in self.param_groups:
            param_group["lr"] = lr

class SGD(optim.SGD, Base):
    """ Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from On the importance of
    initialization and momentum in deep learning
    (http://www.cs.toronto.edu/~hinton/absps/momentum.pdf).
    """
    def __init__(self, params, lr=0.001, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts
                defining parameter groups
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)
        """
        super(SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

class Adam(optim.Adam, Base):
    """ Implements Adam algorithm.

    It has been proposed in Adam: A Method for Stochastic Optimization
    (https://arxiv.org/abs/1412.6980). The implementation of the L2 penalty
    follows changes proposed in Decoupled Weight Decay Regularization
    (https://arxiv.org/abs/1711.05101).
    """
    def __init__(self, params, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts
                defining parameter groups.
            lr (float, optional): learning rate (default: 1e-3)
            betas (tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of
                this algorithm from the paper On the Convergence of Adam and Beyond
                (default: False)
        """
        super(Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

import torch.optim.lr_scheduler as scheduler
from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    'StepLR',
    'MultiStepLR',
    'CosineAnnealingWarmRestarts',
    'WarmupLR',
]


class StepLR(scheduler.StepLR):
    """ Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super(StepLR, self).__init__(optimizer, step_size, gamma, last_epoch)

class MultiStepLR(scheduler.MultiStepLR):
    """ Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay
    can happen simultaneously with other changes to the learning rate from
    outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        """
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            milestones (list[int]): List of epoch indices. Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        super(MultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)

class CosineAnnealingWarmRestarts(scheduler.CosineAnnealingWarmRestarts):  # FIXME: use lr_min, lr_max
    """ Set the learning rate of each parameter group using a cosine annealing
    schedule, where \eta_{max} is set to the initial lr, T_{cur} is the number
    of epochs since the last restart and T_{i} is the number of epochs between
    two warm restarts in SGDR:

    \eta_{t} = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + cos(\frac{T_{cur}}{T_{i}} \pi))

    When T_{cur} = T_{i}, set \eta_{t} = \eta_{min}. when T_{cur} = 0 after
    restart, set \eta_{t} = \eta_{max}.

    It has been proposed in SGDR: Stochastic Gradient Descent with Warm
    Restarts (https://arxiv.org/abs/1608.03983).
    """
    def __init__(self, optimizer, T_0, T_multi=1, eta_min=0, last_epoch=-1):
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, T_0, T_multi, eta_min, last_epoch)

class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iterations, next_scheduler=None):
        self.warmup_iterations = warmup_iterations
        self.next_scheduler = next_scheduler
        self.done = False
        super(WarmupLR, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_iterations:
            if self.next_scheduler:
                if not self.done:
                    self.next_scheduler.base_lrs = self.base_lrs
                    self.done = True
                    print('** WARMUP DONE **')
                return self.next_scheduler.get_last_lr()
            return self.base_lrs
        return [ base_lr * ( self.last_epoch / self.warmup_iterations ) for base_lr in self.base_lrs ]

    def step(self, epoch=None):
        if self.done and self.next_scheduler:
            if epoch is None:
                self.next_scheduler.step()
            else:
                self.next_scheduler.step(epoch - self.warmup_iterations)
        super(WarmupLR, self).step(epoch)

        
if __name__ == '__main__':
    import torch
    from torch.optim.sgd import SGD

    model = [ torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)) ]
    optimizer = SGD(model, lr=0.1)
    optimizer.zero_grad()

    def StepLR_test():
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        for epoch in range(40):
            print(epoch, optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
    #StepLR_test()

    def MultiStepLR_test():
        scheduler = MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        for epoch in range(40):
            print(epoch, optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
    #MultiStepLR_test()

    #scheduler = scheduler_cosine = CosineAnnealingLR(optimizer, T_max=20)
    #scheduler = scheduler_cosine_restart = CosineAnnealingWarmRestarts(optimizer, T_0=20)
    #scheduler = scheduler_warm = WarmupLR(optimizer, warmup_iterations=5)
    #scheduler = scheduler_warm_step = WarmupLR(optimizer, warmup_iterations=5, next_scheduler=scheduler_step)
    #scheduler = scheduler_warm_cosine = WarmupLR(optimizer, warmup_iterations=5, next_scheduler=scheduler_cosine)
    #scheduler = scheduler_warm_cosine_restart = WarmupLR(optimizer, warmup_iterations=5, next_scheduler=scheduler_cosine_restart)

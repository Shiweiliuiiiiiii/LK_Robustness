import torch


class AvgTensor:

    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()
        # FIXME handle distributed operation

    def reset(self):
        self.sum = None
        self.count = None

    def update(self, val: torch.Tensor, n=1):
        if self.sum is None:
            self.sum = torch.zeros_like(val)
            self.count = torch.tensor(0, dtype=torch.long, device=val.device)
        self.sum += (val * n)
        self.count += n

    def compute(self):
        return self.sum / self.count


class TensorEma:

    """Computes and stores the average and current value"""
    def __init__(self, smoothing_factor=0.9, init_zero=False):
        self.smoothing_factor = smoothing_factor
        self.init_zero = init_zero
        self.val = None
        self.reset()
        # FIXME handle distributed operation

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = torch.zeros_like(val) if self.init_zero else val.clone()
        self.val = (1. - self.smoothing_factor) * val + self.smoothing_factor * self.val

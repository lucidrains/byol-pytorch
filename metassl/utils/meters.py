class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ExponentialMovingAverageMeter(AverageMeter):
    """Computes and stores the exp. moving average, the average and current value"""
    
    def __init__(self, name, window, alpha, fmt):
        super().__init__(name, fmt)
        self.alpha = alpha
        self.window = window
        super().reset()
    
    def reset(self):
        super().reset()
        self.ema = 0
    
    def update(self, val, n=1):
        super().update(val, n)
        self.ema = (self.val * (self.alpha / (1 + self.window))) + self.ema * (1 - (self.alpha / (1 + self.window)))

    def __str__(self):
        fmtstr = '{name} val: {val' + self.fmt + '} (avg: {avg' + self.fmt + '}, ema: {ema' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ExponentialLR, ReduceLROnPlateau


class MyOptimizer:
    
    def __init__(
        self, model_size, model_params, max_steps, iter_per_epoch, optimizer, schedule, warmup, factor, lr_low, lr_high,
        clip_grad, weight_decay, scheduler_epochs
        ):
        
        self.model_params = model_params
        
        self.optimizer = optimizer
        self.scheduler_name = schedule
        self.model_size = model_size
        self.max_steps = max_steps
        self.warmup = warmup
        self.factor = factor
        self.lr_low = lr_low
        self.lr_high = lr_high
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay
        self.iter_per_epoch = iter_per_epoch
        self.scheduler_epochs = scheduler_epochs
        
        self.optimizer = self._get_optimizer(self.optimizer, self.model_params, lr=self.lr_high, weight_decay=self.weight_decay)
        self.scheduler = self._get_scheduler(scheduler_name=self.scheduler_name)
        
        self._step = 0
        self._rate = 0
    
    @staticmethod
    def _get_optimizer(optim_name, params, lr, weight_decay=0.0001):
        if optim_name == "adam":
            return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
        elif optim_name == "adamW":
            return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
        elif optim_name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, alpha=0.98, momentum=0.1, eps=1e-9, weight_decay=weight_decay)
        elif optim_name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        # todo for future: implement NOAM class
    
    def _get_scheduler(self, scheduler_name):
        if scheduler_name == "cosine":
            if self.lr_low is None:
                raise TypeError("lr_low must not be None.")
            return CosineAnnealingLR(self.optimizer, T_max=self.scheduler_epochs, eta_min=self.lr_low)
        elif scheduler_name == "cosineWarm":
            if self.lr_low is None:
                raise TypeError("lr_low must not be None.")
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=self.scheduler_epochs, eta_min=self.lr_low)
        elif scheduler_name == "step":
            # reduces lr by factor of <gamma> after step_size epochs
            return StepLR(self.optimizer, step_size=self.scheduler_epochs)
        elif scheduler_name == "exponential":
            if self.factor is None:
                raise TypeError("factor must not be None.")
            return ExponentialLR(self.optimizer, gamma=self.factor)
        elif scheduler_name == "plateau":
            return ReduceLROnPlateau(self.optimizer, patience=5, min_lr=self.lr_low)
        elif scheduler_name in ["const", "noam", "None"]:
            return None
    
    def _get_warmup_lr(self, lr_high=None):
        # 'const' scheduler case
        if lr_high is not None:
            custom_lr_high = lr_high
        else:
            custom_lr_high = self.lr_high
        
        return custom_lr_high * (float(self._step) / float(max(1, self.warmup)))
    
    def step(self, step, val_loss):
        self._step += 1
        
        self.optimizer.step()
        
        rate = self.rate(step, val_loss=val_loss)
        # print(rate)
        # NOAM is the only schedule that does not set the rate automatically. Warmup is done without torch which is why we also need to set
        # the rate manually in this case
        if self._step <= self.warmup or self.scheduler_name == "noam":
            for p in self.optimizer.param_groups:
                p['lr'] = rate
        self._rate = rate
        
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model_params, self.clip_grad)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self, step=None, val_loss=None):
        """
        step: the actual step inside an epoch (will be reset to 0 after each epoch)
        self._step: the global step (not reset to 0 after each epoch)
        """
        
        if self.scheduler_name == "noam":
            # todo: check which step should be used for NOAM
            if step is None:
                step = self._step
            return self.factor * self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        
        elif self.scheduler_name == "const":
            if not (self.lr_low == self.lr_high):
                raise TypeError("For const. LR schedule, please set lr_low==lr_high.")
            if self._step <= self.warmup:
                if self.lr_high >= 0.1:
                    factor = 10
                else:
                    factor = 100
                return self._get_warmup_lr(lr_high=self.lr_high * factor)
            return self.lr_high
        
        elif self.scheduler_name in ["exponential", "step", "plateau"]:
            # must check global step "_step" instead of "step"
            if self._step <= self.warmup:
                return self._get_warmup_lr()
            # schedulers iterating on epochs
            if self._step % self.iter_per_epoch == 0:
                if self.scheduler_name == "plateau":
                    print("step", val_loss)
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if self.scheduler_name == "plateau":
                # plateau does not inherit from _LRScheduler (no get_last_lr())
                # accessing _last_lr only works if step has been called before
                return self.optimizer.param_groups[0]['lr']
            return self.scheduler.get_last_lr()[0]
        
        elif self.scheduler_name in ["cosineWarm", "cosine", "plateau"]:
            # this if-branch assumes one lr parameter group
            if self._step <= self.warmup:
                return self._get_warmup_lr()
            # schedulers iterating on batches, according to https://github.com/pytorch/pytorch/issues/20028; don't use .step() without
            # step parameter in the current setting
            self.scheduler.step(self._step // self.iter_per_epoch + step / self.iter_per_epoch)
            return self.scheduler.get_last_lr()[0]
        
        elif self.scheduler_name == "None":
            # BOHO case
            return self._rate
    
    def get_state_dict(self):
        return {
            "optimizer":        self.optimizer.state_dict(),
            "scheduler":        self.scheduler.state_dict(),
            "model_params":     list(self.model_params),
            "scheduler_name":   self.scheduler_name,
            "model_size":       self.model_size,
            "max_steps":        self.max_steps,
            "warmup":           self.warmup,
            "factor":           self.factor,
            "lr_low":           self.lr_low,
            "lr_high":          self.lr_high,
            "clip_grad":        self.clip_grad,
            "weight_decay":     self.weight_decay,
            "iter_per_epoch":   self.iter_per_epoch,
            "scheduler_epochs": self.scheduler_epochs,
            }

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
        
        self.global_step = 0
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
        elif optim_name == "lars":
            from torchlars import LARS
            return LARS(torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay))
    
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
        elif scheduler_name in ["const", "None"]:
            return None
    
    def _get_warmup_lr(self, lr_high=None):
        # 'const' scheduler case
        if lr_high is not None:
            custom_lr_high = lr_high
        else:
            custom_lr_high = self.lr_high
        
        return custom_lr_high * (float(self.global_step) / float(max(1, self.warmup)))
    
    def step(self, val_loss=None):
        self.global_step += 1
        
        self.optimizer.step()
        
        rate = self.rate(val_loss=val_loss)
        # print(rate)
        # Warmup is done without torch which is why we also need to set the rate manually in this case
        if self.global_step <= self.warmup:
            for p in self.optimizer.param_groups:
                p['lr'] = rate
        self._rate = rate
        
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model_params, self.clip_grad)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self, val_loss=None):
        """
        self.val_loss: the validation loss (required for plateau scheduler)
        """
        
        if self.scheduler_name == "const":
            if not (self.lr_low == self.lr_high):
                raise TypeError("For const. LR schedule, please set lr_low==lr_high.")
            if self.global_step <= self.warmup:
                if self.lr_high >= 0.1:
                    factor = 10
                else:
                    factor = 100
                return self._get_warmup_lr(lr_high=self.lr_high * factor)
            return self.lr_high
        
        elif self.scheduler_name in ["exponential", "step", "plateau"]:
            # must check global step "global_step" instead of "step"
            if self.global_step <= self.warmup:
                return self._get_warmup_lr()
            # schedulers iterating on epochs
            if self.global_step % self.iter_per_epoch == 0:
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
            if self.global_step <= self.warmup:
                return self._get_warmup_lr()
            # schedulers iterating on batches, according to https://github.com/pytorch/pytorch/issues/20028; don't use .step() without
            # step parameter in the current setting
            self.scheduler.step(self.global_step // self.iter_per_epoch)
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
            "_step":            self.global_step,
            "_rate":            self._rate,
            }
    
    def load_state_dict(self, state_dct):
        self.optimizer.load_state_dict(state_dct["optimizer"])
        self.scheduler.load_state_dict(state_dct["scheduler"])
        self.model_params = [] if not state_dct["model_params"] else state_dct["model_params"]
        self.scheduler_name = state_dct["scheduler_name"]
        self.model_size = state_dct["model_size"]
        self.max_steps = state_dct["max_steps"]
        self.warmup = state_dct["warmup"]
        self.factor = state_dct["factor"]
        self.lr_low = state_dct["lr_low"]
        self.lr_high = state_dct["lr_high"]
        self.clip_grad = state_dct["clip_grad"]
        self.weight_decay = state_dct["weight_decay"]
        self.iter_per_epoch = state_dct["iter_per_epoch"]
        self.scheduler_epochs = state_dct["scheduler_epochs"]
        self.global_step = state_dct["_step"]
        self._rate = state_dct["_rate"]
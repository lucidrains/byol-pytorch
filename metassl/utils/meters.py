import numpy as np
import torch
from torch.nn import functional as F


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


def update_grad_stats_meters(grads, meters, warmup):
    backbone_grads_pt_lw, backbone_grads_pt_global = grads["backbone_grads_pt_lw"], grads["backbone_grads_pt_global"]
    backbone_grads_ft_lw, backbone_grads_ft_global = grads["backbone_grads_ft_lw"], grads["backbone_grads_ft_global"]
    
    # global meters
    cos_sim_ema_meter_global = meters["cos_sim_ema_meter_global"]
    cos_sim_ema_meter_standardized_global = meters["cos_sim_ema_meter_standardized_global"]
    
    dot_prod_meter_global = meters["dot_prod_meter_global"]
    eucl_dis_meter_global = meters["eucl_dis_meter_global"]
    norm_pt_meter_global = meters["norm_pt_meter_global"]
    norm_ft_meter_global = meters["norm_ft_meter_global"]
    
    # layer-wise meters
    cos_sim_ema_meter_lw = meters["cos_sim_ema_meter_lw"]
    cos_sim_std_meter_lw = meters["cos_sim_std_meter_lw"]
    cos_sim_ema_meter_standardized_lw = meters["cos_sim_ema_meter_standardized_lw"]
    cos_sim_std_meter_standardized_lw = meters["cos_sim_std_meter_standardized_lw"]
    
    dot_prod_avg_meter_lw = meters["dot_prod_avg_meter_lw"]
    dot_prod_std_meter_lw = meters["dot_prod_std_meter_lw"]
    eucl_dis_avg_meter_lw = meters["eucl_dis_avg_meter_lw"]
    eucl_dis_std_meter_lw = meters["eucl_dis_std_meter_lw"]
    norm_pt_avg_meter_lw = meters["norm_pt_avg_meter_lw"]
    norm_pt_std_meter_lw = meters["norm_pt_std_meter_lw"]
    norm_ft_avg_meter_lw = meters["norm_ft_avg_meter_lw"]
    norm_ft_std_meter_lw = meters["norm_ft_std_meter_lw"]
    
    if not warmup:
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=backbone_grads_ft_lw, metric_type="cosine")
        cos_sim_ema_meter_lw.update(mean), cos_sim_std_meter_lw.update(std)

        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=backbone_grads_ft_lw, metric_type="cosine", standardize_cosine=True)
        cos_sim_ema_meter_standardized_lw.update(mean), cos_sim_std_meter_standardized_lw.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=backbone_grads_ft_lw, metric_type="dot")
        dot_prod_avg_meter_lw.update(mean), dot_prod_std_meter_lw.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=backbone_grads_ft_lw, metric_type="euclidean")
        eucl_dis_avg_meter_lw.update(mean), eucl_dis_std_meter_lw.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=None, metric_type="norm")
        norm_pt_avg_meter_lw.update(mean), norm_pt_std_meter_lw.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=None, metric_type="norm")
        norm_ft_avg_meter_lw.update(mean), norm_ft_std_meter_lw.update(std)
        
        cos_sim_ema_meter_global.update(F.cosine_similarity(backbone_grads_pt_global, backbone_grads_ft_global, dim=0))

        backbone_grads_pt_global_standardized = (backbone_grads_pt_global - backbone_grads_pt_global.mean()) / backbone_grads_pt_global.std()
        backbone_grads_ft_global_standardized = (backbone_grads_ft_global - backbone_grads_ft_global.mean()) / backbone_grads_ft_global.std()
        cos_sim_ema_meter_standardized_global.update(F.cosine_similarity(backbone_grads_pt_global_standardized, backbone_grads_ft_global_standardized, dim=0))
        
        dot_prod_meter_global.update(torch.dot(backbone_grads_pt_global, backbone_grads_ft_global))
        eucl_dis_meter_global.update(torch.linalg.norm(backbone_grads_pt_global - backbone_grads_ft_global, 2))
        norm_pt_meter_global.update(torch.linalg.norm(backbone_grads_pt_global, 2))
        norm_ft_meter_global.update(torch.linalg.norm(backbone_grads_ft_global, 2))
    
    else:
        # global
        cos_sim_ema_meter_global.update(0.)
        cos_sim_ema_meter_standardized_global.update(0.)
        dot_prod_meter_global.update(0.)
        eucl_dis_meter_global.update(0.)
        norm_pt_meter_global.update(torch.linalg.norm(backbone_grads_pt_global, 2))
        norm_ft_meter_global.update(0.)
        
        # layer-wise
        cos_sim_ema_meter_lw.update(0.), cos_sim_std_meter_lw.update(0.)
        cos_sim_ema_meter_standardized_lw.update(0.), cos_sim_std_meter_standardized_lw.update(0.)
        
        dot_prod_avg_meter_lw.update(0.), dot_prod_std_meter_lw.update(0.)
        eucl_dis_avg_meter_lw.update(0.), eucl_dis_std_meter_lw.update(0.)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt_lw, backbone_grads_ft=None, metric_type="norm")
        norm_pt_avg_meter_lw.update(mean), norm_pt_std_meter_lw.update(std)
        
        norm_ft_avg_meter_lw.update(0.), norm_ft_std_meter_lw.update(0.)


def calc_all_layer_wise_stats(
    backbone_grads_pt,
    backbone_grads_ft,
    cos_sim_ema_meter,
    cos_sim_std_meter,
    dot_prod_avg_meter,
    dot_prod_std_meter,
    eucl_dis_avg_meter,
    eucl_dis_std_meter,
    norm_pt_avg_meter,
    norm_pt_std_meter,
    norm_ft_avg_meter,
    norm_ft_std_meter,
    warmup=False
    ):
    if not warmup:
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt, backbone_grads_ft=backbone_grads_ft, metric_type="cosine")
        cos_sim_ema_meter.update(mean), cos_sim_std_meter.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt, backbone_grads_ft=backbone_grads_ft, metric_type="dot")
        dot_prod_avg_meter.update(mean), dot_prod_std_meter.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt, backbone_grads_ft=backbone_grads_ft, metric_type="euclidean")
        eucl_dis_avg_meter.update(mean), eucl_dis_std_meter.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt, backbone_grads_ft=None, metric_type="norm")
        norm_pt_avg_meter.update(mean), norm_pt_std_meter.update(std)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_ft, backbone_grads_ft=None, metric_type="norm")
        norm_ft_avg_meter.update(mean), norm_ft_std_meter.update(std)
    else:
        cos_sim_ema_meter.update(0.), cos_sim_std_meter.update(0.)
        dot_prod_avg_meter.update(0.), dot_prod_std_meter.update(0.)
        eucl_dis_avg_meter.update(0.), eucl_dis_std_meter.update(0.)
        
        mean, std = calc_layer_wise_stats(backbone_grads_pt=backbone_grads_pt, backbone_grads_ft=None, metric_type="norm")
        norm_pt_avg_meter.update(mean), norm_pt_std_meter.update(std)
        
        norm_ft_avg_meter.update(0.), norm_ft_std_meter.update(0.)


def calc_layer_wise_stats(backbone_grads_pt, backbone_grads_ft=None, metric_type="cosine", standardize_cosine=False):
    allowed_types = ["cosine", "euclidean", "norm", "dot"]
    assert metric_type in allowed_types, f"metric_type must be one of {allowed_types}"
    metric_vals = []
    if backbone_grads_ft is not None:
        for (k1, v1), (k2, v2) in zip(backbone_grads_pt.items(), backbone_grads_ft.items()):
            # todo: for each layer, assumes independence of weights and biases but should be considered as one layer
            if k1 == k2 and "bn" not in k1:
                if metric_type == "euclidean":
                    metric_vals.append(torch.linalg.norm(v1 - v2, 2).cpu().numpy())
                elif metric_type == "dot":
                    metric_vals.append(torch.dot(v1, v2).cpu().numpy())
                elif metric_type == "cosine":
                    if standardize_cosine:
                        v1 = (v1 - v1.mean()) / max(v1.std(), 1e-12)
                        v2 = (v2 - v2.mean()) / max(v2.std(), 1e-12)
                    metric_vals.append(F.cosine_similarity(v1, v2, dim=0).cpu().numpy())
    else:
        for k1, v1 in backbone_grads_pt.items():
            if metric_type == "norm":
                metric_vals.append(torch.linalg.norm(v1, 2).cpu().numpy())
    
    return np.mean(metric_vals), np.std(metric_vals)


def initialize_all_meters_global():
    # general meters
    batch_time_meter = AverageMeter('Time', ':6.3f')
    data_time_meter = AverageMeter('Data', ':6.3f')
    losses_pt_meter = AverageMeter('Loss PT', ':.4f')
    losses_ft_meter = AverageMeter('Loss FT', ':.4e')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')
    
    # reward
    reward_meter = AverageMeter('Reward', ':6.2f')
    
    # global meters
    cos_sim_ema_meter_global = ExponentialMovingAverageMeter("Cos. Sim. PT-FT global. average", window=100, alpha=2, fmt=':6.4f')
    cos_sim_ema_meter_standardized_global = ExponentialMovingAverageMeter("Cos. Sim. (standardized) PT-FT global. average", window=100, alpha=2, fmt=':6.4f')
    dot_prod_meter_global = AverageMeter('Dot Product PT-FT', ':6.4f')
    eucl_dis_meter_global = AverageMeter('Eucl. Dist. PT-FT', ':6.4f')
    norm_pt_meter_global = AverageMeter('Norm PT', ':6.4f')
    norm_ft_meter_global = AverageMeter('Norm FT', ':6.4f')
    
    # layer-wise meters
    cos_sim_ema_meter_lw = ExponentialMovingAverageMeter("Cos. Sim. PT-FT layer-w. average", window=100, alpha=2, fmt=':6.4f')
    cos_sim_std_meter_lw = AverageMeter('Cos. Sim. PT-FT layer-w. std.', ':6.4f')
    
    cos_sim_ema_meter_standardized_lw = ExponentialMovingAverageMeter("Cos. Sim. (standardized) PT-FT layer-w. average", window=100, alpha=2, fmt=':6.4f')
    cos_sim_std_meter_standardized_lw = AverageMeter('Cos. Sim. (standardized) PT-FT layer-w. std.', ':6.4f')
    
    dot_prod_avg_meter_lw = AverageMeter('Dot Product PT-FT layer-w. average', ':6.4f')
    dot_prod_std_meter_lw = AverageMeter('Dot Product PT-FT layer-w. std.', ':6.4f')
    eucl_dis_avg_meter_lw = AverageMeter('Eucl. Dist. PT-FT layer-w. average', ':6.4f')
    eucl_dis_std_meter_lw = AverageMeter('Eucl. Dist. PT-FT layer-w. std.', ':6.4f')
    norm_pt_avg_meter_lw = AverageMeter('Norm PT layer-w. average', ':6.4f')
    norm_pt_std_meter_lw = AverageMeter('Norm PT layer-w. std.', ':6.4f')
    norm_ft_avg_meter_lw = AverageMeter('Norm FT layer-w. average', ':6.4f')
    norm_ft_std_meter_lw = AverageMeter('Norm FT layer-w. std.', ':6.4f')
    
    return {
        "batch_time_meter":                      batch_time_meter,
        "data_time_meter":                       data_time_meter,
        "losses_pt_meter":                       losses_pt_meter,
        "losses_ft_meter":                       losses_ft_meter,
        "top1_meter":                            top1_meter,
        "top5_meter":                            top5_meter,
        "reward_meter":                          reward_meter,
        "dot_prod_meter_global":                 dot_prod_meter_global,
        "eucl_dis_meter_global":                 eucl_dis_meter_global,
        "norm_pt_meter_global":                  norm_pt_meter_global,
        "norm_ft_meter_global":                  norm_ft_meter_global,
        "dot_prod_avg_meter_lw":                 dot_prod_avg_meter_lw,
        "dot_prod_std_meter_lw":                 dot_prod_std_meter_lw,
        "eucl_dis_avg_meter_lw":                 eucl_dis_avg_meter_lw,
        "eucl_dis_std_meter_lw":                 eucl_dis_std_meter_lw,
        "norm_pt_avg_meter_lw":                  norm_pt_avg_meter_lw,
        "norm_pt_std_meter_lw":                  norm_pt_std_meter_lw,
        "norm_ft_avg_meter_lw":                  norm_ft_avg_meter_lw,
        "norm_ft_std_meter_lw":                  norm_ft_std_meter_lw,
        "cos_sim_ema_meter_lw":                  cos_sim_ema_meter_lw,
        "cos_sim_std_meter_lw":                  cos_sim_std_meter_lw,
        "cos_sim_ema_meter_standardized_lw":     cos_sim_ema_meter_standardized_lw,
        "cos_sim_std_meter_standardized_lw":     cos_sim_std_meter_standardized_lw,
        "cos_sim_ema_meter_global":              cos_sim_ema_meter_global,
        "cos_sim_ema_meter_standardized_global": cos_sim_ema_meter_standardized_global,
        }

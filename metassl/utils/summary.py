try:
    # For execution in PyCharm
    from metassl.utils.meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter, initialize_all_meters, update_grad_stats_meters
except ImportError:
    # For execution in command line
    from .meters import AverageMeter, ProgressMeter, ExponentialMovingAverageMeter, initialize_all_meters, update_grad_stats_meters
    

def write_to_summary_writer(total_iter, loss_pt, loss_ft, data_time, batch_time, optimizer_pt, optimizer_ft, top1, top5, meters_to_plot, writer):
    writer.add_scalar('Loss/pre-training', loss_pt.item(), total_iter)
    if isinstance(loss_ft, float):
        writer.add_scalar('Loss/fine-tuning', loss_ft, total_iter)
    else:
        writer.add_scalar('Loss/fine-tuning', loss_ft.item(), total_iter)
    writer.add_scalar('Accuracy/@1', top1.val, total_iter)
    writer.add_scalar('Accuracy/@5', top5.val, total_iter)
    writer.add_scalar('Accuracy/@1 average', top1.avg, total_iter)
    writer.add_scalar('Accuracy/@5 average', top5.avg, total_iter)
    writer.add_scalar('Time/Data', data_time.val, total_iter)
    writer.add_scalar('Time/Batch', batch_time.val, total_iter)
    # assuming only one param group
    writer.add_scalar('Learning rate/pre-training', optimizer_pt.param_groups[0]['lr'], total_iter)
    writer.add_scalar('Learning rate/fine-tuning', optimizer_ft.param_groups[0]['lr'], total_iter)
    
    main_stats_meters = meters_to_plot["main_meters"]
    additional_stats_meters = meters_to_plot["additional_stats_meters"]
    
    for stat in main_stats_meters:
        if isinstance(stat, ExponentialMovingAverageMeter):
            writer.add_scalar(f'Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Advanced Stats/{stat.name} average', stat.avg, total_iter)
            # exponential moving average
            writer.add_scalar(f'Advanced Stats/{stat.name} exp. moving average', stat.ema, total_iter)
        else:
            writer.add_scalar(f'Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Advanced Stats/{stat.name} average', stat.avg, total_iter)
    
    for stat in additional_stats_meters:
        if isinstance(stat, ExponentialMovingAverageMeter):
            writer.add_scalar(f'Additional Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Additional Advanced Stats/{stat.name} average', stat.avg, total_iter)
            # exponential moving average
            writer.add_scalar(f'Additional Advanced Stats/{stat.name} exp. moving average', stat.ema, total_iter)
        else:
            writer.add_scalar(f'Additional Advanced Stats/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Additional Advanced Stats/{stat.name} average', stat.avg, total_iter)

    if "aug_param_meters" in meters_to_plot:
        aug_param_meters = meters_to_plot["aug_param_meters"]
        for stat in aug_param_meters:
            writer.add_scalar(f'Aug. Param Meters/{stat.name}', stat.val, total_iter)
            writer.add_scalar(f'Aug. Param Meters/{stat.name} average', stat.avg, total_iter)
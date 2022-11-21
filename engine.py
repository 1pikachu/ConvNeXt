# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import time
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(args, data_loader, model, device, use_amp=False):
    # switch to evaluation mode
    model.eval()

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use NHWC format")
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0
    profile_len = min(len(data_loader), args.num_iter + args.num_warmup) // 2

    if args.profile and args.device == "xpu":
        for i, batch in enumerate(data_loader):
            if i >= args.num_iter:
                break

            images = batch[0]
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            if args.jit and i == 0:
                try:
                    images = images.to(device, non_blocking=True)
                    model = torch.jit.trace(model, images, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)

            elapsed = time.time()
            images = images.to(device, non_blocking=True)
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                output = model(images)
            torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == profile_len:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, batch in enumerate(data_loader):
                if i >= args.num_iter:
                    break

                images = batch[0]
                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                if args.jit and i == 0:
                    try:
                        images = images.to(device, non_blocking=True)
                        model = torch.jit.trace(model, images, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)

                elapsed = time.time()
                images = images.to(device, non_blocking=True)
                with torch.jit.fuser(fuser_mode):
                    output = model(images)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, batch in enumerate(data_loader):
                if i >= args.num_iter:
                    break

                images = batch[0]
                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                if args.jit and i == 0:
                    try:
                        images = images.to(device, non_blocking=True)
                        model = torch.jit.trace(model, images, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)

                elapsed = time.time()
                images = images.to(device, non_blocking=True)
                output = model(images)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i, batch in enumerate(data_loader):
            if i >= args.num_iter:
                break

            images = batch[0]
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            if args.jit and i == 0:
                try:
                    images = images.to(device, non_blocking=True)
                    model = torch.jit.trace(model, images, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)

            elapsed = time.time()
            images = images.to(device, non_blocking=True)
            with torch.jit.fuser(fuser_mode):
                output = model(images)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i, batch in enumerate(data_loader):
            if i >= args.num_iter:
                break

            images = batch[0]
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            if args.jit and i == 0:
                try:
                    images = images.to(device, non_blocking=True)
                    model = torch.jit.trace(model, images, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)

            elapsed = time.time()
            images = images.to(device, non_blocking=True)
            output = model(images)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    return {}

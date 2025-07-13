# # # Copyright (c) Meta Platforms, Inc. and affiliates.
# # #
# # # This source code is licensed under the Apache License, Version 2.0
# # # found in the LICENSE file in the root directory of this source tree.
# #
# # import argparse
# # import logging
# # import math
# # import os
# # from functools import partial
# #
# # from fvcore.common.checkpoint import PeriodicCheckpointer
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# #
# # from dinov2.data import SamplerType, make_data_loader, make_dataset
# # from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
# # from dinov2.fsdp import FSDPCheckpointer
# # from dinov2.logging import MetricLogger
# # from dinov2.utils.config import setup
# # from dinov2.utils.utils import CosineScheduler
# #
# # from dinov2.train.ssl_meta_arch import SSLMetaArch
# #
# # # Specify the GPUs to use
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6"
# #
# # torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
# # logger = logging.getLogger("dinov2")
# #
# #
# # def get_args_parser(add_help: bool = True):
# #     parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
# #     parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
# #     parser.add_argument(
# #         "--no-resume",
# #         action="store_true",
# #         help="Whether to not attempt to resume from the checkpoint directory. ",
# #     )
# #     parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
# #     parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
# #     parser.add_argument(
# #         "opts",
# #         help="""
# # Modify config options at the end of the command. For Yacs configs, use
# # space-separated "PATH.KEY VALUE" pairs.
# # For python-based LazyConfig, use "path.key=value".
# #         """.strip(),
# #         default=None,
# #         nargs=argparse.REMAINDER,
# #     )
# #     parser.add_argument(
# #         "--output-dir",
# #         "--output_dir",
# #         default="",
# #         type=str,
# #         help="Output directory to save logs and checkpoints",
# #     )
# #
# #     return parser
# #
# #
# # def build_optimizer(cfg, params_groups):
# #     return optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
# #
# #
# # def build_schedulers(cfg):
# #     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
# #     lr = dict(
# #         base_value=cfg.optim["lr"],
# #         final_value=cfg.optim["min_lr"],
# #         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
# #         warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
# #         start_warmup_value=0,
# #     )
# #     wd = dict(
# #         base_value=cfg.optim["weight_decay"],
# #         final_value=cfg.optim["weight_decay_end"],
# #         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
# #     )
# #     momentum = dict(
# #         base_value=cfg.teacher["momentum_teacher"],
# #         final_value=cfg.teacher["final_momentum_teacher"],
# #         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
# #     )
# #     teacher_temp = dict(
# #         base_value=cfg.teacher["teacher_temp"],
# #         final_value=cfg.teacher["teacher_temp"],
# #         total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
# #         warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
# #         start_warmup_value=cfg.teacher["warmup_teacher_temp"],
# #     )
# #
# #     lr_schedule = CosineScheduler(**lr)
# #     wd_schedule = CosineScheduler(**wd)
# #     momentum_schedule = CosineScheduler(**momentum)
# #     teacher_temp_schedule = CosineScheduler(**teacher_temp)
# #     last_layer_lr_schedule = CosineScheduler(**lr)
# #
# #     last_layer_lr_schedule.schedule[
# #         : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
# #     ] = 0  # mimicking the original schedules
# #
# #     logger.info("Schedulers ready.")
# #
# #     return (
# #         lr_schedule,
# #         wd_schedule,
# #         momentum_schedule,
# #         teacher_temp_schedule,
# #         last_layer_lr_schedule,
# #     )
# #
# #
# # def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
# #     for param_group in optimizer.param_groups:
# #         is_last_layer = param_group["is_last_layer"]
# #         lr_multiplier = param_group["lr_multiplier"]
# #         wd_multiplier = param_group["wd_multiplier"]
# #         param_group["weight_decay"] = wd * wd_multiplier
# #         param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
# #
# #
# # def do_test(cfg, model, iteration):
# #     new_state_dict = model.teacher.state_dict()
# #
# #     eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
# #     os.makedirs(eval_dir, exist_ok=True)
# #     # save teacher checkpoint
# #     teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
# #     torch.save({"teacher": new_state_dict}, teacher_ckp_path)
# #
# #
# # def do_train(cfg, model, resume=False):
# #     model.train()
# #     inputs_dtype = torch.half
# #     fp16_scaler = model.fp16_scaler  # for mixed precision training
# #
# #     # setup optimizer
# #     optimizer = build_optimizer(cfg, model.get_params_groups())
# #     (
# #         lr_schedule,
# #         wd_schedule,
# #         momentum_schedule,
# #         teacher_temp_schedule,
# #         last_layer_lr_schedule,
# #     ) = build_schedulers(cfg)
# #
# #     # checkpointer
# #     checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
# #     if cfg.MODEL.WEIGHTS and cfg.MODEL.WEIGHTS != "":
# #         print(f"Loading pre-trained weights from {cfg.MODEL.WEIGHTS}")
# #         start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
# #     else:
# #         print("No pre-trained weights provided. Starting training from scratch.")
# #         start_iter = 0
# #
# #     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
# #     max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
# #
# #     periodic_checkpointer = PeriodicCheckpointer(
# #         checkpointer,
# #         period=3 * OFFICIAL_EPOCH_LENGTH,
# #         max_iter=max_iter,
# #         max_to_keep=3,
# #     )
# #
# #     # setup data preprocessing
# #     img_size = cfg.crops.global_crops_size
# #     patch_size = cfg.student.patch_size
# #     n_tokens = (img_size // patch_size) ** 2
# #     mask_generator = MaskingGenerator(
# #         input_size=(img_size // patch_size, img_size // patch_size),
# #         max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
# #     )
# #
# #     data_transform = DataAugmentationDINO(
# #         cfg.crops.global_crops_scale,
# #         cfg.crops.local_crops_scale,
# #         cfg.crops.local_crops_number,
# #         global_crops_size=cfg.crops.global_crops_size,
# #         local_crops_size=cfg.crops.local_crops_size,
# #     )
# #
# #     collate_fn = partial(
# #         collate_data_and_cast,
# #         mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
# #         mask_probability=cfg.ibot.mask_sample_probability,
# #         n_tokens=n_tokens,
# #         mask_generator=mask_generator,
# #         dtype=inputs_dtype,
# #     )
# #
# #     # setup data loader
# #     dataset = make_dataset(
# #         dataset_str=cfg.train.dataset_path,
# #         transform=data_transform,
# #         target_transform=lambda _: (),
# #     )
# #     data_loader = make_data_loader(
# #         dataset=dataset,
# #         batch_size=cfg.train.batch_size_per_gpu,
# #         num_workers=cfg.train.num_workers,
# #         shuffle=True,
# #         seed=start_iter,
# #         sampler_type=SamplerType.INFINITE,
# #         drop_last=True,
# #         collate_fn=collate_fn,
# #     )
# #
# #     # Multi-GPU setup
# #     if torch.cuda.device_count() > 1:
# #         model = nn.DataParallel(model)
# #
# #     # training loop
# #     iteration = start_iter
# #     logger.info("Starting training from iteration {}".format(start_iter))
# #     metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
# #     metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
# #     header = "Training"
# #
# #     for data in metric_logger.log_every(
# #         data_loader,
# #         10,
# #         header,
# #         max_iter,
# #         start_iter,
# #     ):
# #         current_batch_size = data["collated_global_crops"].shape[0] / 2
# #         if iteration > max_iter:
# #             return
# #
# #         # apply schedules
# #         lr = lr_schedule[iteration]
# #         wd = wd_schedule[iteration]
# #         mom = momentum_schedule[iteration]
# #         teacher_temp = teacher_temp_schedule[iteration]
# #         last_layer_lr = last_layer_lr_schedule[iteration]
# #         apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
# #
# #         # compute losses
# #         optimizer.zero_grad(set_to_none=True)
# #         loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
# #
# #         # clip gradients
# #         if fp16_scaler is not None:
# #             if cfg.optim.clip_grad:
# #                 fp16_scaler.unscale_(optimizer)
# #                 nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
# #             fp16_scaler.step(optimizer)
# #             fp16_scaler.update()
# #         else:
# #             if cfg.optim.clip_grad:
# #                 nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
# #             optimizer.step()
# #
# #         # perform teacher EMA update
# #         model.module.update_teacher(mom) if isinstance(model, nn.DataParallel) else model.update_teacher(mom)
# #
# #         # logging
# #         loss_dict_reduced = {k: v.item() for k, v in loss_dict.items() if not math.isnan(v.item())}
# #         losses_reduced = sum(loss_dict_reduced.values())
# #
# #         metric_logger.update(lr=lr)
# #         metric_logger.update(wd=wd)
# #         metric_logger.update(mom=mom)
# #         metric_logger.update(last_layer_lr=last_layer_lr)
# #         metric_logger.update(current_batch_size=current_batch_size)
# #         metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
# #
# #         # checkpointing and testing
# #         if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
# #             do_test(cfg, model, f"training_{iteration}")
# #             torch.cuda.synchronize()
# #         periodic_checkpointer.step(iteration)
# #
# #         iteration += 1
# #
# #     metric_logger.synchronize_between_processes()
# #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
# #
# #
# # def main(args):
# #
# #     cfg = setup(args)
# #
# #     model = SSLMetaArch(cfg).to(torch.device("cuda"))
# #     logger.info("Model:\n{}".format(model))
# #
# #     if torch.cuda.device_count() > 1:
# #         logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
# #
# #     if args.eval_only:
# #         iteration = (
# #             FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
# #             .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
# #             .get("iteration", -1)
# #             + 1
# #         )
# #         return do_test(cfg, model, f"manual_{iteration}")
# #
# #     do_train(cfg, model, resume=not args.no_resume)
# #
# #
# # if __name__ == "__main__":
# #     args = get_args_parser(add_help=True).parse_args()
# #     main(args)
#
#
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This source code is licensed under the Apache License, Version 2.0
# # found in the LICENSE file in the root directory of this source tree.
#
# import argparse
# import logging
# import math
# import os
# from functools import partial
#
# from fvcore.common.checkpoint import PeriodicCheckpointer
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# from dinov2.data import SamplerType, make_data_loader, make_dataset
# from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
# from dinov2.fsdp import FSDPCheckpointer
# from dinov2.logging import MetricLogger
# from dinov2.utils.config import setup
# from dinov2.utils.utils import CosineScheduler
#
# from dinov2.train.ssl_meta_arch import SSLMetaArch
#
# # Specify the GPUs to use
#
#
# torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
# logger = logging.getLogger("dinov2")
#
#
# def get_args_parser(add_help: bool = True):
#     parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
#     parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
#     parser.add_argument(
#         "--no-resume",
#         action="store_true",
#         help="Whether to not attempt to resume from the checkpoint directory. ",
#     )
#     parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
#     parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
#     parser.add_argument(
#         "opts",
#         help="""
# Modify config options at the end of the command. For Yacs configs, use
# space-separated "PATH.KEY VALUE" pairs.
# For python-based LazyConfig, use "path.key=value".
#         """.strip(),
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
#     parser.add_argument(
#         "--output-dir",
#         "--output_dir",
#         default="",
#         type=str,
#         help="Output directory to save logs and checkpoints",
#     )
#
#     return parser
#
#
# def build_optimizer(cfg, params_groups):
#     return optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
#
#
# def build_schedulers(cfg):
#     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
#     lr = dict(
#         base_value=cfg.optim["lr"],
#         final_value=cfg.optim["min_lr"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#         warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
#         start_warmup_value=0,
#     )
#     wd = dict(
#         base_value=cfg.optim["weight_decay"],
#         final_value=cfg.optim["weight_decay_end"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#     )
#     momentum = dict(
#         base_value=cfg.teacher["momentum_teacher"],
#         final_value=cfg.teacher["final_momentum_teacher"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#     )
#     teacher_temp = dict(
#         base_value=cfg.teacher["teacher_temp"],
#         final_value=cfg.teacher["teacher_temp"],
#         total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
#         warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
#         start_warmup_value=cfg.teacher["warmup_teacher_temp"],
#     )
#
#     lr_schedule = CosineScheduler(**lr)
#     wd_schedule = CosineScheduler(**wd)
#     momentum_schedule = CosineScheduler(**momentum)
#     teacher_temp_schedule = CosineScheduler(**teacher_temp)
#     last_layer_lr_schedule = CosineScheduler(**lr)
#
#     last_layer_lr_schedule.schedule[
#         : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
#     ] = 0  # mimicking the original schedules
#
#     logger.info("Schedulers ready.")
#
#     return (
#         lr_schedule,
#         wd_schedule,
#         momentum_schedule,
#         teacher_temp_schedule,
#         last_layer_lr_schedule,
#     )
#
#
# def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
#     for param_group in optimizer.param_groups:
#         is_last_layer = param_group["is_last_layer"]
#         lr_multiplier = param_group["lr_multiplier"]
#         wd_multiplier = param_group["wd_multiplier"]
#         param_group["weight_decay"] = wd * wd_multiplier
#         param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
#
#
# def do_test(cfg, model, iteration):
#     new_state_dict = model.module.teacher.state_dict() if isinstance(model, DDP) else model.teacher.state_dict()
#
#     eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
#     os.makedirs(eval_dir, exist_ok=True)
#     # save teacher checkpoint
#     teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
#     torch.save({"teacher": new_state_dict}, teacher_ckp_path)
#
#
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
#
# def load_adjusted_checkpoint(checkpointer, path):
#     """
#     Loads a model checkpoint with adjustments for FSDP.
#
#     Args:
#         checkpointer (FSDPCheckpointer): The checkpointer instance managing the FSDP model.
#         path (str): Path to the checkpoint file.
#
#     Returns:
#         dict: The state dictionary loaded from the checkpoint.
#     """
#     print("Loading checkpoint from:", path)
#     try:
#         # Load the checkpoint file
#         checkpoint = torch.load(path, map_location='cpu')
#
#         # Adjust format if 'model' key is missing (this is for compatibility with non-FSDP saved models)
#         if 'model' not in checkpoint:
#             checkpoint = {'model': checkpoint}  # Wrap the loaded dict if it was saved without the 'model' key
#
#         # Ensure the correct state dict type is set before loading
#         if isinstance(checkpointer.model, FSDP):
#             # 如果模型是 FSDP 类型，使用 FULL_STATE_DICT 来加载
#             with FSDP.state_dict_type(checkpointer.model, StateDictType.FULL_STATE_DICT):
#                 checkpointer.model.load_state_dict(checkpoint['model'], strict=False)
#         else:
#             # 如果不是 FSDP 类型，直接使用普通方式加载
#             checkpointer.model.load_state_dict(checkpoint['model'], strict=False)
#
#         print("Checkpoint loaded successfully.")
#         return {"iteration": checkpoint.get("iteration", 0)}
#
#     except KeyError as e:
#         print(f"KeyError: {e}")
#         raise
#     except Exception as e:
#         print(f"Error loading checkpoint: {e}")
#         raise
#
#
#
# def do_train(cfg, model, resume=False):
#     # scaler = torch.cuda.amp.GradScaler()
#
#     model.train()  # 调用修复后的 train 方法
#     inputs_dtype = torch.half
#     fp16_scaler = model.module.fp16_scaler if hasattr(model, "module") else model.fp16_scaler
#
#     optimizer = build_optimizer(cfg, model.module.get_params_groups() if hasattr(model, "module") else model.get_params_groups())
#     (
#         lr_schedule,
#         wd_schedule,
#         momentum_schedule,
#         teacher_temp_schedule,
#         last_layer_lr_schedule,
#     ) = build_schedulers(cfg)
#
#     checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
#     start_iter = 0
#     if cfg.MODEL.WEIGHTS and cfg.MODEL.WEIGHTS != "":
#         # load_adjusted_checkpoint(cfg.MODEL.WEIGHTS, checkpointer)
#         # start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
#         start_iter = load_adjusted_checkpoint(checkpointer, cfg.MODEL.WEIGHTS).get("iteration", -1) + 1
#
#
#     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
#     max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
#
#     # periodic_checkpointer = PeriodicCheckpointer(
#     #     checkpointer,
#     #     period=3 * OFFICIAL_EPOCH_LENGTH,
#     #     max_iter=max_iter,
#     #     max_to_keep=3,
#     # )
#     periodic_checkpointer = PeriodicCheckpointer(
#         checkpointer,
#         period=3 * OFFICIAL_EPOCH_LENGTH,
#         max_iter=max_iter,
#         max_to_keep=3,
#     )
#
#     img_size = cfg.crops.global_crops_size
#     patch_size = cfg.student.patch_size
#     n_tokens = (img_size // patch_size) ** 2
#     mask_generator = MaskingGenerator(
#         input_size=(img_size // patch_size, img_size // patch_size),
#         max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
#     )
#
#     data_transform = DataAugmentationDINO(
#         cfg.crops.global_crops_scale,
#         cfg.crops.local_crops_scale,
#         cfg.crops.local_crops_number,
#         global_crops_size=cfg.crops.global_crops_size,
#         local_crops_size=cfg.crops.local_crops_size,
#     )
#
#     collate_fn = partial(
#         collate_data_and_cast,
#         mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#         mask_probability=cfg.ibot.mask_sample_probability,
#         n_tokens=n_tokens,
#         mask_generator=mask_generator,
#         dtype=inputs_dtype,
#     )
#
#     dataset = make_dataset(
#         dataset_str=cfg.train.dataset_path,
#         transform=data_transform,
#         target_transform=lambda _: (),
#     )
#     data_loader = make_data_loader(
#         dataset=dataset,
#         batch_size=cfg.train.batch_size_per_gpu,
#         num_workers=cfg.train.num_workers,
#         shuffle=True,
#         seed=start_iter,
#         sampler_type=SamplerType.INFINITE,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )
#
#     iteration = start_iter
#     logger.info("Starting training from iteration {}".format(start_iter))
#     metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
#     metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
#     header = "Training"
#
#     for data in metric_logger.log_every(
#         data_loader,
#         10,
#         header,
#         max_iter,
#         start_iter,
#     ):
#         if iteration > max_iter:
#             return
#
#         # Apply learning rate schedules
#         lr = lr_schedule[iteration]
#         wd = wd_schedule[iteration]
#         mom = momentum_schedule[iteration]
#         teacher_temp = teacher_temp_schedule[iteration]
#         last_layer_lr = last_layer_lr_schedule[iteration]
#
#         # Update optimizer
#         apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
#
#         # Forward and backward
#         optimizer.zero_grad(set_to_none=True)
#         # with torch.cuda.amp.autocast():
#         #     loss_dict = model.module.forward_backward(data, teacher_temp=teacher_temp)
#
#         loss_dict = model.module.forward_backward(data, teacher_temp=teacher_temp)
#
#         # scaler.scale(sum(loss_dict.values())).backward()
#
#         # Update gradients
#         if fp16_scaler is not None:
#             if cfg.optim.clip_grad:
#                 fp16_scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
#             fp16_scaler.step(optimizer)
#             fp16_scaler.update()
#         else:
#             if cfg.optim.clip_grad:
#                 nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
#             optimizer.step()
#
#             # 优化器更新
#         # scaler.step(optimizer)
#         # scaler.update()
#         # Teacher update
#         model.module.update_teacher(mom) if hasattr(model, "module") else model.update_teacher(mom)
#
#         # Logging
#         loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
#         metric_logger.update(lr=lr, wd=wd, mom=mom, total_loss=sum(loss_dict_reduced.values()), **loss_dict_reduced)
#
#         # Periodic checkpoint and validation
#         if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
#             do_test(cfg, model, f"training_{iteration}")
#             torch.cuda.synchronize()
#         periodic_checkpointer.step(iteration)
#
#         iteration += 1
#
#     metric_logger.synchronize_between_processes()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
#
#
# def main(args):
#
#     cfg = setup(args)
#     # local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     local_rank = int(os.environ['LOCAL_RANK'])
#     # 初始化分布式训练
#     if not dist.is_initialized():
#         dist.init_process_group(backend="nccl")
#
#
#     # torch.cuda.set_device(local_rank)
#     # dist.init_process_group(backend='nccl')
#
#     torch.cuda.set_device(local_rank)
#
#     device = torch.device("cuda", local_rank)
#
#     # 初始化模型
#     model = SSLMetaArch(cfg).to(device).half()
#
#     # 使用 DistributedDataParallel 包装模型
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank)
#
#     logger.info("Model:\n{}".format(model))
#
#     if args.eval_only:
#         iteration = (
#             FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
#             .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
#             .get("iteration", -1)
#             + 1
#         )
#         return do_test(cfg, model, f"manual_{iteration}")
#
#     do_train(cfg, model, resume=not args.no_resume)
#
#     # 销毁分布式进程组
#     dist.destroy_process_group()
#
#
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
#     args = get_args_parser(add_help=True).parse_args()
#     main(args)


import argparse
import logging
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler
from dinov2.train.ssl_meta_arch import SSLMetaArch

# Set the GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--no-resume", action="store_true", help="Whether to not attempt to resume from the checkpoint directory.")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs. For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--output-dir", default="", type=str, help="Output directory to save logs and checkpoints")
    return parser

def build_optimizer(cfg, params_groups):
    return optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)
    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = 0

    logger.info("Schedulers ready.")
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule

def do_test(cfg, model, iteration):
    new_state_dict = model.module.teacher.state_dict() if isinstance(model, DDP) else model.teacher.state_dict()

    eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
    os.makedirs(eval_dir, exist_ok=True)
    # save teacher checkpoint
    teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
    torch.save({"teacher": new_state_dict}, teacher_ckp_path)

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

def load_adjusted_checkpoint(checkpointer, path):
    print("Loading checkpoint from:", path)
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if 'model' not in checkpoint:
            checkpoint = {'model': checkpoint}
        if isinstance(checkpointer.model, FSDP):
            with FSDP.state_dict_type(checkpointer.model, StateDictType.FULL_STATE_DICT):
                checkpointer.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            checkpointer.model.load_state_dict(checkpoint['model'], strict=False)
        print("Checkpoint loaded successfully.")
        return {"iteration": checkpoint.get("iteration", 0)}
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def save_adjusted_checkpoint(checkpointer, path, iteration=0, **kwargs):
    print(f"Saving checkpoint to: {path}")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {"iteration": iteration}
        if isinstance(checkpointer.model, FSDP):
            with FSDP.state_dict_type(checkpointer.model, StateDictType.LOCAL_STATE_DICT):
                checkpoint['model'] = checkpointer.model.state_dict()
        else:
            checkpoint['model'] = checkpointer.model.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved successfully at {path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        raise

def do_train(cfg, model, resume=False):

    model.train()
    inputs_dtype = torch.half
    # inputs_dtype = torch.float32
    fp16_scaler = model.module.fp16_scaler if hasattr(model, "module") else model.fp16_scaler
    optimizer = build_optimizer(cfg, model.module.get_params_groups() if hasattr(model, "module") else model.get_params_groups())
    lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = 0
    if cfg.MODEL.WEIGHTS and cfg.MODEL.WEIGHTS != "" and cfg.MODEL.C2F:
        start_iter = 1
        print(" From C 2 F FFFFFFFFF")
    elif cfg.MODEL.WEIGHTS and cfg.MODEL.WEIGHTS != "":
        start_iter = load_adjusted_checkpoint(checkpointer, cfg.MODEL.WEIGHTS).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    dataset = make_dataset(dataset_str=cfg.train.dataset_path, transform=data_transform, target_transform=lambda _: ())
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=SamplerType.INFINITE,
        drop_last=True,
        collate_fn=collate_fn,
    )
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    # fp16_scaler = None
    for data in metric_logger.log_every(data_loader, 10, header, max_iter, start_iter):
        if iteration > max_iter:
            return
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.module.forward_backward(data, teacher_temp=teacher_temp)
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad)
            optimizer.step()
        model.module.update_teacher(mom) if hasattr(model, "module") else model.update_teacher(mom)
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        metric_logger.update(lr=lr, wd=wd, mom=mom, total_loss=sum(loss_dict_reduced.values()), **loss_dict_reduced)
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        if iteration % (3 * OFFICIAL_EPOCH_LENGTH) == 0 or iteration == max_iter - 1:
            save_path = os.path.join(cfg.train.output_dir, f"checkpoint_{iteration}.pth")
            save_adjusted_checkpoint(checkpointer, save_path, iteration=iteration, optimizer_state=optimizer.state_dict())
        iteration += 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    cfg = setup(args)
    local_rank = int(os.environ['LOCAL_RANK'])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = SSLMetaArch(cfg).to(device).half()
    # model = SSLMetaArch(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1) + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    do_train(cfg, model, resume=not args.no_resume)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

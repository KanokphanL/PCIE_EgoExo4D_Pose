# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json
import torch

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    # num_layers = len(model.blocks) + 1
    num_layers = len(model.img_backbone.blocks)
    print("num_layers =", num_layers)

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        #TODO: get layer id for ConvNeXt
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
        
    for k, v in param_groups.items():
        print(k, v['lr_scale'], v['weight_decay'])
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    # if name in ['cls_token', 'pos_embed']:
    if 'pos_embed' in name:
        return 0
    elif 'cls_token' in name:
        return 0
    # elif name.startswith('patch_embed'):
    elif 'patch_embed' in name:
        return 0
    elif name.startswith('img_backbone.blocks'):
        return int(name.split('.')[2]) + 1
    elif name.startswith('backbone.layers'):
        return int(name.split('.')[2]) + 1
    elif name.startswith('depth_backbone.blocks'):
        return int(name.split('.')[2]) + 1
    elif name.startswith('depth_backbone.layers'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers


def create_optimizer(config, model):
    optimizer = None
    scheduler = None

    lr = config.TRAIN.LR
    weight_decay = config.TRAIN.WEIGHT_DECAY
    momentum = config.TRAIN.get("MOMENTUM", 0.9)

    optimizer_name = config.TRAIN.OPTIMIZER
    scheduler_name = config.TRAIN.get("SCHEDULER", "CosineAnnealingLR")

    layer_decay = config.TRAIN.get("LAYER_DECAY", 1.0)

    if layer_decay < 1.0: 
        print("Using LR decay")
        no_weight_decay_list = ['pos_embed', 'cls_token']
        param_groups = param_groups_lrd(
            model, 
            weight_decay=weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        
        return optimizer, scheduler

    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Cosine Annealing LR Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["TRAIN"]["END_EPOCH"],  # 总的 epoch 数
            eta_min=config["TRAIN"].get("MIN_LR", 0),  # 最小学习率
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 根据配置选择调度器
    if scheduler_name == 'StepLR':
        step_size = config["TRAIN"].get("STEP_SIZE", 10)
        gamma = config["TRAIN"].get("GAMMA", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'CosineAnnealingLR':
        end_epoch = config["TRAIN"].get('END_EPOCH', 10)
        min_lr = config["TRAIN"].get("MIN_LR", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=min_lr)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler

# For InternViT layer wise lr delay
# def get_layer_id_for_internvit(name, num_layers=24):
#     """
#     Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
#     consecutive blocks, including possible neighboring downsample layers;
#     adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
#     """
#     # num_max_layer = 12

#     if name.startswith("backbone."):
#         name = name[9:]

#         # if name in ['cls_token', 'pos_embed']:
#     if 'position_embedding' in name:
#         return 0
#     elif 'class_embedding' in name:
#         return 0
#     # elif name.startswith('patch_embed'):
#     elif 'patch_embedding' in name:
#         return 0
#     elif name.startswith('backbone.blocks'):
#         return int(name.split('.')[2]) + 1
#     elif name.startswith('encoder.layers'):
#         return int(name.split('.')[2]) + 1
#     else:
#         return num_layers


# def parameter_groups_lrd_for_internvit(
#     model, 
#     weight_decay=1e-5, 
#     skip_list=(),
#     layer_decay=0.9, 
#     get_layer_id=get_layer_id_for_internvit, 
#     get_layer_scale=None):

#     parameter_group_names = {}
#     parameter_group_vars = {}

#     num_layers = len(model.backbone.encoder.layers)
#     print('num_layers =', num_layers)

#     layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
#             group_name = "no_decay"
#             this_weight_decay = 0.
#         else:
#             group_name = "decay"
#             this_weight_decay = weight_decay
#         if get_layer_id is not None:
#             layer_id = get_layer_id(name, num_layers)
#             group_name = "layer_%d_%s" % (layer_id, group_name)
#         else:
#             layer_id = None

#         if group_name not in parameter_group_names:
#             # if get_layer_scale is not None:
#             #     scale = get_layer_scale(layer_id)
#             # else:
#             #     scale = 1.
#             if layer_id is None:
#                 scale = 1.
#             else:
#                 scale = layer_scales[layer_id]

#             parameter_group_names[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr_scale": scale
#             }
#             parameter_group_vars[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr_scale": scale
#             }

#         parameter_group_vars[group_name]["params"].append(param)
#         parameter_group_names[group_name]["params"].append(name)
#     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#     return list(parameter_group_vars.values())



# # support convnext layer wise lr decay
# def get_num_layer_for_convnext(var_name):
#     """
#     Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
#     consecutive blocks, including possible neighboring downsample layers;
#     adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
#     """
#     num_max_layer = 12

#     if var_name.startswith("backbone."):
#         var_name = var_name[9:]

#     if var_name.startswith("downsample_layers"):
#         stage_id = int(var_name.split('.')[1])
#         if stage_id == 0:
#             layer_id = 0
#         elif stage_id == 1 or stage_id == 2:
#             layer_id = stage_id + 1
#         elif stage_id == 3:
#             layer_id = 12
#         return layer_id

#     elif var_name.startswith("stages"):
#         stage_id = int(var_name.split('.')[1])
#         block_id = int(var_name.split('.')[2])
#         if stage_id == 0 or stage_id == 1:
#             layer_id = stage_id + 1
#         elif stage_id == 2:
#             layer_id = 3 + block_id // 3 
#         elif stage_id == 3:
#             layer_id = 12
#         return layer_id
#     else:
#         return num_max_layer

# def get_parameter_groups_for_convnext(
#     model, 
#     weight_decay=1e-5, 
#     skip_list=(),
#     layer_decay=0.9, 
#     get_num_layer=get_num_layer_for_convnext, 
#     get_layer_scale=None):
#     parameter_group_names = {}
#     parameter_group_vars = {}

#     num_layers = 12
#     print('num_layers =', num_layers)

#     layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
#             group_name = "no_decay"
#             this_weight_decay = 0.
#         else:
#             group_name = "decay"
#             this_weight_decay = weight_decay
#         if get_num_layer is not None:
#             layer_id = get_num_layer(name)
#             group_name = "layer_%d_%s" % (layer_id, group_name)
#         else:
#             layer_id = None

#         if group_name not in parameter_group_names:
#             # if get_layer_scale is not None:
#             #     scale = get_layer_scale(layer_id)
#             # else:
#             #     scale = 1.
#             if layer_id is None:
#                 scale = 1.
#             else:
#                 scale = layer_scales[layer_id]

#             parameter_group_names[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr_scale": scale
#             }
#             parameter_group_vars[group_name] = {
#                 "weight_decay": this_weight_decay,
#                 "params": [],
#                 "lr_scale": scale
#             }

#         parameter_group_vars[group_name]["params"].append(param)
#         parameter_group_names[group_name]["params"].append(name)
#     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
#     return list(parameter_group_vars.values())


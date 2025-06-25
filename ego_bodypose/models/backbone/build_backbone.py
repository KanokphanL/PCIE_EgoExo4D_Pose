import torch
import torchvision.models as models
from timm.models import create_model
from .rgbd_resnet import RGBDResNet
from torchvision.models.video import r3d_18
from .vit_mae import vit_base_patch16_224, vit_small_patch16_224

def build_backbone(config):
    backbone_type = config.get('TYPE', 'resnet')
    pretrained = config.get('PRETRAINED', False)
    local_pretrained_path = config.get('LOCAL_PRETRAINED_PATH', None)

    print(f"Initialize backbone: {backbone_type}")

    if backbone_type == 'resnet' or  backbone_type == 'resnet50':
        backbone = models.resnet50(weights='DEFAULT')
        # 去掉最后2层 - 池化层, 全连接层，只保留特征提取部分 
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # (N, C, H, W)
    elif backbone_type == 'resnet18':
        # backbone = models.resnet18(pretrained=True)
        backbone = models.resnet18(pretrained=True)
        # 去掉最后2层 - 池化层, 全连接层，只保留特征提取部分 
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # (N, C, H, W)
    elif backbone_type == 'resnet34':
        backbone = models.resnet18(pretrained=True)
        # 去掉最后2层 - 池化层, 全连接层，只保留特征提取部分 
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # (N, C, H, W)
    elif backbone_type == 'rgbd_resnet':
        version = config.get('VERSION', 18)
        print(f'version = {version}')
        backbone = RGBDResNet(resnet_version=version)
    elif backbone_type == 'r3d_18':
        # print('building r3d_18')
        backbone = r3d_18(weights='DEFAULT')
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # (N, C, H, W)
    elif backbone_type == 'vit_s':
        # load VideoMae v2 - small model
        num_frames = config.get('NUM_FRAMES', 16)
        backbone = vit_small_patch16_224(pretrained=pretrained, all_frames=num_frames)
        # 去掉最后2层分类头
        backbone.head = torch.nn.Identity()
        backbone.head_dropout = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    # load checkpoint
    if local_pretrained_path is not None:
        checkpoint = torch.load(local_pretrained_path, map_location="cpu")
        ckpt = checkpoint["module"]

        # 加载状态字典，忽略缺失的键
        missing_keys, unexpected_keys = backbone.load_state_dict(ckpt, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    return backbone


# 示例配置
if __name__ == "__main__":
    config = {
        'TYPE': 'resnet',
        'PRETRAINED': True,
        'LOCAL_PRETRAINED_PATH': 'path/to/your/local/pretrained_weights.pth'
    }

    backbone = build_backbone(config)
    # 将骨干网络设置为评估模式
    backbone.eval()

    # 模拟输入数据，这里假设输入为 1 张 3 通道，224x224 大小的图像
    input_tensor = torch.randn(1, 3, 224, 224)

    # 使用骨干网络提取特征
    with torch.no_grad():
        features = backbone(input_tensor)
        # 对于 ResNet，输出维度是 (N, C, 1, 1)，这里进行调整
        if config['TYPE'] == 'resnet':
            features = features.view(features.size(0), -1)

    print("特征的形状:", features.shape)
import torch
import torch.nn as nn

class LossAnnotation2D(nn.Module):
    def __init__(self):
        super(LossAnnotation2D, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (batch_size, 17, 2)
        target: (batch_size, 17, 2)
        mask: (batch_size, 17, 1)
        """
        loss = torch.nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
        
        loss = loss / mask.sum()
        
        return loss
    
class LossAnnotation3D(nn.Module):
    def __init__(self):
        super(LossAnnotation3D, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (batch_size, 17, 3)
        target: (batch_size, 17, 3)
        mask: (batch_size, 17, 1)
        """
        loss = torch.nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
        
        loss = loss / mask.sum()
        
        return loss
    
class MPJPELoss(nn.Module):
    def __init__(self):
        super(MPJPELoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: (batch_size, num_joints, 3)
        target: (batch_size, num_joints, 3)
        """
        # 检查 target 中的 NaN 值，并创建掩码
        valid_mask = ~torch.isnan(target).any(dim=-1)  # 如果一个关节的任何一个坐标是 NaN，则该关节无效
        target[~valid_mask] = 0
        # 计算每个关节的欧几里得距离
        distance = torch.norm(pred - target, dim=-1) * valid_mask

        # 计算平均值
        mpjpe = torch.sum(distance) / (valid_mask.sum() + 1e-8)  # 添加一个小的常数避免除以零
        mpjpe = mpjpe * 100 # 将距离转换为厘米
        return mpjpe
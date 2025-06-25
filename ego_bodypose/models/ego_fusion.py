import torch.nn as nn
import torch
import einops
import torch.nn.functional as F

from models.output_head_transformer import OutputHeadTransformer
from models.backbone.build_backbone import build_backbone

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.BatchNorm1d(d_model)  # 使用 BatchNorm 替代 LayerNorm
        self.norm2 = nn.BatchNorm1d(d_model)  # 使用 BatchNorm 替代 LayerNorm
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src.permute(0, 2, 1)).permute(0, 2, 1)  # 注意 BatchNorm 的输入维度
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src.permute(0, 2, 1)).permute(0, 2, 1)  # 注意 BatchNorm 的输入维度
        return src


class EgoFusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        print("Initializing EgoFusion ...")
        
        input_dim = config["DATASET"]["INPUT_DIM"]

        self.use_imu = config["DATASET"]["USE_IMU"]
        self.use_image = False if config["DATASET"]["USE_IMAGE_MODE"] == 'none' else True
        self.use_depth = config.DATASET.get('USE_DEPTH', False)

        # use IMU
        if self.use_imu:
            embed_dim = config["MODEL"]["IMU_BACKBONE"]["EMBED_DIM"]
            nhead = config["MODEL"]["IMU_BACKBONE"]["NHEAD"]
            num_layer = config["MODEL"]["IMU_BACKBONE"]["NUM_LAYER"]
            dropout_rate = config["MODEL"]["IMU_BACKBONE"].get("DROPOUT", 0)
            self.linear_embedding = nn.Linear(input_dim, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer) 

        # use image
        if self.use_image:
            self.img_backbone_cfg = config["MODEL"]["IMAGE_BACKBONE"] 
            # img_backbone_cfg = config.MODEL.IMAGE_BACKBONE
            self.img_backbone = build_backbone(config=self.img_backbone_cfg)
            self.img_feat_dim = config.MODEL.IMAGE_BACKBONE.EMBED_DIM

        # use depth image
        if self.use_depth:
            self.depth_backbone_cfg = config["MODEL"]["DEPTH_BACKBONE"] 
            # img_backbone_cfg = config.MODEL.IMAGE_BACKBONE
            self.depth_backbone = build_backbone(config=self.depth_backbone_cfg)
            self.depth_feat_dim = config.MODEL.DEPTH_BACKBONE.EMBED_DIM
            
        # Head
        output_head_cfg = config['MODEL']["OUTPUT_HEAD"]
        self.head_type = output_head_cfg["TYPE"]
        if output_head_cfg["TYPE"] == "mlp":
            feature_dim = output_head_cfg["FEATURE_DIM"]
            hidden_dim = output_head_cfg["HIDDEN_DIM"]
            dropout_rate = output_head_cfg["DROPOUT"]
            # embed_dim = embed_dim + img_feat_dim
            self.stabilizer = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.ReLU(),
                        # nn.Dropout(dropout_rate),
                        nn.Linear(hidden_dim, 17*3)
            )            
        elif output_head_cfg["TYPE"] == "transfromer_decoder":
            feature_dim = output_head_cfg["FEATURE_DIM"]
            decoder_dim = output_head_cfg["DECODER_DIM"]
            decoder_depth = output_head_cfg["DECODER_DEPTH"]
            
            self.stabilizer = OutputHeadTransformer(
                feature_dim=feature_dim,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth,
                num_feature_pos_enc=None,
                feature_mapping_mlp=False,
                queries="per_joint",
                joints_num=17,
            )

        self._initialize()
                    
        ### Class token
        # self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        inputs_image=None,
        input_imu=None, 
        input_depth=None):

        batch_size = input_imu.shape[0]

        # IMU feat
        if self.use_imu:
            x = self.linear_embedding(input_imu)
            x = self.transformer_encoder(x)  # No need to permute if batch_first=True
            imu_feat = x[:, -1, :]  # Select the last time step
        else:
            imu_feat = torch.empty(batch_size, 0).to(input_imu.device)

        # Image feat
        if self.use_image:
            curr_img = inputs_image[:, -1] # take current frame image  (N, T, C, H, W) -> (N, C, H, W)
            img_feat = self.img_backbone(curr_img)
            img_feat = einops.rearrange(img_feat, 'b c h w -> b (h w) c')
            img_feat = F.avg_pool1d(img_feat.permute(0, 2, 1), kernel_size=49).squeeze(-1)
        else:
            img_feat = torch.empty(batch_size, 0).to(input_imu.device)

        # Depth Image feat
        if self.use_depth:
            x = input_depth[:, -1] # take current frame image  (N, T, C, H, W) -> (N, C, H, W)
            x = self.depth_backbone(x)
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
            depth_feat = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=49).squeeze(-1)
        else:
            depth_feat = torch.empty(batch_size, 0).to(input_imu.device)

        ### fusion layer - imu + image
        # TODO: attention based fusion

        # fused_feat = torch.cat((imu_feat, img_feat), dim=1)
        fused_feat = torch.cat((imu_feat, img_feat, depth_feat), dim=1)

        # if self.head_type == "mlp":
        #     img_feat = F.avg_pool1d(img_feat.permute(0, 2, 1), kernel_size=49).squeeze(-1)
        #     if self.use_image and self.use_imu:
        #         fused_feat = torch.cat((imu_feat, img_feat), dim=1)
        #     elif self.use_imu:
        #         fused_feat = imu_feat
        #     elif self.use_image:
        #         fused_feat = img_feat
        # else:
        #     # TODO: implement fusion layer for transformer head
        #     img_feat = F.avg_pool1d(img_feat.permute(0, 2, 1), kernel_size=49).squeeze(-1)
        #     if self.use_image and self.use_imu:
        #         fused_feat = torch.cat((imu_feat, img_feat), dim=1)
        #     elif self.use_imu:
        #         fused_feat = imu_feat
        #     elif self.use_image:
        #         fused_feat = img_feat

        #     fused_feat = fused_feat.unsqueeze(1)

        # Output head
        output = self.stabilizer(fused_feat)
        output = output.view(-1, 17, 3)
        
        return output
        

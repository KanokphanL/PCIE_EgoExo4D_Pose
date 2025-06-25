import torch.nn as nn
import torch
import einops
import torch.nn.functional as F

from models.output_head_transformer import OutputHeadTransformer
from models.backbone.build_backbone import build_backbone

class EgoRGBDFusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        print("Initializing EgoRGBDFusion ...")
        
        input_dim = config["DATASET"]["INPUT_DIM"]

        self.use_imu = config["DATASET"]["USE_IMU"]
        self.use_image = False if config["DATASET"]["USE_IMAGE_MODE"] == 'none' else True
        self.use_depth = config.DATASET.get('USE_DEPTH', False)
        self.use_rgbd = True if self.use_image and self.use_depth else False

        # use IMU
        if self.use_imu:
            embed_dim = config["MODEL"]["IMU_BACKBONE"]["EMBED_DIM"]
            nhead = config["MODEL"]["IMU_BACKBONE"]["NHEAD"]
            num_layer = config["MODEL"]["IMU_BACKBONE"]["NUM_LAYER"]
            dropout_rate = config["MODEL"]["IMU_BACKBONE"].get("DROPOUT", 0)
            self.linear_embedding = nn.Linear(input_dim, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer) 

        self.temporal_fusion = config.MODEL.get("TEMPORAL_FUSION", "mean")
        
        # use rgbd
        if self.use_rgbd:
            self.img_backbone_cfg = config["MODEL"]["RGBD_IMAGE_BACKBONE"] 
            self.img_backbone = build_backbone(config=self.img_backbone_cfg)
            self.img_feat_dim = config.MODEL.RGBD_IMAGE_BACKBONE.EMBED_DIM

        if self.temporal_fusion == 'transformer_encoder':
            encoder_layer = nn.TransformerEncoderLayer(self.img_feat_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer) 
            
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
        input_image=None,
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
        if self.use_rgbd:
            img_feat_list = []
            _, T, _, _, _ = input_image.shape
            sliced_depth = input_depth[:, :, 0:1, :, :]
            input_rgbd = torch.cat((input_image, sliced_depth), dim=2)
            for i in range(T):
                x = input_rgbd[:, i]
                x = self.img_backbone(x)
                x = x.squeeze()
                img_feat_list.append(x)
            x = torch.stack(img_feat_list, dim=1)
            
            # temporal fusion - attention (transformer encoder)
            x = self.temporal_encoder(x)
            img_feat = x[:, -1, :]                
        else:
            img_feat = torch.empty(batch_size, 0).to(input_imu.device)

        ### fusion layer - imu + image + depth
        # TODO: attention based fusion
        fused_feat = torch.cat((imu_feat, img_feat), dim=1)

        # Output head
        output = self.stabilizer(fused_feat)
        output = output.view(-1, 17, 3)
        
        return output
        

from ast import Tuple
from typing import Optional
import torch
import torch.nn as nn



class OutputHeadTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        num_feature_pos_enc: Optional[int] = None,
        feature_mapping_mlp: bool = False,
        queries: str = "per_joint",
        joints_num: int = 21, # 21 joints for hand, 17 joints for body
    ):
        super().__init__()
        self.joints_num = joints_num
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_dim // 64,
            dim_feedforward=decoder_dim * 4,
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        if feature_mapping_mlp:
            self.feature_mapping = nn.Sequential(
                nn.Linear(feature_dim, decoder_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(decoder_dim, decoder_dim),
            )
        else:
            self.feature_mapping = nn.Linear(feature_dim, decoder_dim)

        self.queries = queries
        
        if queries== "single":
                self.embedding = nn.Parameter(torch.zeros(1, decoder_dim))

                self.head = nn.Linear(
                    # decoder_dim, 6 * 16 * 2 + 3 * 2 + 10 * 2 + 3 + 3 + 1 #表示右手和左手的姿态信息，其中 6 表示每个关节的旋转参数维度，16 表示手部的关节数，2 表示右手和左手 #表示右手和左手的根位置信息，其中 3 表示三维坐标，2 表示右手和左手。
                    decoder_dim,  joints_num * 3 
    )                                                                    
        elif queries== "per_joint":
                self.embedding = nn.Parameter(torch.randn(joints_num, decoder_dim))
                self.pose_r_head = nn.Linear(decoder_dim,  3) 
        elif queries== "per_joint_rle":
                self.embedding = nn.Parameter(torch.randn(joints_num, decoder_dim))
                self.pose_r_head = nn.Linear(decoder_dim,  3 *2) 
        else:
                raise ValueError(f"Unknown query type {queries}")
        self.feature_pos_enc = (
            nn.Parameter(torch.randn(1, num_feature_pos_enc, decoder_dim))
            if num_feature_pos_enc is not None
            else None
        )

    def forward(self, features):
        B = features.shape[0]

        if features.dim() == 4:
            context = self.feature_mapping(
                features.reshape(B, features.shape[1], -1).transpose(1, 2) #[B, C, H* W]-->[B, H* W, C] #  [B, 49, 512]
            ) 
        else: 
            context = self.feature_mapping(features)
            
        if self.feature_pos_enc is not None:
            context = context + self.feature_pos_enc 
        x = self.embedding.expand(B, -1, -1) # [B, 1, 512]
        out = self.decoder(x, context)

        if self.queries== "single":
                out = self.head(out[:, 0])  # B, decoder_dim
                out = out.view(B, self.joints_num, 3)
        elif self.queries=="per_joint":
                out = self.pose_r_head(out)   
        elif self.queries=="per_joint_rle":
                out = self.pose_r_head(out)   
        else:
                assert False
        return out      
       
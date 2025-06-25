import torch.nn as nn
import torch

from models.output_head_transformer import OutputHeadTransformer

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super(CustomTransformerEncoderLayer, self).__init__()
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


class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init__()
        
        input_dim = config["DATASET"]["INPUT_DIM"]
        embed_dim = config["MODEL"]["BASELINE"]["EMBED_DIM"]
        nhead = config["MODEL"]["BASELINE"]["NHEAD"]
        num_layer = config["MODEL"]["BASELINE"]["NUM_LAYER"]
        dropout_rate = config["MODEL"]["BASELINE"].get("DROPOUT", 0)

        self.linear_embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        if config.get("SELECT_OUTPUT_HEAD") == None:
            # compatible with old config that does not have SELECT_OUTPUT_HEAD
            config["SELECT_OUTPUT_HEAD"] = "MLP"
            config["OUTPUT_HEAD"] = {
                "MLP": {
                    "HIDDEN_DIM": 256,
                },
            }
            
        output_head = config["OUTPUT_HEAD"][config["SELECT_OUTPUT_HEAD"]]
        if config["SELECT_OUTPUT_HEAD"] == "MLP":
            hidden_dim = output_head["HIDDEN_DIM"]
            
            self.stabilizer = nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 17*3)
            )
            
        elif config["SELECT_OUTPUT_HEAD"] == "TRANSFORMER_DECODER":
            decoder_dim = output_head["EMBED_DIM"]
            decoder_depth = output_head["NUM_LAYER"]
            
            self.stabilizer = OutputHeadTransformer(
                feature_dim=embed_dim,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth,
                num_feature_pos_enc=None,
                feature_mapping_mlp=False,
                queries="per_joint",
                joints_num=17,
            )
            
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, input_tensor):
        x = self.linear_embedding(input_tensor)
        
        # Add class token to the input sequence
        batch_size = x.size(0)

        # TODO: cls_token needed?
        # class_token = self.class_token.expand(batch_size, -1, -1)  # Expand class token to match batch size
        # x = torch.cat((class_token, x), dim=1)  # Concatenate class token to the input sequence
        
        x = self.transformer_encoder(x)  # No need to permute if batch_first=True
        feature_token = x[:, -1, :]  # Select the last time step

        # Output head
        output = self.stabilizer(feature_token)
        output = output.view(-1, 17, 3)
        
        return output
        

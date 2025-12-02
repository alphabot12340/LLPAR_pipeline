import math
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from models_CNNHELPERS.vit import *
from models_CNNHELPERS.resnet18_student import ResNet18Student
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=768, pretrain_path='checkpoints/jx_vit_base_p16_224-80ecf9dd.pth', use_resnet18=True):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(768, dim)

        self.vit = vit_base()
        if pretrain_path and os.path.exists(pretrain_path):
            self.vit.load_param(pretrain_path)
        elif pretrain_path:
            print(f"Warning: ViT pretrain path '{pretrain_path}' not found. Assuming weights will be loaded from checkpoint.")
        
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        
        if use_resnet18:
            # Use ResNet18 as student model
            self.student_model = ResNet18Student(
                attr_num=attr_num,
                feature_dim=dim,
                img_size_h=256,
                img_size_w=128
            )
        else:
            # Original SNN student model
            # Note: import * not allowed in function, but create_model is from timm
            self.student_model = create_model(
                'Spikingformer',
                pretrained=False,
                drop_rate=0.,
                drop_path_rate=0.2,
                drop_block_rate=None,
                img_size_h=256, img_size_w=128,
                patch_size=16, embed_dims=768, num_heads=8, mlp_ratios=4,
                in_channels=3, num_classes=26, qkv_bias=False,
                depths=8, sr_ratios=1,
                T=8,
            )

    def forward(self, imgs, word_vec, label=None):
        features = self.vit(imgs)
        S_train_logits, S_features = self.student_model(imgs)
        
        word_embed = self.word_embed(word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        
        tex_embed = word_embed + self.tex_embed
        vis_embed = features + self.vis_embed

        teacher_tokens = vis_embed[:, 1:, :]
        if S_features.shape[1] != teacher_tokens.shape[1]:
            S_features = F.interpolate(
                S_features.permute(0, 2, 1),
                size=teacher_tokens.shape[1],
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        x = torch.cat([tex_embed, vis_embed], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        T_train_logits = self.bn(logits)


        return T_train_logits,vis_embed[:,1:,:],tex_embed,S_train_logits, S_features
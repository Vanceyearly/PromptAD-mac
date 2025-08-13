import torch
import random
import torch.nn as nn
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image
from scipy.ndimage import gaussian_filter

from .CLIPAD import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()   # local tokenizer, no padding, no sos, no eos

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert('RGB')


class CrossModalAttention(nn.Module):
    """跨模态注意力融合模块"""
    def __init__(self, visual_dim, text_dim, hidden_dim=256, num_heads=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 将视觉和文本特征投影到相同维度
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=False  # 使用 seq_len, batch, embed_dim 格式
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, visual_features, text_features):
        # 确保输入是3D张量 [B, N, D]
        if len(visual_features.shape) == 2:
            visual_features = visual_features.unsqueeze(1)  # [B, D] -> [B, 1, D]
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)      # [B, D] -> [B, 1, D]
        
        # 投影到相同维度
        v_proj = self.visual_proj(visual_features)  # [B, N, hidden_dim]
        t_proj = self.text_proj(text_features)      # [B, M, hidden_dim]
        
        # 转换为 [N, B, D] 格式 (batch_first=False)
        v_proj = v_proj.transpose(0, 1)  # [N, B, hidden_dim]
        t_proj = t_proj.transpose(0, 1)  # [M, B, hidden_dim]
        
        # 跨模态注意力：视觉特征作为query，文本特征作为key和value
        v_attended, _ = self.multihead_attn(v_proj, t_proj, t_proj)
        v_attended = self.layer_norm1(v_proj + v_attended)
        
        # 前馈网络
        v_ff = self.ffn(v_attended)
        v_output = self.layer_norm2(v_attended + v_ff)
        
        # 跨模态注意力：文本特征作为query，视觉特征作为key和value
        t_attended, _ = self.multihead_attn(t_proj, v_proj, v_proj)
        t_attended = self.layer_norm1(t_proj + t_attended)
        
        # 前馈网络
        t_ff = self.ffn(t_attended)
        t_output = self.layer_norm2(t_attended + t_ff)
        
        # 转换回 [B, N, D] 格式
        v_output = v_output.transpose(0, 1)
        t_output = t_output.transpose(0, 1)
        
        return v_output, t_output


class FusionWeightNetwork(nn.Module):
    """特征融合权重学习网络"""
    def __init__(self, visual_dim, text_dim, hidden_dim=128):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        
        # 权重预测网络
        self.weight_net = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 输出两个权重：视觉权重和文本权重
            nn.Softmax(dim=-1)
        )
        
    def forward(self, visual_features, text_features):
        # 全局平均池化获得特征表示
        if len(visual_features.shape) > 2:
            v_global = visual_features.mean(dim=1)  # [B, visual_dim]
        else:
            v_global = visual_features
            
        if len(text_features.shape) > 2:
            t_global = text_features.mean(dim=1)    # [B, text_dim]
        else:
            t_global = text_features
            
        # 拼接特征
        combined = torch.cat([v_global, t_global], dim=-1)  # [B, visual_dim + text_dim]
        
        # 预测融合权重
        weights = self.weight_net(combined)  # [B, 2]
        
        return weights[:, 0:1], weights[:, 1:2]  # 返回视觉权重和文本权重


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre):
        super().__init__()

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        state_anomaly1 = state_anomaly + class_state_abnormal[classname]

        if classname in class_mapping:
            classname = class_mapping[classname]

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_handle = len(state_anomaly1)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_handle

        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_handle = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_handle])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = clip_model.token_embedding(tokenized_abnormal_prompts_handle).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_handle, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_handle
        # self.name_lens = name_lens

    def forward(self):

        # learned normal prompt
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # handle abnormal prompt
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_handle = self.abnormal_token_prefix_handle
        abnormal_suffix_handle = self.abnormal_token_suffix_handle

        abnormal_prompts_handle = torch.cat(
            [
                abnormal_prefix_handle,     # (n_pro * n_ab_handle, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_handle, n_ctx, dim)
                abnormal_suffix_handle,     # (n_pro * n_ab_handle, *, dim)
            ],
            dim=1,
        )

        # learned abnormal prompt
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_handle, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_handle

        return normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned


class PromptAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(PromptAD, self).__init__()

        self.shot = kwargs['k_shot']

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
        self.phrase_form = '{}'
        self.device = device

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual
        
        # 添加可学习的视觉-文本融合模块
        self.enable_fusion = kwargs.get('enable_fusion', True)
        if self.enable_fusion:
            self._init_fusion_modules()

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision)
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)
    
    def _init_fusion_modules(self):
        """初始化视觉-文本融合模块"""
        # 根据实际测试，视觉特征维度情况：
        # - pooled特征: 640维
        # - tokens特征: 640维 (feature1) 和 896维 (feature2, feature3)
        # - 文本特征: 640维
        text_dim = 640  # 文本特征维度
        
        print(f"初始化融合模块 - 文本维度: {text_dim}")
        
        # 为不同维度的视觉特征创建不同的跨模态注意力模块
        # 640维特征的融合模块 (用于pooled和tokens特征)
        self.cross_modal_attention_640 = CrossModalAttention(
            visual_dim=640,
            text_dim=text_dim,
            hidden_dim=256,
            num_heads=8
        ).to(self.device)
        
        # 896维特征的融合模块 (用于feature2和feature3)
        self.cross_modal_attention_896 = CrossModalAttention(
            visual_dim=896,
            text_dim=text_dim,
            hidden_dim=256,
            num_heads=8
        ).to(self.device)
        
        # 特征融合权重学习模块 (为不同维度创建)
        self.fusion_weight_net_640 = FusionWeightNetwork(
            visual_dim=640,
            text_dim=text_dim,
            hidden_dim=128
        ).to(self.device)
        
        self.fusion_weight_net_896 = FusionWeightNetwork(
            visual_dim=896,
            text_dim=text_dim,
            hidden_dim=128
        ).to(self.device)
        
        # 融合后的特征投影层 (注意：CrossModalAttention输出256维，所以拼接后是512维)
        self.fused_projection_640 = nn.Linear(256 + 256, 640).to(self.device)  # 256+256=512 -> 640
        self.fused_projection_896 = nn.Linear(256 + 256, 896).to(self.device)  # 256+256=512 -> 896
        
        if self.precision == 'fp16':
            self.cross_modal_attention_640 = self.cross_modal_attention_640.half()
            self.cross_modal_attention_896 = self.cross_modal_attention_896.half()
            self.fusion_weight_net_640 = self.fusion_weight_net_640.half()
            self.fusion_weight_net_896 = self.fusion_weight_net_896.half()
            self.fused_projection_640 = self.fused_projection_640.half()
            self.fused_projection_896 = self.fused_projection_896.half()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]
    
    @torch.no_grad()
    def encode_fused_features(self, image: torch.Tensor, use_fusion=True):
        """使用视觉-文本融合的特征编码方法"""
        if not self.enable_fusion or not use_fusion:
            return self.encode_image(image)
            
        if self.precision == "fp16":
            image = image.half()
            
        # 获取原始视觉特征
        visual_features = self.model.encode_image(image)  # [pooled, tokens, feature1, feature2]
        
        # 获取文本特征（正常和异常）
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        
        # 编码文本特征
        normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
        abnormal_text_features_handle = self.encode_text_embedding(abnormal_text_embeddings_handle, self.tokenized_abnormal_prompts_handle)
        abnormal_text_features_learned = self.encode_text_embedding(abnormal_text_embeddings_learned, self.tokenized_abnormal_prompts_learned)
        
        # 合并异常文本特征
        abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
        
        # 对每个视觉特征层进行融合
        fused_features = []
        
        for i, v_feat in enumerate(visual_features):
            if i == 0:  # pooled features
                # 使用正常文本特征进行融合
                text_feat = normal_text_features.mean(dim=0, keepdim=True)  # 平均池化多个prompt
                
                # 扩展文本特征维度以匹配视觉特征
                if len(v_feat.shape) == 3:  # [B, N, D]
                    text_feat = text_feat.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                elif len(v_feat.shape) == 2:  # [B, D]
                    text_feat = text_feat.expand(v_feat.shape[0], -1)
                
                # 根据特征维度选择对应的融合模块
                feat_dim = v_feat.shape[-1]
                if feat_dim == 640:
                    cross_attention = self.cross_modal_attention_640
                    fusion_weight_net = self.fusion_weight_net_640
                    projection = self.fused_projection_640
                elif feat_dim == 896:
                    cross_attention = self.cross_modal_attention_896
                    fusion_weight_net = self.fusion_weight_net_896
                    projection = self.fused_projection_896
                else:
                    raise ValueError(f"不支持的特征维度: {feat_dim}")
                    
                # 跨模态注意力融合
                v_fused, t_fused = cross_attention(v_feat.unsqueeze(1) if len(v_feat.shape) == 2 else v_feat, 
                                                 text_feat.unsqueeze(1) if len(text_feat.shape) == 2 else text_feat)
                
                # 学习融合权重
                v_weight, t_weight = fusion_weight_net(v_feat, text_feat)
                
                # 处理注意力输出维度
                if len(v_feat.shape) == 2:
                    v_fused = v_fused.squeeze(1)
                    t_fused = t_fused.squeeze(1)
                else:
                    # 对于3D特征，取平均
                    v_fused = v_fused.mean(1)
                    t_fused = t_fused.mean(1)
                
                # 特征拼接和投影
                combined_feat = torch.cat([v_fused, t_fused], dim=-1)
                projected_feat = projection(combined_feat)
                
                # 应用融合权重
                if len(v_feat.shape) == 3:  # 对于tokens特征，需要扩展维度
                    projected_feat = projected_feat.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                    v_weight = v_weight.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                    t_weight = t_weight.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                
                fused_feat = v_weight * projected_feat + (1 - v_weight - t_weight) * v_feat
                    
            else:  # tokens, feature1, feature2
                # 对于其他特征，使用异常文本特征进行融合
                text_feat = abnormal_text_features.mean(dim=0, keepdim=True)
                
                # 扩展文本特征到batch维度，但保持2D用于注意力计算
                text_feat = text_feat.expand(v_feat.shape[0], -1)  # [B, D]
                
                # 根据特征维度选择对应的融合模块
                feat_dim = v_feat.shape[-1]
                if feat_dim == 640:
                    cross_attention = self.cross_modal_attention_640
                    fusion_weight_net = self.fusion_weight_net_640
                    projection = self.fused_projection_640
                elif feat_dim == 896:
                    cross_attention = self.cross_modal_attention_896
                    fusion_weight_net = self.fusion_weight_net_896
                    projection = self.fused_projection_896
                else:
                    raise ValueError(f"不支持的特征维度: {feat_dim}")
                    
                # 跨模态注意力融合
                v_fused, t_fused = cross_attention(v_feat,  # 保持原始形状
                                                 text_feat)  # [B, D]
                
                # 学习融合权重
                v_weight, t_weight = fusion_weight_net(v_feat, text_feat)
                
                # 处理注意力输出维度
                if len(v_feat.shape) == 2:
                    v_fused = v_fused.squeeze(1)
                    t_fused = t_fused.squeeze(1)
                else:
                    # 对于3D特征，取平均
                    v_fused = v_fused.mean(1)
                    t_fused = t_fused.mean(1)
                
                # 特征拼接和投影
                combined_feat = torch.cat([v_fused, t_fused], dim=-1)
                projected_feat = projection(combined_feat)
                
                # 应用融合权重
                if len(v_feat.shape) == 3:  # 对于tokens特征，需要扩展维度
                    projected_feat = projected_feat.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                    v_weight = v_weight.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                    t_weight = t_weight.unsqueeze(1).expand(-1, v_feat.shape[1], -1)
                
                fused_feat = v_weight * projected_feat + (1 - v_weight - t_weight) * v_feat
            
            # 归一化
            fused_feat = fused_feat / fused_feat.norm(dim=-1, keepdim=True)
            fused_features.append(fused_feat)
            
        return fused_features

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        self.feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1))

        b2, n2, d2 = features2.shape
        self.feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0]
            global_normality_and_abnormality_score = (t * global_feature @ self.text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[1].shape[0]

        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def forward(self, images, task, use_fusion=None):
        """前向传播，支持融合特征"""
        # 如果未指定use_fusion，则根据enable_fusion决定
        if use_fusion is None:
            use_fusion = self.enable_fusion
            
        # 使用融合特征编码或传统编码
        if use_fusion and self.enable_fusion:
            visual_features = self.encode_fused_features(images, use_fusion=True)
        else:
            visual_features = self.encode_image(images)
            
        if task == 'seg':
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            #
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
            # anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            # anomaly_map = visual_anomaly_map
            # anomaly_map = textual_anomaly_map

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            am_img_list = []
            for i in range(textual_anomaly.shape[0]):
                am_img_list.append(textual_anomaly[i])

            return am_img_list, am_pix_list
        else:
            assert 'task error'

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

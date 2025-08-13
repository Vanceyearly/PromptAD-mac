#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试融合功能的维度问题
"""

import torch
from PromptAD import PromptAD

def debug_fusion_dimensions():
    """调试融合功能的维度问题"""
    # 设置基本参数
    args = {
        'dataset': 'mvtec',
        'class_name': 'carpet',
        'img_resize': 240,
        'img_cropsize': 240,
        'resolution': 400,
        'batch_size': 4,
        'k_shot': 1,
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 4,
        'n_ctx_ab': 1,
        'n_pro': 1,
        'n_pro_ab': 4,
        'enable_fusion': True,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'out_size_h': 400,
        'out_size_w': 400,
        'load_memory': False,
        'version': ''
    }
    
    print(f"使用设备: {args['device']}")
    
    try:
        # 创建模型
        print("正在创建PromptAD模型...")
        model = PromptAD(**args)
        model = model.to(args['device'])
        model.eval()
        
        # 创建测试数据
        batch_size = 2
        images = torch.randn(batch_size, 3, 240, 240).to(args['device'])
        if model.precision == 'fp16':
            images = images.half()
        print(f"\n测试图像形状: {images.shape}, dtype: {images.dtype}")
        
        # 获取原始视觉特征
        print("\n=== 获取原始视觉特征 ===")
        visual_features = model.model.encode_image(images)
        for i, feat in enumerate(visual_features):
            print(f"视觉特征 {i}: {feat.shape}")
        
        # 获取文本特征
        print("\n=== 获取文本特征 ===")
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = model.prompt_learner()
        print(f"normal_text_embeddings: {normal_text_embeddings.shape}")
        print(f"abnormal_text_embeddings_handle: {abnormal_text_embeddings_handle.shape}")
        print(f"abnormal_text_embeddings_learned: {abnormal_text_embeddings_learned.shape}")
        
        # 编码文本特征
        normal_text_features = model.encode_text_embedding(normal_text_embeddings, model.tokenized_normal_prompts)
        abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_embeddings_handle, model.tokenized_abnormal_prompts_handle)
        abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_embeddings_learned, model.tokenized_abnormal_prompts_learned)
        
        print(f"normal_text_features: {normal_text_features.shape}")
        print(f"abnormal_text_features_handle: {abnormal_text_features_handle.shape}")
        print(f"abnormal_text_features_learned: {abnormal_text_features_learned.shape}")
        
        # 合并异常文本特征
        abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
        print(f"合并后的abnormal_text_features: {abnormal_text_features.shape}")
        
        # 测试第一个特征（pooled）的融合
        print("\n=== 测试pooled特征融合 ===")
        v_feat = visual_features[0]  # pooled特征
        print(f"v_feat形状: {v_feat.shape}")
        
        text_feat = normal_text_features.mean(dim=0, keepdim=True)
        print(f"text_feat形状: {text_feat.shape}")
        
        # 扩展文本特征
        if len(v_feat.shape) == 2:  # [B, D]
            text_feat = text_feat.expand(v_feat.shape[0], -1)
        print(f"扩展后text_feat形状: {text_feat.shape}")
        
        # 选择融合模块
        feat_dim = v_feat.shape[-1]
        print(f"特征维度: {feat_dim}")
        
        if feat_dim == 640:
            cross_attention = model.cross_modal_attention_640
            fusion_weight_net = model.fusion_weight_net_640
            projection = model.fused_projection_640
        elif feat_dim == 896:
            cross_attention = model.cross_modal_attention_896
            fusion_weight_net = model.fusion_weight_net_896
            projection = model.fused_projection_896
        
        print(f"投影层输入维度: {projection.in_features}")
        print(f"投影层输出维度: {projection.out_features}")
        
        # 跨模态注意力融合
        print("\n=== 跨模态注意力融合 ===")
        v_input = v_feat.unsqueeze(1) if len(v_feat.shape) == 2 else v_feat
        t_input = text_feat.unsqueeze(1) if len(text_feat.shape) == 2 else text_feat
        print(f"注意力输入 - v_input: {v_input.shape}, t_input: {t_input.shape}")
        
        v_fused, t_fused = cross_attention(v_input, t_input)
        print(f"注意力输出 - v_fused: {v_fused.shape}, t_fused: {t_fused.shape}")
        
        # 处理注意力输出维度
        if len(v_feat.shape) == 2:
            v_fused = v_fused.squeeze(1)
            t_fused = t_fused.squeeze(1)
        else:
            v_fused = v_fused.mean(1)
            t_fused = t_fused.mean(1)
        
        print(f"处理后 - v_fused: {v_fused.shape}, t_fused: {t_fused.shape}")
        
        # 特征拼接
        combined_feat = torch.cat([v_fused, t_fused], dim=-1)
        print(f"拼接后特征: {combined_feat.shape}")
        
        print("\n✅ 调试完成！")
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    print("开始调试融合功能维度...")
    success = debug_fusion_dimensions()
    if success:
        print("\n🎉 调试完成！")
    else:
        print("\n💥 调试失败。")
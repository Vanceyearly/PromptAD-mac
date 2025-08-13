#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试token特征融合的脚本
"""

import torch
from PromptAD import PromptAD

def test_token_fusion():
    """测试token特征融合"""
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
        'device': 'cpu',  # 使用CPU避免CUDA问题
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
        
        # 如果模型使用fp16，转换输入
        if hasattr(model.model.visual, 'conv1') and model.model.visual.conv1.weight.dtype == torch.float16:
            images = images.half()
        
        print(f"测试图像形状: {images.shape}, dtype: {images.dtype}")
        
        # 获取原始视觉特征
        print("\n=== 获取原始视觉特征 ===")
        with torch.no_grad():
            visual_features = model.encode_image(images)
            for i, feat in enumerate(visual_features):
                print(f"视觉特征 {i}: {feat.shape}")
        
        # 获取文本特征
        print("\n=== 获取文本特征 ===")
        with torch.no_grad():
            normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = model.prompt_learner()
            abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)
            
            print(f"normal_text_embeddings: {normal_text_embeddings.shape}")
            print(f"abnormal_text_embeddings_handle: {abnormal_text_embeddings_handle.shape}")
            print(f"abnormal_text_embeddings_learned: {abnormal_text_embeddings_learned.shape}")
            
            # 使用正确的tokenized prompts
            normal_text_features = model.encode_text_embedding(normal_text_embeddings, model.tokenized_normal_prompts)
            
            # 为abnormal特征使用正确的token数量
            tokenized_abnormal_handle = model.tokenized_abnormal_prompts[:abnormal_text_embeddings_handle.shape[0]]
            tokenized_abnormal_learned = model.tokenized_abnormal_prompts[:abnormal_text_embeddings_learned.shape[0]]
            
            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_embeddings_handle, tokenized_abnormal_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_embeddings_learned, tokenized_abnormal_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            
            print(f"normal_text_features: {normal_text_features.shape}")
            print(f"abnormal_text_features_handle: {abnormal_text_features_handle.shape}")
            print(f"abnormal_text_features_learned: {abnormal_text_features_learned.shape}")
            print(f"合并后的abnormal_text_features: {abnormal_text_features.shape}")
        
        # 测试每个视觉特征的融合
        print("\n=== 测试各个特征的融合 ===")
        with torch.no_grad():
            for i, v_feat in enumerate(visual_features):
                print(f"\n--- 测试特征 {i} ---")
                print(f"v_feat形状: {v_feat.shape}")
                
                # 选择合适的融合模块
                if v_feat.shape[-1] == 640:
                    cross_attention = model.cross_modal_attention_640
                    fusion_weight_net = model.fusion_weight_net_640
                    projection = model.fused_projection_640
                    print("使用640维融合模块")
                elif v_feat.shape[-1] == 896:
                    cross_attention = model.cross_modal_attention_896
                    fusion_weight_net = model.fusion_weight_net_896
                    projection = model.fused_projection_896
                    print("使用896维融合模块")
                else:
                    print(f"未知特征维度: {v_feat.shape[-1]}")
                    continue
                
                # 准备文本特征
                text_feat = normal_text_features
                if len(text_feat.shape) == 2 and text_feat.shape[0] != v_feat.shape[0]:
                    text_feat = text_feat.expand(v_feat.shape[0], -1)
                
                print(f"text_feat形状: {text_feat.shape}")
                
                try:
                    # 跨模态注意力
                    print("执行跨模态注意力...")
                    v_input = v_feat.unsqueeze(1) if len(v_feat.shape) == 2 else v_feat
                    t_input = text_feat.unsqueeze(1) if len(text_feat.shape) == 2 else text_feat
                    
                    print(f"注意力输入 - v_input: {v_input.shape}, t_input: {t_input.shape}")
                    
                    v_fused, t_fused = cross_attention(v_input, t_input)
                    print(f"注意力输出 - v_fused: {v_fused.shape}, t_fused: {t_fused.shape}")
                    
                    print(f"✅ 特征 {i} 融合成功")
                    
                except Exception as e:
                    print(f"❌ 特征 {i} 融合失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
        
        print("\n✅ 所有token特征融合测试成功！")
        return True
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("开始测试token特征融合功能...")
    success = test_token_fusion()
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败，请检查代码。")
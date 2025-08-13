#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型特征维度的脚本
"""

import torch
from PromptAD import PromptAD

def check_model_dimensions():
    """检查模型的实际特征维度"""
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
        'enable_fusion': False,  # 先禁用融合
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
        
        # 检查模型属性
        print("\n=== 模型维度信息 ===")
        if hasattr(model.model, 'visual'):
            visual = model.model.visual
            print(f"visual.embed_dim: {getattr(visual, 'embed_dim', 'N/A')}")
            print(f"visual.width: {getattr(visual, 'width', 'N/A')}")
            print(f"visual.output_dim: {getattr(visual, 'output_dim', 'N/A')}")
            
        if hasattr(model.model, 'text_projection'):
            print(f"text_projection.shape: {model.model.text_projection.shape}")
            
        # 创建测试数据
        print("\n=== 测试特征维度 ===")
        batch_size = 2
        images = torch.randn(batch_size, 3, 240, 240).to(args['device'])
        
        # 测试图像编码
        with torch.no_grad():
            features = model.encode_image(images)
            print(f"encode_image返回特征数量: {len(features)}")
            for i, feat in enumerate(features):
                if feat is not None:
                    print(f"  特征 {i}: {feat.shape}")
                else:
                    print(f"  特征 {i}: None")
        
        # 测试文本编码
        print("\n=== 测试文本特征 ===")
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = model.prompt_learner()
        print(f"normal_text_embeddings: {normal_text_embeddings.shape}")
        print(f"abnormal_text_embeddings_handle: {abnormal_text_embeddings_handle.shape}")
        print(f"abnormal_text_embeddings_learned: {abnormal_text_embeddings_learned.shape}")
        
        # 编码文本特征
        normal_text_features = model.encode_text_embedding(normal_text_embeddings, model.tokenized_normal_prompts)
        print(f"normal_text_features: {normal_text_features.shape}")
        
        print("\n✅ 维度检查完成！")
        
    except Exception as e:
        print(f"❌ 检查过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    print("开始检查模型维度...")
    success = check_model_dimensions()
    if success:
        print("\n🎉 维度检查完成！")
    else:
        print("\n💥 维度检查失败。")
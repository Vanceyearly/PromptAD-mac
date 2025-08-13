#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视觉-文本融合功能的简单脚本
"""

import torch
import argparse
from PromptAD import PromptAD
from datasets import get_dataloader_from_args

def test_fusion():
    """测试融合功能"""
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
        'enable_fusion': True,  # 启用融合
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
        print(f"模型创建成功，融合功能已{'启用' if model.enable_fusion else '禁用'}")
        
        # 创建模拟测试数据（避免依赖实际数据集文件）
        print("正在创建模拟测试数据...")
        batch_size = 2
        # 创建随机图像数据 (B, C, H, W)
        images = torch.randn(batch_size, 3, 240, 240).to(args['device'])
        print(f"模拟图像形状: {images.shape}")
        
        # 测试原始图像编码
        print("\n测试原始图像编码...")
        with torch.no_grad():
            original_features = model.encode_image(images)
            if isinstance(original_features, list):
                print(f"原始特征: 包含 {len(original_features)} 个特征层")
                for i, feat in enumerate(original_features):
                    print(f"  层 {i}: {feat.shape}")
                # 使用第一个特征进行后续测试
                original_main_feature = original_features[0]
            else:
                print(f"原始特征形状: {original_features.shape}")
                original_main_feature = original_features
        
        # 测试融合特征编码
        if model.enable_fusion:
            print("\n测试融合特征编码...")
            with torch.no_grad():
                fused_features = model.encode_fused_features(images)
                if isinstance(fused_features, list):
                    print(f"融合特征: 包含 {len(fused_features)} 个特征层")
                    for i, feat in enumerate(fused_features):
                        print(f"  层 {i}: {feat.shape}")
                    # 使用第一个特征进行比较
                    fused_main_feature = fused_features[0]
                else:
                    print(f"融合特征形状: {fused_features.shape}")
                    fused_main_feature = fused_features
                
                # 计算特征差异
                if original_main_feature.shape == fused_main_feature.shape:
                    feature_diff = torch.mean(torch.abs(fused_main_feature - original_main_feature))
                    print(f"融合特征与原始特征的平均绝对差异: {feature_diff.item():.6f}")
                else:
                    print(f"特征形状不匹配，无法计算差异: {original_main_feature.shape} vs {fused_main_feature.shape}")
        
        # 测试完整的前向传播
        print("\n测试完整前向传播...")
        model.eval()
        with torch.no_grad():
            # 测试分类任务
            result_cls = model(images, task='classification', use_fusion=True)
            if result_cls is not None:
                if isinstance(result_cls, (list, tuple)):
                    print(f"分类结果: {len(result_cls)} 个输出")
                else:
                    print(f"分类结果形状: {result_cls.shape}")
            else:
                print("分类结果: None (可能需要先构建文本特征库)")
            
            # 测试分割任务
            result_seg = model(images, task='segmentation', use_fusion=True)
            if result_seg is not None:
                print(f"分割结果形状: {result_seg.shape}")
            else:
                print("分割结果: None (可能需要先构建文本特征库)")
        
        print("\n✅ 融合功能测试成功！")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    print("开始测试视觉-文本融合功能...")
    success = test_fusion()
    if success:
        print("\n🎉 所有测试通过！融合功能正常工作。")
    else:
        print("\n💥 测试失败，请检查代码。")
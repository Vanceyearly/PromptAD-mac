# 分类指标使用说明

## 概述

现在代码支持额外的分类指标计算，同时保持原有功能不变。您可以通过命令行参数来启用详细的分类指标。

## 新增功能

### 1. 新增指标
- **ROC AUC** (原有)
- **Accuracy** (准确率)
- **Precision** (精确率)  
- **Recall** (召回率)
- **F1-Score** (F1分数)
- **PR AUC** (Precision-Recall曲线下面积)
- **混淆矩阵** (TP, FP, TN, FN)
- **最佳阈值** (基于F1-Score优化)

### 2. 新增配置参数

```bash
--use-classification-metrics [True/False]  # 是否启用详细分类指标 (默认: False)
--model-selection-metric [roc_auc/f1_score/accuracy]  # 模型选择指标 (默认: roc_auc)
```

## 使用方法

### 基础使用 (保持原有功能)
```bash
python train_cls.py --dataset mvtec --class_name carpet
```

### 启用分类指标
```bash
python train_cls.py --dataset mvtec --class_name carpet \
    --use-classification-metrics True
```

### 使用F1-Score进行模型选择
```bash
python train_cls.py --dataset mvtec --class_name carpet \
    --use-classification-metrics True \
    --model-selection-metric f1_score
```

### 使用Accuracy进行模型选择
```bash
python train_cls.py --dataset mvtec --class_name carpet \
    --use-classification-metrics True \
    --model-selection-metric accuracy
```

## 输出示例

### 训练过程中的输出
```
Epoch 0 - Classification Metrics:
  ROC AUC: 95.23%
  Accuracy: 89.45%
  Precision: 91.20%
  Recall: 87.30%
  F1-Score: 89.21%
  PR AUC: 92.15%
  Best Threshold: 0.4523
  TP:45, FP:8, TN:102, FN:12
```

### 最终结果输出
```
Object:carpet =========================== Image-AUROC:95.23

============= Detailed Classification Results for carpet =============
ROC AUC: 95.23%
Accuracy: 89.45%
Precision: 91.20%
Recall: 87.30%
F1-Score: 89.21%
PR AUC: 92.15%
Best Threshold: 0.4523
Confusion Matrix - TP:45, FP:8, TN:102, FN:12
======================================================================
```

## 代码结构说明

### 新增函数
- `metric_cal_img()`: 原始函数，只计算ROC AUC (保持不变)
- `metric_cal_img_classification()`: 新函数，计算所有分类指标

### 条件逻辑
- 只有当 `--use-classification-metrics True` 时才会计算额外指标
- 模型选择策略可以通过 `--model-selection-metric` 配置
- 所有原有功能完全保持不变

## 注意事项

1. **向后兼容**: 不传递新参数时，代码行为与原来完全一致
2. **性能影响**: 启用分类指标会增加少量计算开销
3. **模型选择**: 建议根据任务特点选择合适的指标
   - **不平衡数据集**: 推荐使用 `f1_score` 或 `pr_auc`
   - **平衡数据集**: 可以使用 `accuracy`
   - **保持原有行为**: 使用 `roc_auc` (默认)

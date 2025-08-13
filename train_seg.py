import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'SEG'

def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)


def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    features1 = []
    features2 = []
    for (data, mask, label, name, img_type) in train_data:
        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]

        data = torch.stack(data, dim=0).to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    print(f"开始训练，总共 {args.Epoch} 个epoch")
    
    for epoch in range(args.Epoch):
        print(f"\n=== Epoch {epoch+1}/{args.Epoch} ===")
        
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        epoch_v2t_loss = 0.0
        epoch_trip_loss = 0.0
        epoch_fusion_loss = 0.0
        batch_count = 0
        
        # 使用tqdm显示训练进度
        train_pbar = tqdm(train_data, desc=f"训练 Epoch {epoch+1}", leave=False)
        
        for (data, mask, label, name, img_type) in train_pbar:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            # 获取原始视觉特征和融合特征
            _, feature_map, _, _ = model.encode_image(data)
            
            # 计算融合损失（如果启用融合）
            fusion_loss = 0.0
            if hasattr(model, 'enable_fusion') and model.enable_fusion:
                # 获取融合特征
                fused_features = model.encode_fused_features(data, use_fusion=True)
                _, fused_feature_map, _, _ = fused_features
                
                # 融合一致性损失：确保融合特征与原始特征保持相关性
                fusion_consistency_loss = F.mse_loss(fused_feature_map, feature_map.detach())
                
                # 融合判别损失：融合特征应该更好地区分正常和异常
                fused_normal_sim = torch.einsum('nic,c->ni', fused_feature_map, normal_text_features.mean(dim=0))
                fused_abnormal_sim = torch.einsum('nic,c->ni', fused_feature_map, abnormal_text_features.mean(dim=0))
                
                # 期望融合特征与正常文本更相似，与异常文本差异更大
                fusion_discriminative_loss = F.relu(fused_abnormal_sim - fused_normal_sim + 0.1).mean()
                
                # 总融合损失，使用可配置的权重
                fusion_loss = (args.fusion_consistency_weight * fusion_consistency_loss + 
                              args.fusion_discriminative_weight * fusion_discriminative_loss)
                
                # 使用融合特征进行后续计算
                feature_map = fused_feature_map

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nic,cj->nij', feature_map, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', feature_map, abnormal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            target_v2t = torch.zeros([logits_v2t.shape[0], logits_v2t.shape[1]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t.transpose(1, 2), target_v2t)

            trip_loss = criterion_tip(feature_map, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1 + args.fusion_loss_weight * fusion_loss

            loss.backward()
            optimizer.step()
            
            # 统计损失
            batch_count += 1
            epoch_loss += loss.item()
            epoch_v2t_loss += loss_v2t.item()
            epoch_trip_loss += trip_loss.item()
            epoch_fusion_loss += fusion_loss if isinstance(fusion_loss, (int, float)) else fusion_loss.item()
            
            # 更新进度条
            avg_loss = epoch_loss / batch_count
            train_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'V2T': f'{epoch_v2t_loss/batch_count:.4f}',
                'Trip': f'{epoch_trip_loss/batch_count:.4f}',
                'Fusion': f'{epoch_fusion_loss/batch_count:.4f}'
            })

        scheduler.step()
        
        # 打印epoch训练统计
        print(f"\n训练完成 - 平均损失: {epoch_loss/batch_count:.4f} | "
              f"V2T损失: {epoch_v2t_loss/batch_count:.4f} | "
              f"Triplet损失: {epoch_trip_loss/batch_count:.4f} | "
              f"融合损失: {epoch_fusion_loss/batch_count:.4f}")
        
        model.build_text_feature_gallery()

        # 评估阶段
        print("开始评估...")
        model.eval()
        score_maps = []
        test_imgs = []
        gt_mask_list = []
        names = []

        # 使用tqdm显示评估进度
        eval_pbar = tqdm(dataloader, desc=f"评估 Epoch {epoch+1}", leave=False)
        
        for (data, mask, label, name, img_type) in eval_pbar:

            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m in zip(data, name, label, mask):
                test_imgs += [denormalization(d.cpu().numpy())]
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_mask_list += [m]

            data = data.to(device)
            score_map = model(data, 'seg')
            score_maps += score_map

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)
        
        # 打印当前epoch的评估结果
        current_auroc = result_dict['p_roc']
        print(f"当前AUROC: {current_auroc:.4f}")

        if best_result_dict is None:
            best_result_dict = result_dict
            save_check_point(model, check_path)
            print(f"✅ 保存最佳模型 (首次评估) - AUROC: {current_auroc:.4f}")
            if args.vis:
                plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)

        elif best_result_dict['p_roc'] < result_dict['p_roc']:
            improvement = current_auroc - best_result_dict['p_roc']
            best_result_dict = result_dict
            save_check_point(model, check_path)
            print(f"🎉 发现更好模型！AUROC: {current_auroc:.4f} (提升: +{improvement:.4f})")
            if args.vis:
                plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)
        else:
            best_auroc = best_result_dict['p_roc']
            print(f"当前结果未超过最佳 - 最佳AUROC: {best_auroc:.4f}")
    
    # 训练完成总结
    final_auroc = best_result_dict['p_roc']
    print(f"\n🏁 训练完成！")
    print(f"📊 最终最佳AUROC: {final_auroc:.4f}")
    print(f"📁 模型已保存至: {check_path}")

    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path, train_data=train_dataloader)

    p_roc = round(metrics['p_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=1)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=4) #100

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    
    # fusion related parameters
    parser.add_argument("--enable-fusion", type=str2bool, default=True)
    parser.add_argument("--fusion-loss-weight", type=float, default=0.1)
    parser.add_argument("--fusion-consistency-weight", type=float, default=1.0)
    parser.add_argument("--fusion-discriminative-weight", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)

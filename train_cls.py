import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils_ori import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
import random
from tqdm import tqdm

TASK = 'CLS'


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
    print("==== build image feature gallery done ====")

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            # data = data[0:1, :, :, :].to(device)
            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)
            # n_text_pooled, n_text_tokens, n_text_features_1, n_text_features_2 = model.model.text(normal_text_prompt)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            cls_feature, _, _, _ = model.encode_image(data)

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

            # 修复拼写错误并添加指数化操作
            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half().exp().to(device)
            else:
                logit_scale = model.model.logit_scale.exp().to(device) 

            # 确保所有计算都发生在 float32
            # 即使 model.precision == 'fp16', 也强制所有输出为 float32
            cls_feature = cls_feature.float()
            normal_text_features_ahchor = normal_text_features_ahchor.float()
            abnormal_text_features = abnormal_text_features.float()
            abnormal_text_features_ahchor = abnormal_text_features_ahchor.float()

            logits_v2t = (torch.cat([l_pos.float(), l_neg_v2t.float()], dim=-1) * logit_scale.float()).float() # 添加.float()确保最终类型为float32

            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t, target_v2t)

            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1

            loss.backward()
            optimizer.step()

            print(f"label: {args.class_name}, epoch: {epoch}, loss: {loss.item()}")
        scheduler.step()
        model.build_text_feature_gallery()

        scores_img = []
        score_maps = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []

        for (data, mask, label, name, img_type) in dataloader:

            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m in zip(data, name, label, mask):
                test_imgs += [denormalization(d.cpu().numpy())]
                l = l.numpy()
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_list += [l]
                gt_mask_list += [m]

            data = data.to(device)
            score_img, score_map = model(data, 'cls')
            score_maps += score_map
            scores_img += score_img

        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))
        
        # 如果启用了分类指标，计算额外的分类指标
        if args.use_classification_metrics:
            from utils.metrics import metric_cal_img_classification
            classification_result_dict = metric_cal_img_classification(np.array(scores_img), gt_list, np.array(score_maps))
            
            # 打印所有指标
            print(f"Epoch {epoch} - Classification Metrics:")
            print(f"  ROC AUC: {classification_result_dict['i_roc']:.2f}%")
            print(f"  Accuracy: {classification_result_dict['accuracy']:.2f}%")
            print(f"  Precision: {classification_result_dict['precision']:.2f}%")
            print(f"  Recall: {classification_result_dict['recall']:.2f}%")
            print(f"  F1-Score: {classification_result_dict['f1_score']:.2f}%")
            print(f"  PR AUC: {classification_result_dict['pr_auc']:.2f}%")
            print(f"  Best Threshold: {classification_result_dict['best_threshold']:.4f}")
            print(f"  TP:{classification_result_dict['tp']}, FP:{classification_result_dict['fp']}, TN:{classification_result_dict['tn']}, FN:{classification_result_dict['fn']}")
            
            # 将分类指标合并到result_dict中
            result_dict.update(classification_result_dict)

        if best_result_dict is None:
            save_check_point(model, check_path)
            best_result_dict = result_dict

        # 根据配置选择不同的指标作为模型选择标准
        if args.model_selection_metric == 'roc_auc':
            # 使用ROC AUC (原始方式)
            should_save = best_result_dict['i_roc'] < result_dict['i_roc']
        elif args.model_selection_metric == 'f1_score' and args.use_classification_metrics:
            # 使用F1-Score作为标准
            should_save = best_result_dict.get('f1_score', 0) < result_dict.get('f1_score', 0)
        elif args.model_selection_metric == 'accuracy' and args.use_classification_metrics:
            # 使用Accuracy作为标准
            should_save = best_result_dict.get('accuracy', 0) < result_dict.get('accuracy', 0)
        else:
            # 默认使用ROC AUC
            should_save = best_result_dict['i_roc'] < result_dict['i_roc']
            
        if should_save:
            save_check_point(model, check_path)
            best_result_dict = result_dict

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
        # device = f"mps"
    kwargs['device'] = device

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    # 全部转为 float
    # model = model.float()

    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    # 输出最终结果
    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}')
    
    # 如果启用了分类指标，输出详细的分类结果
    if kwargs.get('use_classification_metrics', False):
        print(f'\n============= Detailed Classification Results for {object} =============')
        print(f'ROC AUC: {metrics["i_roc"]:.2f}%')
        if 'accuracy' in metrics:
            print(f'Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'Precision: {metrics["precision"]:.2f}%')
            print(f'Recall: {metrics["recall"]:.2f}%')
            print(f'F1-Score: {metrics["f1_score"]:.2f}%')
            print(f'PR AUC: {metrics["pr_auc"]:.2f}%')
            print(f'Best Threshold: {metrics["best_threshold"]:.4f}')
            print(f'Confusion Matrix - TP:{metrics["tp"]}, FP:{metrics["fp"]}, TN:{metrics["tn"]}, FN:{metrics["fn"]}')
        print('=' * 70)
    print()

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
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
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

    parser.add_argument("--use-cpu", type=int, default=1)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=10)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)

    # classification metrics options
    parser.add_argument("--use-classification-metrics", type=str2bool, default=False,
                        help="Enable detailed classification metrics (Accuracy, Precision, Recall, F1-Score, PR AUC)")
    parser.add_argument("--model-selection-metric", type=str, default="roc_auc",
                        choices=['roc_auc', 'f1_score', 'accuracy'],
                        help="Metric to use for model selection and checkpoint saving")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
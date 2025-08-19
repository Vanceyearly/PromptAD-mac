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
import random
from tqdm import tqdm  # 确保 tqdm 已导入

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

def save_image(img_tensor, path):
    img = img_tensor.squeeze(0).cpu().numpy() 
    img = (img * 255).astype(np.uint8)
    img = img[:, :, ::-1]  # RGB to BGR for OpenCV
    cv2.imwrite(path, img)

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
    for (data, mask, label, name, img_type, perlin_data, cutpaste_data) in tqdm(train_data, desc="Building feature gallery"):
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
    criterion_ab = nn.CrossEntropyLoss().to(device)
    criterion_tip_ab = TripletLoss(margin=0.0)

    best_result_dict = None
    epoch_progress = tqdm(range(args.Epoch), desc=f"Epoch 0/{args.Epoch}", position=0)
    for cur_epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {cur_epoch + 1}/{args.Epoch}")
        data_progress = tqdm(train_data, desc=f"Epoch {cur_epoch + 1}/{args.Epoch}", position=1, leave=False)
        for (data, mask, label, name, img_type, perlin_data_list, aug_mask_list) in data_progress:
            aa = 1

            if False:
                # 保存图片
                save_dir = './saved_images_1'
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                # for i, (d, pl, n) in enumerate(zip(data, perlin_data_list, name)):
                    
                save_image(data, os.path.join(save_dir, f'{name[0]}_original.png'))
                save_image(perlin_data_list[0], os.path.join(save_dir, f'{name[0]}_slightly.png'))
                save_image(perlin_data_list[1], os.path.join(save_dir, f'{name[0]}_medium.png'))
                save_image(perlin_data_list[2], os.path.join(save_dir, f'{name[0]}_significantly.png'))

                print(f"Original image saved to {os.path.join(save_dir, f'{name[0]}_original.png')}")
                print(f"Slightly augmented image saved to {os.path.join(save_dir, f'{name[0]}_slightly.png')}")
                print(f"Medium augmented image saved to {os.path.join(save_dir, f'{name[0]}_medium.png')}")
                print(f"Significantly augmented image saved to {os.path.join(save_dir, f'{name[0]}_significantly.png')}")

                for i, aug_mask in enumerate(aug_mask_list):
                    save_image(aug_mask, os.path.join(save_dir, f'{name[0]}_aug_mask_{i}.png'))
            
                bb = 0

            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)
            
            # data = data[0:1, :, :, :].to(device)
            data = data.to(device)

            pl_data_list = []
            for perlin_data in perlin_data_list:
                pl_data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in perlin_data]
                pl_data = torch.stack(pl_data, dim=0).to(device)
                pl_data = pl_data.to(device)
                pl_data_list.append(pl_data)

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

            cls_feature, _, _, _ = model.encode_image(data)
            # pl_cls_feature, _, _, _ = model.encode_image(pl_data)
            # cp_cls_feature, _, _, _ = model.encode_image(cp_data)
            pl_cls_feature_list = []
            for pl_data in pl_data_list:
                pl_cls_feature, _, _, _ = model.encode_image(pl_data)
                pl_cls_feature_list.append(pl_cls_feature)

            # ab_cls_feature = torch.cat([pl_cls_feature, cp_cls_feature], dim=0)
            # alpha = args.a
            # ab_cls_feature = (1 - alpha) * pl_cls_feature + alpha * cp_cls_feature
            
            # ab_cls_feature = torch.cat(pl_cls_feature_list, dim=0)
            ab_cls_feature = 1/3 * (pl_cls_feature_list[0] + pl_cls_feature_list[1] + pl_cls_feature_list[2])

            # ab_cls_feature = 0.1 * pl_cls_feature_list[0] + 0.3 * pl_cls_feature_list[1] + 0.6 * pl_cls_feature_list[2]

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

            l_pos_ab = torch.einsum('nc,cm->nm', ab_cls_feature, abnormal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t_ab = torch.einsum('nc,cm->nm', ab_cls_feature, normal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)
            loss_v2t = criterion(logits_v2t, target_v2t)

            logits_v2t_ab = torch.cat([l_pos_ab, l_neg_v2t_ab], dim=-1) * logit_scale
            target_v2t_ab = torch.zeros([logits_v2t_ab.shape[0]], dtype=torch.long).to(device)
            loss_v2t_ab = criterion_ab(logits_v2t_ab, target_v2t_ab)

            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            trip_loss_ab = criterion_tip_ab(ab_cls_feature, abnormal_text_features_ahchor, normal_text_features_ahchor)

            # loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1 + trip_loss_ab + loss_v2t_ab
            # loss = loss_match_abnormal * args.lambda1 + trip_loss_ab + loss_v2t_ab
            loss = trip_loss + loss_match_abnormal * args.lambda1
            ## 保留4位小数
            print(f"loss: {loss.item():.4f}, loss_v2t: {loss_v2t:.4f}, trip_loss: {trip_loss:.4f}, "
                  f"loss_match_abnormal: {loss_match_abnormal:.4f}, trip_loss_ab: {trip_loss_ab:.4f}, "
                  f"loss_v2t_ab: {loss_v2t_ab:.4f}")

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()

        scores_img = []
        score_maps = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []

        for (data, mask, label, name, img_type, perlin_data, cutpaste_data) in dataloader:
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

        if best_result_dict is None:
            save_check_point(model, check_path)
            best_result_dict = result_dict

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
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
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec_dtd', choices=['mvtec', 'visa', 'mvtec_dtd'])
    parser.add_argument('--class_name', type=str, default='wood')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=4) #400
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
    parser.add_argument("--Epoch", type=int, default=4) # 100

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002) #0.002
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001) #0.001

    # a超参数
    parser.add_argument("--a", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)

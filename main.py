import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import argparse
import os
import random
from tqdm import tqdm

import math
import models   
import datasets
import utils
from utils import depth_ext_transforms as det
from utils.visualizer import Visualizer
from metrics import StreamSegMetrics
from PIL import Image


def get_argparser():
    parser = argparse.ArgumentParser()
    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str,
                        default='voc', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=256,
                        help="num classes (default: None)")

    # Model Option
    parser.add_argument("--model", type=str,
                        default='deeplabv3plus_mobilenet', help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--show_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=41e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--drop_last", action='store_true', default=False,
                        help='whether to drop last batch')

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warmup',
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')

    parser.add_argument("--train_crop_size", type=int, default=448)

    parser.add_argument("--val_crop_size", type=int, default=448)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training",
                        action='store_true', default=False)

    parser.add_argument("--loss_type", type=str,
                        default='cross_entropy', help="loss type (default: False)")
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--random_seed", type=int, default=114514,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--enable_multi_scale_test", action='store_true', default=False,
                        help='whether to enable multi scale test')
    parser.add_argument("--enable_apex", action='store_true', default=False,
                        help='whether to add depth image')
    parser.add_argument("--label_smoothing", action='store_true', default=False,
                        help='whether to use label smoothing')


    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):

    train_transform = det.ExtCompose([
        det.ExtRandomScale((0.5, 2.0)),
        det.ExtRandomCrop(
            size=(opts.train_crop_size, opts.train_crop_size), pad_if_needed=True),
        det.ExtRandomHorizontalFlip(),
        det.ExtToTensor(),
        det.ExtNormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:

        val_transform = det.ExtCompose([
            det.ExtResize((464, 464)),
            det.ExtScale(1.0),
            det.ExtToTensor(),
            det.ExtNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = det.ExtCompose([
            det.ExtPass(),
            det.ExtScale(1.0),
            det.ExtToTensor(),
            det.ExtNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

    if(opts.dataset == 'nyuv2'):
        train_dst = datasets.Nyuv2(opts,
                                   split='train', transform=train_transform)
        val_dst = datasets.Nyuv2(opts,
                                 split='val', transform=val_transform)
    elif(opts.dataset == 'sunrgbd'):
        train_dst = datasets.Sunrgbd(opts,
                                     split='train', transform=train_transform)
        val_dst = datasets.Sunrgbd(opts,
                                   split='val', transform=val_transform)
    return train_dst, val_dst


def getModel(opts):
    if(opts.model == 'fafnet'):
        model = models.GetFAFNet(opts.num_classes, True)
    elif(opts.model == 'faf_d'):
        model = models.GetFAFNet_D(opts.num_classes, True)
    else:
        raise AssertionError("No matching models!")
    return model


def getLoss(opts):

    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(
            ignore_index=opts.ignore_index, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(
            ignore_index=opts.ignore_index, reduction="mean")
    elif opts.loss_type == 'ce_aux':
        criterion = utils.FSAuxCELoss(
            ignore_index=opts.ignore_index)
    else:
        raise AssertionError("No matching loss!")
    return criterion


def getOptimizer(opts, model):

    params = model.get_params()
    optimizer = torch.optim.SGD(params=[
        {'params': params[0], 'lr': opts.lr},
        {'params': params[1], 'lr': opts.lr}
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'cos':
        def lambda1(epoch): return (epoch / 1000) if epoch < 1000 else 0.5 * \
            (math.cos((epoch - 1000)/(opts.total_itrs - 1000) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return scheduler, optimizer


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()

    if opts.show_val_results:

        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader)):
            images, depths, labels = data
            b, c, h, w = images[0].shape
            multi_avg = torch.zeros(
                (b, opts.num_classes, h, w), dtype=torch.float32).to(device)

            for i in range(len(images)):
                '''multi-scale-test'''
                in_image = images[i].to(device, dtype=torch.float32)
                in_depth = depths[i].to(device, dtype=torch.float32)

                outputs = model(in_image, in_depth)
                if (isinstance(outputs, tuple)):
                    outputs = torch.nn.functional.interpolate(
                        outputs[0], size=(h, w), mode='bilinear', align_corners=True)
                else:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=(h, w), mode='bilinear', align_corners=True)
                multi_avg = multi_avg+outputs

            multi_avg = multi_avg / len(images)
            preds = torch.argmax(
                multi_avg, dim=1).cpu().numpy().astype(np.uint8)

            targets = labels[0].cpu().numpy()

            metrics.update(targets, preds)

            if opts.show_val_results:

                for i in range(len(targets)):
                    image = images[0][i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1,
                                                            2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(
                        target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(
                        'results/%d_image.png' % img_id)
                    Image.fromarray(target).save(
                        'results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                    Image.blend(Image.fromarray(image), Image.fromarray((pred)), 0.5).save(
                        'results/%d_blend.png' % img_id)

                    img_id += 1
        score = metrics.get_results()
    return score


def main():
    torch.backends.cudnn.benchmark = True
    opts = get_argparser().parse_args()
    if(opts.enable_apex):
        from torch.cuda.amp import autocast as autocast
        from torch.cuda.amp import GradScaler as GradeScaler
        scaler = GradeScaler()
    if opts.dataset.lower() == 'nyuv2':
        if opts.num_classes == 256:
            opts.num_classes = 40
    elif opts.dataset.lower() == 'sunrgbd':
        if opts.num_classes == 256:
            opts.num_classes = 37
    else:
        opts.num_classes = 3

    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    train_dst, val_dst = get_dataset(opts)
    train_loader = torch.utils.data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=opts.drop_last)
    val_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4, drop_last=False)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    model = getModel(opts)

    metrics = StreamSegMetrics(opts.num_classes)
    scheduler, optimizer = getOptimizer(opts, model)
    criterion = getLoss(opts)

    def save_ckpt(path):
    
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!]Retrain")
        model.to(device)

    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    if opts.test_only:

        model.eval()
        val_score= validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        if vis is not None:
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for data in tqdm(train_loader):
            cur_itrs += 1
            images, depth, labels = data
            images = images[0].to(device, dtype=torch.float32)
            depth = depth[0].to(device, dtype=torch.float32)
            labels = labels[0].to(device, dtype=torch.long)
            optimizer.zero_grad()
            if(opts.enable_apex):

                with autocast():
                    outputs = model(images, depth)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images, depth)
                loss = criterion(outputs, labels)
            
            if(opts.enable_apex):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
            else:
                loss.backward()
                optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Total Loss', cur_itrs, np_loss)
                vis.vis_scalar("Loss Curve", cur_itrs,
                               optimizer.state_dict()['param_groups'][0]['lr'])

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10

                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:

                save_ckpt('checkpoints/latest_%s_%s.pth' %
                          (opts.model, opts.dataset))
                print("validation...")
                model.eval()
                val_score= validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  

                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s.pth' %
                              (opts.model, opts.dataset))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc",
                                   cur_itrs, val_score['Overall Acc'])

                    vis.vis_scalar("[Val] Mean IoU", cur_itrs,
                                   val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                model.train()
            scheduler.step()

            if(opts.enable_apex):
                scaler.update()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()

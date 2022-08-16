import os
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import visdom
from PIL import Image, ImageFilter
import cv2
import torch
import utils.depth_ext_transforms as det


def create():
    pal_root = os.getcwd() + '/datasets/palette.txt'
    f = open(pal_root, 'r')
    impalette = [int(i) for i in f.read().split()]
    cmap = [[impalette[3 * i], impalette[3 * i + 1], impalette[3 * i + 2]]
            for i in range(int(len(impalette) / 3))]
    cm2lbl = np.zeros(256**3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(cmap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引
    return cm2lbl, cmap


class Sunrgbd(data.Dataset):

    cm2lbl, cmap = create()

    def __init__(self, opts, split='train', transform=None):
        self.opts = opts
        self.root = self.opts.data_root
        self.transform = transform
        self.split = split
        if split == "train":
            self.f = open(self.root + '/train.txt', 'r')
            self.indexes = self.f.read().split()
            self.istrain = True
        else:
            self.f = open(self.root + '/test.txt', 'r')
            self.indexes = self.f.read().split()
            self.istrain = False
        self.num_labels = 38
        self.ignore_label = 0

    def binalize(self, img):
        img = img.filter(ImageFilter.FIND_EDGES)
        img = img.filter(ImageFilter.SHARPEN)

        def fn(x):
            return 255 if x > 40 else 0

        img = img.point(fn, '1')
        return img

    def image2label(self, im, cm2lbl):  # 输入是(h,w,c)
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵

    def label2image(self, pred, cmap):
        colormap = np.array(cmap, dtype='uint8')
        X = pred
        return colormap[X, :]  # 输出的是(3, h, w)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        index = self.indexes[index]
        rgb_image = Image.open(
            os.path.join(self.root, "rgb",
                         str(index) + ".jpg"))
        depth_image = Image.open(
            os.path.join(self.root, "hha",
                         str(index) + ".png"))
        '''depth_image = cv2.imread(
            os.path.join(self.root, "depth",
                         str(index) + ".png"))'''
        mask = Image.open(os.path.join(self.root, "label",
                                       str(index) + ".png"))
        rgb_images = []
        depth_images = []
        masks = []
        if (self.split == 'train'):
            tmp_rgb_image, tmp_depth_image, tmp_mask = self.transform(
                rgb_image, depth_image, mask)
            rgb_images.append(tmp_rgb_image)
            depth_images.append(tmp_depth_image)
            masks.append(tmp_mask)
            return rgb_images, depth_images, masks
        else:
            if (self.opts.enable_multi_scale_test):
                # for rate in [1.0,0.5,1.25]:#shuffle 505:48.17%
                # for rate in [1.0,0.5,1.5]:#48.5% 505:49.3% shuffle 505:48.0%
                # for rate in [1.0,0.5,1.75]:#48.5% 505:49.6% shuffle 505:48.0%
                # for rate in [1.0,0.75,1.5]:#48.4% 505:49.3% shuffle 505:47.5%
                for rate in [1.0,0.5,1.25,1.75]:#48.8% shuffle 505:48.22%
                # for rate in [1.0,0.5,1.5,1.75]:#shuffle 505:47.9%
                # for rate in [1.0,0.5,1.25,1.5]: #shuffle 505:48.21%
                # for rate in [1.0,0.5,0.75,1.25,1.5]: #shuffle 505:48.12%
                # for rate in [1.0,0.75,1.25,1.75]:#shuffle 505:47.68%
                # for rate in [1.0,0.5,1.25,1.5,1.75]:#shuffle 505:48.1%
                # for rate in [1.0,0.5,0.75,1.25,1.75]:#shuffle 505:48.21%
                # for rate in [1.0,0.5, 0.75, 1.25,1.5,1.75]:#48.8% 505:50.0% shuffle 505:48.16%
                    self.transform.transforms[1].scale = rate
                    tmp_rgb_image, tmp_depth_image, tmp_mask = self.transform(
                        rgb_image, depth_image, mask)
                    rgb_images.append(tmp_rgb_image)
                    depth_images.append(tmp_depth_image)
                    masks.append(tmp_mask)
                return rgb_images, depth_images, masks
            else:
                rgb_image, depth_image, mask = self.transform(
                    rgb_image, depth_image, mask)
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
                masks.append(mask)
                return rgb_images, depth_images, masks

    def __len__(self):
        if (self.istrain):
            return len(self.indexes)
        else:
            return len(self.indexes)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.label2image(cls, mask, cls.cmap)


class option(object):
    def __init__(self, op1=True, op2=False):
        super(option, self).__init__()
        self.data_root = './nyudv2_final'
        self.enable_multi_scale_test = op1
        self.add_edge = op2


if __name__ == '__main__':
    vis = visdom.Visdom(env='test')
    opts = option(True,True)
    train_transform = det.ExtCompose([
        # det.ExtResize(size=opts.crop_size),
        det.ExtRandomScale((0.5, 2.0)),
        det.ExtRandomCrop(size=(480, 480), pad_if_needed=True),
        det.ExtRandomHorizontalFlip(),
        # added an random vertical flip
        # det.ExtRandomVerticalFlip(),
        det.ExtToTensor(),
        det.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
    ])

    val_transform = det.ExtCompose([
        det.ExtCenterCrop(480),
        det.ExtScale(1.0),
        det.ExtToTensor(),
        det.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225]),
    ])
    s = Nyuv2(opts, 'train', transform=train_transform)
    print('amd yes!')
    for index, data in enumerate(s):
        if (opts.add_edge):
            rgb_images, depth_images, masks, edges = data
        else:
            rgb_images, depth_images, masks = data
        for i in range(len(rgb_images)):
            vis.images(rgb_images[i], win='rgb %s ' % i)
            vis.images(depth_images[i], win='depth %s' % i)
            vis.images(masks[i], win='mask %s ' % i)
            if (opts.add_edge):
                vis.images(edges[i], win='edges %s' % i)

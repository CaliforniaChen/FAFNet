from sklearn .decomposition import PCA
import visdom
import torch
import torch.nn.functional as F
import cv2
import numpy as np

'''Visualize Flow Start'''

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - \
        np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC,
               2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - \
        np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM,
               0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (
            maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def draw_flow_arrow(img, flow, id, step=12, reverse=False, input_img=False):
    h, w = img.shape[:2]
    # print(img.shape)
    # print(flow.shape)
    # img = np.ones((h, w, 3)) * 255

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    fx, fy = fx * (id+1), fy*(id+1)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = img
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    # vis = img
    # cv2.arrowedLine(vis,(y,x),(y+fy,x+fx),(255,0,0) )
    # cv2.polylines(vis, lines, 0, (255, 0, 0))
    for (x1, y1), (x2, y2) in lines:
        if(reverse):
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (255, 255, 255),
                            line_type=cv2.LINE_4, thickness=1, tipLength=0.3)
        else:
            cv2.arrowedLine(vis, (x2, y2), (x1, y1), (255, 255, 255),
                            line_type=cv2.LINE_4, thickness=1, tipLength=0.3)

    if(input_img==False):
        R,G,B = cv2.split(vis)
        _,A1= cv2.threshold(R,254,255,0)
        _,A2= cv2.threshold(G,254,255,0)
        _,A3= cv2.threshold(B,254,255,0)
        A = np.minimum(A3,np.minimum(A1,A2))
        vis = cv2.merge([R,G,B,A])
    return vis


def VisualizeFlow(input_tensor, title='default', num=0, scale=1, id=0, input_img=None, reverse=False):
    out = input_tensor
    h, w = out.size()[-2:]
    out = F.interpolate(out, size=(480*scale, 640*scale),
                        # out = F.interpolate(out, size=(h*scale, w*scale),
                        mode='bilinear', align_corners=True)
    out = out[num]  # shape: [c,h,w]
    out = out.permute(2, 1, 0)  # shape:[w,h,c]
    out = out.cpu().numpy()
    img = flow_to_image(out)
    if(input_img is None):
        img2 = draw_flow_arrow(
            img, out, id=id, reverse=reverse, input_img=False)
    else:
        img2 = draw_flow_arrow(input_img, out, id=id,
                               reverse=reverse, input_img=True)
    img = img.swapaxes(0, 2)  # shape [ c,h,w]
    tmp = img2.swapaxes(0, 2)  # shape:[c,h,w]

    vis3.images([img, tmp[:-1]], win=title, opts=dict(
        title=title))
    return img2


'''Visualize Flow End'''

'''Visualize Feature Map PCA Start'''


def VisualizeFeatureMapPCA(input_tensor, title='default', num=0, scale=1):

    feature = input_tensor.data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    feature = feature[num]
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(
        1, 0).reshape(3, h, w).transpose(1, 2, 0)

    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(
        img_out_pca, (w, h), interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)
    img_out = img_out.transpose(2, 0, 1)
    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    vis3.image(img_out, win=title, opts=dict(
        title=title))
    return img_out


'''Visualize Feature Map PCA End'''

'''Visualize Feature Map PCA DIfferent Color Start'''


def VisualizeFeatureMapPCA_Alter(input_tensor, title='default', num=0, scale=1):

    feature = input_tensor.data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    feature = feature[num]
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(
        1, 0).reshape(3, h, w).transpose(1, 2, 0)
    img_out_pca = 255-img_out_pca
    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(
        img_out_pca, (w, h), interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)
    img_out = img_out.transpose(2, 0, 1)

    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    vis3.image(img_out, win=title, opts=dict(
        title=title))
    return img_out


'''Visualize Feature Map PCA End'''

vis3 = visdom.Visdom(env='visualize')


def VisualizeChannel(input_tensor, title='default', num=0, scale=1):
    out = input_tensor
    h, w = out.size()[-2:]
    out = F.interpolate(out, size=(h*scale, w*scale),
                        mode='bilinear', align_corners=True)
    vis3.images(torch.unsqueeze(out[num], dim=1), win=title, opts=dict(
        title=title))


def VisualizeSingleChannel(input_tensor, title="default", num=0, scale=1):
    out = input_tensor  # shape: [b,c,h,w]
    h, w = out.size()[-2:]
    out = F.interpolate(out, size=(h*scale, w*scale),
                        mode='bilinear', align_corners=True)
    out = out[num]  # shape : [c,h,w]
    out = torch.mean(out, dim=0)  # shape:[h,w]
    out = torch.unsqueeze(out, dim=0)
    # for i in range(input_tensor.size()[0]):
    vis3.image(out, win=title, opts=dict(
        title=title))

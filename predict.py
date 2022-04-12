import cv2
import torch
import argparse
import numpy as np

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description='Train the TransUNet on images and target masks')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()     


def model_load(
    model_path: str, 
    num_classes: int,  
    img_size: int = 224,
    n_skip: int = 3,
    vit_name: str = 'R50-ViT-B_16',
    vit_patches_size: int = 16):

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip

    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    # 模型加载
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    return net


def inference_single(image, net):
    input = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    return prediction


def model_detect(net, path: str, num_classes: int, img_size: int = 224):

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    picture = inference_single(img, net)

    h, w, _ = image.shape
    picture_cover = cv2.resize(picture, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # image = cv2.imread(path)
    # for i in range(1, num_class+1):
    #     image[:, :, 0][picture_cover == i] = int(i/num_class * 255)
    # if num_class == 4:
    #     name = 'crack'
    # else:
    #     name = 'irrsign'
    # cv2.imwrite('./prediction/unet' + name + '.jpg', image)

    labels = list()
    for i in np.unique(picture_cover):  # 51
        if i != 0:
            sel = picture_cover == i
            sel = sel.astype(np.uint8) * 255
            contours, _ = cv2.findContours(sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for contour in contours:
                labels.append([i, count, contour.tolist()])
                count += 1
    return labels


if __name__ == '__main__':
    args = get_args()
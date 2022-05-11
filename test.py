import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models.FSRCNN import FSRCNN
from tools.utils import convert_ycbcr_to_rgb, preprocess, calc_psnr
from tools.pytorch_ssim import ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--image-hr-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--color_channels', type=int, default=1, 
        help='1: 1 channel in YCrCb; 2: 3 channel in YCrCb; 3: 3 channel in RGB')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')


    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = pil_image.open(args.image_hr_file).convert('RGB')

    # hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr, _ = preprocess(lr, device)
    # hr, _ = preprocess(hr, device)
    hr = np.array(hr).astype(np.float32)
    hr = torch.from_numpy(hr).to(device)
    _, ycbcr = preprocess(bicubic, device)

    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)
    print('preds.shape: {}, hr.shape: {}'.format(preds.shape, hr.shape))
    psnr = calc_psnr(hr, preds)
    ssim_num = ssim(hr, preds)
    print('PSNR: {:.2f}'.format(psnr))
    print('SSIM: {:.4f}'.format(ssim_num))

    # preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    if args.color_channels == 1:
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    elif args.color_channels == 3:
        output = np.clip(preds*255, 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_model_x{}.'.format(args.scale)))


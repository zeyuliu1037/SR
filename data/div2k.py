from email.mime import image
from os import listdir
from os.path import join
from tkinter import Scale

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, color_channels):
    # print('img: ', img)
    if color_channels == 1:
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
        # print('y: ', y)
        return y
    elif color_channels == 2: # 
        img = Image.open(filepath).convert('YCbCr')
        return img
    else:
        img = Image.open(filepath).convert('RGB')
        return img

class DIV2KTrain(Dataset):
    def __init__(self, image_dir, scale=2, patch_size=32, color_channels=1):
        super(DIV2KTrain, self).__init__()
        train_image_dir = image_dir + '_LR_bicubic/X2'
        target_image_dir = image_dir + '_HR'
        train_image_list = sorted(listdir(train_image_dir))
        target_image_list = sorted(listdir(target_image_dir))
        self.train_image_filenames = [join(train_image_dir, x) for x in train_image_list if is_image_file(x)]
        self.target_image_filenames = [join(target_image_dir, x) for x in target_image_list if is_image_file(x)]
        self.scale = scale
        self.patch_size = patch_size
        self.color_channels = color_channels

        assert len(self.train_image_filenames) == len(self.target_image_filenames)
        # self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
        #                             #   transforms.Resize(crop_size//zoom_factor),  # subsampling the image (half size)
        #                             #   transforms.Resize(crop_size, interpolation=Image.BICUBIC),  # bicubic upsampling to get back the original size 
        #                               transforms.ToTensor()])
        # self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size*2), # since it's the target, we keep its original quality
        #                                transforms.ToTensor()])

    def __getitem__(self, index):
        input = load_img(self.train_image_filenames[index], self.color_channels)
        target = load_img(self.target_image_filenames[index], self.color_channels)

        w_lr, h_lr = input.size
        lr_top = random.randint(0, h_lr - self.patch_size)
        lr_left = random.randint(0, w_lr - self.patch_size)
        lr_box = (lr_top, lr_left, lr_top+self.patch_size, lr_left+self.patch_size)
        input = input.crop(lr_box)

        # print('lr_top: {}, scale: {}'.format(lr_top, self.scale))
        hr_top = int(lr_top * self.scale)
        hr_left = int(lr_left * self.scale)
        hr_box = (hr_top, hr_left, hr_top+self.patch_size*self.scale, hr_left+self.patch_size*self.scale)
        target = target.crop(hr_box)

        input = transforms.ToTensor()(input)
        target = transforms.ToTensor()(target)

        # input = input.filter(ImageFilter.GaussianBlur(1)) 
        # input = self.input_transform(input)
        # target = self.target_transform(target)

        return input, target

    def __len__(self):
        
        return len(self.train_image_filenames)

class DIV2KEval(Dataset):
    def __init__(self, image_dir, scale=2, color_channels=1):
        super(DIV2KEval, self).__init__()
        train_image_dir = image_dir + '_LR_bicubic/X2'
        target_image_dir = image_dir + '_HR'
        train_image_list = sorted(listdir(train_image_dir))
        target_image_list = sorted(listdir(target_image_dir))
        self.train_image_filenames = [join(train_image_dir, x) for x in train_image_list if is_image_file(x)]
        self.target_image_filenames = [join(target_image_dir, x) for x in target_image_list if is_image_file(x)]
        self.color_channels = color_channels
        # f = open("dataset/model_Weight.txt",'a')
        # for s in self.train_image_filenames:
        #     f.write(s)
        #     f.write(' ')
        # f.write('\n')
        # for s in self.target_image_filenames:
        #     f.write(s)
        #     f.write(' ')

        assert len(self.train_image_filenames) == len(self.target_image_filenames)

    def __getitem__(self, index):
        input = load_img(self.train_image_filenames[index], self.color_channels)
        target = load_img(self.target_image_filenames[index], self.color_channels)
        
        input = transforms.ToTensor()(input)
        target = transforms.ToTensor()(target)

        # print('indes: {}, input shape: {}, filename: {}'.format(index, input.shape, self.train_image_filenames[index]))
        # print('target shape: {}, filename: {}'.format(target.shape, self.target_image_filenames[index]))

        return input, target

    def __len__(self):
        
        return len(self.train_image_filenames)

if __name__ == '__main__':
    # test = DIV2KTrain('dataset/DIV2K_valid')
    test = DIV2KEval('dataset/DIV2K_valid')
    print(test)
    print(len(test))
    print(len(test[0]))
    print(test[0][0].shape)
    print(test[0][1].shape)
    # print(test[0][0][0][0])
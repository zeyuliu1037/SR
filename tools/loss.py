from tkinter import image_names
import torch
from torch import nn


class AddLoss(nn.Module):
    def __init__(self, add_loss):
        super(AddLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # vgg = vgg19(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        self.add_loss = add_loss
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Adversarial Loss
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # perception_loss = self.mdfloss(out_images, target_images)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        if not self.add_loss:
            return image_loss
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
import argparse
import os
import sys
import copy
from wsgiref import validate

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold

from models.FSRCNN import FSRCNN
from models.SRCNN import SRCNN
from models.ESPCN import ESPCN
from models.VDSR import VDSR
from models.EDSR import EDSR
from models.SRDN import SRDN
from models.REDN import REDN

from data.div2k import DIV2KTrain, DIV2KEval
from tools.utils import AverageMeter, calc_psnr
from tools.loss import AddLoss
import tools.pytorch_ssim as torch_ssim
from early_stop import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

def train(train_dataloader, val_dataloader, epoch, fold, best_epoch_psnr, Epoch_Counter):
    model.train()
    train_losses = AverageMeter()
    train_psnr = AverageMeter()
    train_ssim = AverageMeter()
    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print('inputs shape: {}, labels shape: {}'.format(inputs.shape, labels.shape))
            preds = model(inputs)
            # print('\ninputs shape: {}, preds shape: {}, labels shape: {}'.format(inputs.shape, preds.shape, labels.shape))
            # exit()
            loss = criterion(preds, labels)
            ssim = torch_ssim.ssim(preds, labels)
            # ssim_loss = torch_ssim.ssim(preds, labels)
            # total_loss = loss + ssim_loss

            train_losses.update(loss.item(), len(inputs))
            train_psnr.update(calc_psnr(preds, labels), len(inputs))
            train_ssim.update(ssim.item(), len(inputs))
            # ssim_losses.update(ssim_loss.item(), len(inputs))
            # total_losses.update(total_loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            # total_loss.backward()
            if args.model_name == 'VDSR':
                nn.utils.clip_grad_norm(model.parameters(),args.clip)
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(train_losses.avg))
            t.update(len(inputs))
    
    model.eval()
    val_psnr = AverageMeter()
    val_losses = AverageMeter()
    val_ssim = AverageMeter()
    for data in val_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        loss = criterion(preds, labels)
        ssim = torch_ssim.ssim(preds, labels)
        val_losses.update(loss.item(), len(inputs))
        val_psnr.update(calc_psnr(preds, labels), len(inputs))
        val_ssim.update(ssim.item(), len(inputs))

    f.write('\nfold: {}, epoch: {}, train loss: {:.4f}, train psnr: {:.2f}, train ssim: {:.4f}, val loss: {:.4f}, val psnr: {:.2f}, val ssim: {:.4f} '.format(
                    fold, epoch, train_losses.avg, train_psnr.avg, train_ssim.avg, val_losses.avg, val_psnr.avg, val_ssim.avg))
    writer.add_scalars('Loss', {
            'train': train_losses.avg,
            'val': val_losses.avg,
        }, Epoch_Counter)
    writer.add_scalars('PSNR', {
            'train': train_psnr.avg,
            'val': val_psnr.avg,
        }, Epoch_Counter)
    writer.add_scalars('SSIM', {
            'train': train_ssim.avg,
            'val': val_ssim.avg,
        }, Epoch_Counter)
    early_stopping(val_losses.avg, model)

    if val_psnr.avg > best_epoch_psnr[1]:
        best_epoch_psnr[0] = epoch
        best_epoch_psnr[1] = val_psnr.avg
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'best.pth'))
    
def test(eval_dataloader):
    model.eval()
    epoch_psnr = AverageMeter()
    test_losses = AverageMeter()
    ssims = AverageMeter()
    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        loss = criterion(preds, labels)
        ssim = torch_ssim.ssim(preds, labels)
        test_losses.update(loss.item(), len(inputs))
        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        ssims.update(ssim.item(), len(inputs))

    f.write('\nIn test set, test loss: {:.4f}, test psnr: {:.2f}, test ssim: {:.4f} '.format(test_losses.avg, epoch_psnr.avg, ssims.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='../dataset/DIV2K_train')
    parser.add_argument('--eval-file', type=str, default='../dataset/DIV2K_valid')
    parser.add_argument('--outputs-dir', type=str, default='output')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--model-name', type=str, default='FSRCNN')
    parser.add_argument('--clip', type=float, default=0.4)
    parser.add_argument('--color_channels', type=int, default=1)
    parser.add_argument('--add_loss', action='store_true')
    

    args = parser.parse_args()

    outputs_dir = os.path.join(args.outputs_dir, '{}_x{}_{}'.format(args.model_name, args.scale, datetime.datetime.now().strftime('%Y%m%d%H%M')
))

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if args.log:
        log_file = os.path.join(outputs_dir, '{}_{}.log'.format(args.model_name, args.color_channels))
        print(log_file)
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    f.write('\n\n Arguments:')
    for arg in vars(args):
        f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))

    if args.model_name == 'FSRCNN':
        model = FSRCNN(scale_factor=args.scale).to(device)
    elif args.model_name == 'SRCNN':
        model = SRCNN(color_channels=args.color_channels).to(device)
    elif args.model_name == 'ESPCN': 
        model = ESPCN(color_channels=args.color_channels, upscale_factor=args.scale).to(device)
    elif args.model_name == 'VDSR':
        model = VDSR().to(device)
    elif args.model_name == 'EDSR':
        model = EDSR(color_channels=args.color_channels).to(device)
    elif args.model_name == 'SRDN':
        model = SRDN(growth_rate=16, num_blocks=8, num_layers=8).to(device)
    elif args.model_name == 'REDN':
        model = REDN(scale_factor=args.scale, num_channels=1, \
        num_features=64, growth_rate=64, num_blocks=16, num_layers=8).to(device)
    f.write('\n {}'.format(model))

    if args.weights_file:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = AddLoss(args.add_loss)
    scheduler = None
    if args.model_name == 'FSRCNN':
        optimizer = optim.Adam([
            {'params': model.first_part.parameters()},
            {'params': model.mid_part.parameters()},
            {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)
    elif args.model_name == 'SRCNN':
        optimizer = optim.Adam(  # we use Adam instead of SGD like in the paper, because it's faster
        [
            {"params": model.conv1.parameters(), "lr": 0.0001},  
            {"params": model.conv2.parameters(), "lr": 0.0001},
            {"params": model.conv3.parameters(), "lr": 0.00001},
        ], lr=0.00001,)
    elif args.model_name == 'ESPCN':
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    elif args.model_name == 'VDSR':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[30, 80, 160], gamma=0.1)
    elif args.model_name == 'EDSR':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.5)
    elif args.model_name == 'SRDN':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.5)
    elif args.model_name == 'REDN':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    f.write('\n {}'.format(optimizer))

    train_dataset = DIV2KTrain(args.train_file, color_channels=args.color_channels)
    # train_dataloader = DataLoader(dataset=train_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.num_workers,
    #                               pin_memory=True)
    eval_dataset = DIV2KEval(args.eval_file, color_channels=args.color_channels)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # Define the K-fold Cross Validator
    k_folds = 3
    kfold = KFold(n_splits=k_folds, shuffle=True)

    early_stopping = EarlyStopping(patience=10, verbose=True,
        path=os.path.join(outputs_dir, 'early_stop.pth'), trace_func=f.write)
    best_epoch_psnr = [0, 0.0]
    tensorboard_log_name = 'runs' + outputs_dir
    writer = SummaryWriter(tensorboard_log_name)
    Epoch_Counter = 0
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        # print('train_subsampler: {}, val_subsampler: {}'.format(train_subsampler, val_subsampler))
        train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=0,
                                  sampler=train_subsampler,
                                  pin_memory=True)
        val_dataloader = DataLoader(dataset=train_dataset,
                                sampler=val_subsampler,
                                num_workers=0,
                                batch_size=1)

        for epoch in range(args.num_epochs):
            if not args.test_only:
                Epoch_Counter += 1
                train(train_dataloader, val_dataloader, epoch, fold, best_epoch_psnr, Epoch_Counter)
            test(eval_dataloader)
            if early_stopping.early_stop:
                early_stopping.counter = 0
                early_stopping.early_stop = False
                f.write('\nEarly stopping!')
                break
            if scheduler:
                scheduler.step()


    f.write('\nbest epoch: {}, psnr: {:.2f}'.format(best_epoch_psnr[0], best_epoch_psnr[1]))

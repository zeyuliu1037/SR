from models.EDSR import EDSR
import torch
if __name__ == '__main__': 
    from models import common
    net = EDSR()
    input = torch.rand(1,1,32,32)
    pred = net(input)
    print(pred.shape)